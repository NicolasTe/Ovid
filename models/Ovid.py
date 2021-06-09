from sklearn.base import BaseEstimator

from configuration import Configuration
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, concatenate, Input, Activation, MultiHeadAttention, Reshape, \
    LayerNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import random as python_random
from ast import literal_eval
from dateutil.parser import parse

import tensorflow as tf
import numpy as np
import pandas as pd
import os.path
import pickle

from util import parse_csv_repeating_headers, parse_map_feature_list, \
    parse_timestamp_feature, geom_dist, finish_row


def add_config_entries(configuration: Configuration):
    configuration.add_model_entry("dynamicUserFeatures", "Path to user features file")
    configuration.add_model_entry("staticUserFeatures", "Path to user features file")
    configuration.add_model_entry("changesetFeatures", "Path to changeset features file")
    configuration.add_model_entry("comments", "Path to the comments file")
    configuration.add_model_entry("edits", "Path to the edits file")
    configuration.add_model_entry("currentObjectFeatures", "Path to the current object features file")
    configuration.add_model_entry("previousObjectFeatures", "Path to the current object features file")
    configuration.add_model_entry("mapFeatureList", "List of established key/value pairs")
    configuration.add_model_entry("objectVersionFeatures", "Object version features")

    configuration.add_model_entry("cachePath", "Path to cache precalculations")


# This class contains the Ovid model for vandalism detection in OpenStreetMap.
# This class takes care of feature preprocessing, while the OvidEstimator class contains the neural network.
class Ovid:
    def __init__(self, config: Configuration):
        self.config = config
        self.scaler = None
        self.clf = None
        self.map_feature_list = parse_map_feature_list(config.mapFeatureList)

        # dimensions
        self.no_changeset_features = None
        self.no_user_features = None
        self.no_edit_features = None

        self.max_edits = 20

    def store(self, path):
        print("Storing model to: " + path)
        with open(path + "_scaler", 'wb') as fo:
            pickle.dump((self.scaler, self.max_edits), fo, pickle.HIGHEST_PROTOCOL)
        self.clf.save(path + "_network")

    def load(self, path):
        with open(path + "_scaler", 'rb') as fi:
            self.scaler, self.max_edits = pickle.load(fi)

        self.clf = OvidEstimator(1, 1, 1)
        self.clf.load_model(path + "_network")

    def compute_features(self):
        # user features
        user_features = parse_csv_repeating_headers(self.config.dynamicUserFeatures,
                                                    sep="\t").set_index("target_changeset")

        top_tags = ["building", "source", "highway", "name", "natural", "surface",
                    "landuse", "power", "waterway", "amenity", "service", "oneway"]
        top_tags = ["create_" + x for x in top_tags]
        top_tags = user_features[top_tags].sum(axis=1)
        user_features["top_tags"] = top_tags

        user_features = user_features[["contributions",
                                       "deletes",
                                       "modifies",
                                       "creates",
                                       "active_weeks",
                                       "create_nodes",
                                       "create_ways",
                                       "create_relations",
                                       "top_tags"]]

        # user created
        static_user_features = pd.read_csv(self.config.staticUserFeatures,
                                           sep="\t",
                                           error_bad_lines=False).set_index("changeset")

        static_user_features["account_created"] = static_user_features["account_created"].apply(parse_timestamp_feature)

        user_features = user_features.join(static_user_features)

        self.no_user_features = user_features.shape[1]

        # changeset features
        changeset_features = pd.read_csv(self.config.changesetFeatures,
                                         sep="\t").set_index("changeset")

        feature_names = ["no_nodes",
                         "no_ways",
                         "no_relations",
                         "no_creates",
                         "no_modifications",
                         "no_deletions",
                         "num_changes",
                         "min_lat",
                         "min_lon",
                         "max_lat",
                         "max_lon",
                         "bbox_area",
                         "editor"]

        changeset_features = changeset_features[feature_names]

        changeset_features = changeset_features.join(pd.get_dummies(changeset_features["editor"]))
        del changeset_features["editor"]

        changeset_features = changeset_features.replace([np.inf], 0)

        self.no_changeset_features = changeset_features.shape[1]
        features = user_features.join(changeset_features)

        comments = pd.read_csv(self.config.comments, sep="\t", index_col=["changeset"])
        comments["comment"] = comments["comment"].fillna("")
        comments["comment_length"] = comments["comment"].str.len()
        comments = comments[["comment_length"]]

        self.no_changeset_features += comments.shape[1]
        features = features.join(comments)

        # edit features
        target_features = ["node",
                           "way",
                           "relation",
                           "create",
                           "delete",
                           "modify",
                           "n_users",
                           "time_to_previous",
                           "no_tags",
                           "valid_tags",
                           "valid_tags_po",
                           "geom_dist",
                           "tags_deleted",
                           "tags_added",
                           "name_changed"]

        edit_cache = os.path.join(self.config.cachePath, "ovid_edits.pkl")
        if os.path.isfile(edit_cache):
            edits_df = pd.read_pickle(edit_cache)
        else:
            edits_df = pd.read_csv(self.config.edits, sep="\t", index_col=["changeset", "type", "id", "version"])
            edits_df["timestamp"] = edits_df["timestamp"].apply(parse)

            operation_dummies = pd.get_dummies(edits_df["operation"])
            edits_df = edits_df.join(operation_dummies)

            dummies = pd.get_dummies(edits_df.index.get_level_values(1))
            dummies = dummies.set_index(edits_df.index)

            edits_df = edits_df.join(dummies)

            object_version = pd.read_csv(self.config.objectVersionFeatures,
                                         sep="\t").set_index(["changeset", "type", "id", "version"])
            edits_df = edits_df.join(object_version)

            current_object = pd.read_csv(self.config.currentObjectFeatures, sep="\t",
                                         index_col=["changeset", "type", "id", "version"])
            edits_df = edits_df.join(current_object, rsuffix='_current')

            prev_object = pd.read_csv(self.config.previousObjectFeatures,
                                      sep="\t",
                                      index_col=["changeset", "type", "id", "version"])
            edits_df = edits_df.join(prev_object, rsuffix='_po')

            edit_data = edits_df.reset_index()

            col_idx = {}
            for c in ["geometry", "geometry_po", "timestamp", "prev_timestamp", "tags", "tags_po"]:
                col_idx[c] = edit_data.columns.get_loc(c)

            pr = Parallel(n_jobs=-1)(
                delayed(self.process_edit_row)(i, col_idx) for i in
                tqdm(edit_data.to_numpy(), desc="Calculating edit rows"))

            cols = ["changeset",
                    "type",
                    "id",
                    "version",
                    "time_to_previous",
                    "no_tags",
                    "valid_tags",
                    "valid_tags_po",
                    "geom_dist",
                    "tags_deleted",
                    "tags_added",
                    "name_changed"]
            edit_row_features = pd.DataFrame(pr, columns=cols)
            edit_row_features.set_index(["changeset", "type", "id", "version"], inplace=True)

            edits_df = edits_df.join(edit_row_features)

            edits_df = edits_df[target_features]
            edits_df = edits_df.sort_index()

            if not os.path.isdir(self.config.cachePath):
                os.makedirs(self.config.cachePath)
            edits_df.to_pickle(edit_cache)

        current_changeset = -1
        edit_count = -1
        rows = []
        current_row = []
        current_mask_row = []
        for r in tqdm(edits_df.iterrows(), total=len(edits_df.index), desc="Computing edit features"):
            changeset = r[0][0]

            if changeset != current_changeset:
                if current_changeset != -1:
                    rows.append(
                        finish_row(current_row, edit_count, self.max_edits, len(target_features) + 1, current_mask_row))

                current_changeset = changeset
                edit_count = 0
                current_row = [current_changeset]
                current_mask_row = []

            # version
            current_row.append(r[0][3])

            # other features
            for current_feature in target_features:
                current_row.append(r[1][current_feature])

            current_mask_row.append(True)

            edit_count += 1
        rows.append(finish_row(current_row, edit_count, self.max_edits, len(target_features) + 1, current_mask_row))

        edit_cols = ["changeset"]
        for i in range(self.max_edits):
            for tf in ["version"] + target_features:
                edit_cols.append(str(i) + "_" + tf)

        for i in range(self.max_edits):
            edit_cols.append("mask_" + str(i))

        edit_features = pd.DataFrame(rows, columns=edit_cols).set_index("changeset")
        self.no_edit_features = edit_features.shape[1] - self.max_edits

        features = features.join(edit_features, how="left").fillna(0)

        return features

    def process_edit_row(self, row, col_idx):
        try:
            current_ts = row[col_idx["timestamp"]]
            prev_ts = row[col_idx["prev_timestamp"]]

            tags = row[col_idx["tags"]]
            tags_po = row[col_idx["tags_po"]]
            geometry = row[col_idx["geometry"]]
            geometry_po = row[col_idx["geometry_po"]]
            osm_type = row[1]

            result = list(row[:4])

            if type(prev_ts) == str:
                prev_ts = parse(prev_ts)
                diff = (current_ts - prev_ts).total_seconds()
                result.append(diff)
            else:
                result.append(0)

            parsed_tags = None
            if (type(tags) == str) and (tags != "deleted"):
                parsed_tags = literal_eval(tags)
                result.append(len(parsed_tags))
                result.append(self.no_valid_tags(parsed_tags))
            else:
                result.append(0)
                result.append(0)

            parsed_tags_po = None
            if (type(tags_po) == str) and (tags_po != "deleted"):

                parsed_tags_po = literal_eval(tags_po)
                result.append(self.no_valid_tags(parsed_tags_po))
            else:
                result.append(0)

            # tags deleted and tags added and name changed
            if (parsed_tags is not None) and (parsed_tags_po is not None):

                tags_deleted = 0
                for tp in parsed_tags_po:
                    if tp not in parsed_tags:
                        tags_deleted += 1
                result.append(tags_deleted)

                tags_added = 0
                for t in parsed_tags:
                    if t not in parsed_tags_po:
                        tags_added += 1
                result.append(tags_added)

                if "name" in parsed_tags and "name" in parsed_tags_po:
                    result.append(parsed_tags["name"] != parsed_tags_po["name"])
                else:
                    result.append(False)
            else:
                result.append(0)
                result.append(0)
                result.append(False)

            result.append(geom_dist(geometry, geometry_po, osm_type))
        except:
            raise
        return result

    def no_valid_tags(self, tags):
        valid_tags = 0
        for t in tags:
            if t in self.map_feature_list and tags[t] in self.map_feature_list[t]:
                valid_tags += 1
        return valid_tags

    def _get_input_parts(self, X):
        changeset_features_end = self.no_user_features + self.no_changeset_features
        edit_features_end = changeset_features_end + self.no_edit_features

        user_features = X[:, 0:self.no_user_features]
        changeset_features = X[:, self.no_user_features:changeset_features_end]

        edit_features = X[:, changeset_features_end:edit_features_end]

        return [changeset_features, user_features, edit_features]

    def get_edit_mask(self, X):
        edit_features_end = self.no_user_features + self.no_changeset_features + self.no_edit_features
        edit_mask_end = edit_features_end + self.max_edits
        edit_mask = X[:, edit_features_end:edit_mask_end]
        return edit_mask.astype('uint8')

    def fit(self, X, y, X_val, y_val):
        edit_mask = self.get_edit_mask(X)

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        val_edit_mask = self.get_edit_mask(X_val)
        X_val_scale = self.scaler.transform(X_val)

        val_data = (self._get_input_parts(X_val_scale) + [val_edit_mask], y_val)

        self.clf = OvidEstimator(no_changeset_features=self.no_changeset_features,
                                 no_user_features=self.no_user_features,
                                 no_edit_features=self.no_edit_features,
                                 max_edits=self.max_edits)

        self.clf.build_model()
        self.clf.fit(self._get_input_parts(X) + [edit_mask], y, val_data)

    def predict(self, X):
        edit_mask = self.get_edit_mask(X)
        X = self.scaler.transform(X)
        result = self.clf.predict(self._get_input_parts(X) + [edit_mask])
        result = result >= 0.5
        return result.reshape(-1, )

    def predict_proba(self, X):
        edit_mask = self.get_edit_mask(X)
        X = self.scaler.transform(X)
        result = self.clf.predict(self._get_input_parts(X) + [edit_mask])
        return result.reshape(-1, )


# This class contains the neural network of the Ovid model
class OvidEstimator(BaseEstimator):
    def __init__(self,
                 no_changeset_features,
                 no_user_features,
                 no_edit_features,
                 max_edits=10,
                 dropout=0.5,
                 patience=10,
                 n_fuse=1,
                 n_pred=1,
                 n_edit=1,
                 n_features=1,
                 num_heads=10,
                 hidden_pred=24):

        # parameters
        self.max_edits = max_edits
        self.hidden_pred = hidden_pred
        self.dropout = dropout
        self.patience = patience
        self.n_fuse = n_fuse
        self.n_pred = n_pred
        self.n_edit = n_edit
        self.num_heads = num_heads
        self.n_features = n_features

        # dimensions
        self.no_changeset_features = no_changeset_features
        self.no_user_features = no_user_features
        self.no_edit_features = no_edit_features

        # static properties
        self.activation = "relu"
        self.regularizer = 'l2'
        self.batch_size = 2000

    def build_model(self):
        # changeset features
        input_dropout = 0.0

        changeset_input = Input(shape=(self.no_changeset_features,), name="changeset_input")
        changeset_input_drp = Dropout(input_dropout)(changeset_input)

        ch_in = changeset_input_drp
        for i in range(self.n_features):
            changeset_dense = Dense(self.no_changeset_features + self.no_user_features,
                                    activation='linear',
                                    kernel_regularizer=self.regularizer,
                                    name="Changeset_dense_" + str(i))(ch_in)
            changeset_norm = LayerNormalization(name="Changeset_norm_" + str(i))(changeset_dense)
            changeset_act = Activation(self.activation, name="Changeset_act_" + str(i))(changeset_norm)
            changeset_drp = Dropout(self.dropout, name="Changeset_dropout_" + str(i))(changeset_act)

            ch_in = changeset_drp

        ch_out = ch_in

        # user features
        user_input = Input(shape=(self.no_user_features,), name="user_input")
        user_input_drp = Dropout(input_dropout)(user_input)

        user_in = user_input_drp
        for i in range(self.n_features):
            user_dense = Dense(self.no_changeset_features + self.no_user_features,
                               activation='linear',
                               kernel_regularizer=self.regularizer,
                               name="User_dense_" + str(i))(user_in)
            user_norm = LayerNormalization(name="User_norm_" + str(i))(user_dense)
            user_act = Activation(self.activation, name="User_act_" + str(i))(user_norm)
            user_drp = Dropout(self.dropout, name="User_dropout_" + str(i))(user_act)

            user_in = user_drp

        user_out = user_in

        cu_concat = concatenate([ch_out, user_out])

        # fuse cu-features
        fuse_in = cu_concat

        for i in range(self.n_fuse):
            fuse_dense = Dense(self.no_changeset_features + self.no_user_features,
                               activation='linear',
                               kernel_regularizer=self.regularizer,
                               name="Fuse_dense_" + str(i))(fuse_in)
            fuse_norm = LayerNormalization(name="Fuse_norm_" + str(i))(fuse_dense)
            fuse_act = Activation(self.activation, name="Fuse_act_" + str(i))(fuse_norm)
            fuse_drp = Dropout(self.dropout, name="Fuse_dropout_" + str(i))(fuse_act)

            fuse_in = fuse_drp

        fuse_out = fuse_in

        # edit features
        edit_input = Input(shape=(self.no_edit_features,), name="edit_input")
        edit_input_drp = Dropout(input_dropout)(edit_input)

        d_e = int(self.no_edit_features / self.max_edits)

        edit_matrix = Reshape((self.max_edits, d_e))(edit_input_drp)

        edit_preproc = Dense(d_e,
                             activation=self.activation,
                             kernel_regularizer=self.regularizer)(edit_matrix)

        edit_mask_input = Input(shape=(self.max_edits,), name="edit_mask_input")
        edit_mask = Reshape((1, self.max_edits))(edit_mask_input)

        cu_queries = Reshape((1, self.no_changeset_features + self.no_user_features))(fuse_out)

        edit_attention = MultiHeadAttention(num_heads=self.num_heads,
                                            key_dim=d_e,
                                            name="at_1",
                                            kernel_regularizer=self.regularizer,
                                            dropout=self.dropout)(cu_queries, edit_preproc, attention_mask=edit_mask)

        attention_features = Reshape((edit_attention.shape[2],))(edit_attention)

        edit_in = attention_features
        for i in range(self.n_edit):
            edit_dense = Dense(self.hidden_pred,
                               activation='linear',
                               kernel_regularizer=self.regularizer,
                               name="Edit_dense_" + str(i))(edit_in)
            edit_norm = LayerNormalization(name="Edit_norm_" + str(i))(edit_dense)
            edit_act = Activation(self.activation, name="Edit_act_" + str(i))(edit_norm)
            edit_drp = Dropout(self.dropout, name="Edit_dropout_" + str(i))(edit_act)

            edit_in = edit_drp

        edit_out = edit_in

        all_features_2 = concatenate([fuse_out, edit_out])

        # prediction
        pred_in = all_features_2

        for i in range(self.n_pred):
            pred_dense = Dense(self.hidden_pred,
                               activation='linear',
                               kernel_regularizer=self.regularizer,
                               name="Pred_dense_" + str(i))(pred_in)
            pred_norm = LayerNormalization(name="Pred_norm_" + str(i))(pred_dense)
            pred_act = Activation(self.activation, name="Pred_act_" + str(i))(pred_norm)
            pred_drp = Dropout(self.dropout, name="Pred_dropout_" + str(i))(pred_act)

            pred_in = pred_drp

        pred_out = pred_in

        prediction = Dense(1, activation='sigmoid', kernel_regularizer=self.regularizer)(pred_out)

        # optimization
        model = Model(inputs=[changeset_input, user_input, edit_input, edit_mask_input], outputs=prediction)

        adam = Adam()

        bc = BinaryCrossentropy(from_logits=False)

        model.compile(optimizer=adam, loss=bc)
        self.clf = model

    def predict(self, X):
        result = self.clf.predict(X)
        result = result >= 0.5
        return result.reshape(-1, )

    def predict_proba(self, X):
        result = self.clf.predict(X)
        return result.reshape(-1, )

    def fit(self, X, y, val_data):

        tf.random.set_seed(42)
        python_random.seed(42)
        np.random.seed(42)

        es = EarlyStopping(monitor='val_loss', patience=self.patience)

        self.clf.fit(X,
                     y,
                     batch_size=self.batch_size,
                     epochs=100,
                     validation_data=val_data,
                     callbacks=[es])

    def save(self, path):
        self.clf.save(path + ".h5")

        with open(path + "_parameters.pkl", 'wb') as fo:
            params = (
                self.max_edits,
                self.hidden_pred,
                self.dropout,
                self.patience,
                self.n_fuse,
                self.n_pred,
                self.n_edit,
                self.num_heads,
                self.n_features,
                self.no_changeset_features,
                self.no_user_features,
                self.no_user_features,
                self.no_edit_features)
            pickle.dump(params, fo, pickle.HIGHEST_PROTOCOL)

    def load_model(self, path):
        with open(path + "_parameters.pkl", 'rb') as fi:
            self.max_edits,
            self.hidden_pred,
            self.dropout,
            self.patience,
            self.n_fuse,
            self.n_pred,
            self.n_edit,
            self.num_heads,
            self.n_features,
            self.no_changeset_features,
            self.no_user_features,
            self.no_user_features,
            self.no_edit_features = pickle.load(fi)
        self.clf = load_model(path + ".h5")
