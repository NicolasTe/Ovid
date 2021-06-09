from configuration import Configuration
from argparse import Namespace
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import models.Ovid

import random as python_random


def run(config: Namespace):
    np.random.seed(42)
    python_random.seed(42)

    # load labels
    labels_df = pd.read_csv(config.labelPath, sep="\t").set_index("changeset")

    # create model
    model = models.Ovid.Ovid(config)

    features_df = model.compute_features()

    labels_features = labels_df.join(features_df, how="left").reset_index()
    labels_features = labels_features.fillna(0)

    y = labels_features["label"].to_numpy()
    X = labels_features.drop(["label", "changeset"], axis=1).to_numpy()

    # 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 10% validation, 70% train
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

    model.fit(X_train, y_train, X_val, y_val)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


def setup_config(config: Configuration):
    config.add_entry("lp", "labelPath", "Path of the label file")

    models.Ovid.add_config_entries(config)

    config.parse()


if __name__ == "__main__":
    config = Configuration()
    setup_config(config)

    for c in config.get_configs():
        run(c)

