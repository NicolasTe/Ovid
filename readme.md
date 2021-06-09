# Ovid Repository
This repository contains the Ovid (OpenStreetMap Vandalism Detection) model source code and vandalism labels data extracted from OpenStreetMap (https://www.openstreetmap.org/).

The source code is available under the MIT license. 
The vandalism label data is available under the [Open Database License (ODbL)] (https://opendatacommons.org/licenses/odbl/summary/).

# Ovid Model
The Ovid model combines a set of selected features with a neural network architecture that adopts a multi-head attention mechanism to summarize information indicating vandalism from OpenStreetMap changesets effectively.

The source code is located in the Ovid.py file. We provide a sample of the training data in the training_sample directory. Due to the size restrictions on GitHub, we only include the training data for a subset of the labels. 
We provide the subset to illustrate the model configuration in the sample configuration file config.ini.

 The Ovid model can be run with the following command:
```
python3 main.py -c config.ini
```

# OpenStreetMap Vandalism Labels
We extracted changesets from the OSM history reported to constitute vandalism by considering reverts that mention vandalism in the OSM history.
We found 9,138 examples of vandalism in the OSM history. We create negative (non-vandalism) examples by random sampling changesets from the OSM history.

We provide the labels as a CSV file in the labels folder. The column "changeset" provides the changeset id. The column "vandalism" specifies if the changeset constitutes vandalism (True corresponds to vandalism).
