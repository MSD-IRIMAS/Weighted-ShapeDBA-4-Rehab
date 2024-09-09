# Weighted-ShapeDBA-4-Rehab

This is the code of our paper "[Weighted Average of Human Motion Sequences for Improving Rehabilitation Assessment](hhttps://ecml-aaltd.github.io/aaltd2024/articles/Fawaz_AALTD24.pdf)" accepted at the [9th International Workshop on Advanced Analytics and Learning on Temporal Data](https://ecml-aaltd.github.io/aaltd2024/), at [ECML-PKDD 2024](https://2024.ecmlpkdd.org/)<br>
This work was done by [Ali Ismail-Fawaz](https://hadifawaz1999.github.io/), [Maxime Devanne](https://maxime-devanne.com/), [Stefano Berretti](http://www.micc.unifi.it/berretti/), [Jonathan Weber](https://www.jonathan-weber.eu/) and [Germain Forestier](https://germain-forestier.info/).<br><br>

## Requirements

```
tensorflow
numpy
hydra-core
aeon
scikit-learn
matplotlib
```

## Usage

### Create the extended dataset

This code utilizes the hydra configuration setup, simply edit the parameters of the configuration file ('config/config_hydra.yaml`) and run the following:

```
python main_data_extension.py
```

It will create the base dataset and extend it using a noisy version (baseline) as well as using the weighted shape DBA for various number of neighbors.
The different parameters include:

- `original_dir`: The directory where the original Kimore dataset is stored
- `res_directory`: The output directory where the results are stored
- `exercise`: The exercise to consider
- `num_folds`: The number of folds per exercise to consider
- `num_neighbors_wsdba`: The list of number of neighbors used for weighted shape DBA


### Train and evaluate a Regressor

This code utilizes the hydra configuration setup, simply edit the parameters of the configuration file ('config/config_hydra.yaml`) and run the following:

```
python main_data_extension.py
```

It creates and trains and avaluates a regressor for all version of the dataset (original + extended version). For now, only the FCN regressor is available.
The different parameters include:

- `res_directory`: The output directory where the results are stored
- `exercise`: The exercise to consider
- `num_folds`: The number of folds per exercise to consider
- `num_neighbors_wsdba`: The list of number of neighbors used for weighted shape DBA
- `nb_epochs`: Number of epochs to train the regressor
- `batch_size`: The batch size to use in the regressor

