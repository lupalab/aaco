# Acquisition Conditioned Oracle for Nongreedy Active Feature Acquisition (AACO)

This repository contains the implementation of the Acquisition Conditioned Oracle (AACO), which was used in the [paper](https://proceedings.mlr.press/v235/valancius24a.html) "**Acquisition Conditioned Oracle for Nongreedy Active Feature Acquisition**" published at ICML 2024.

## Repository Overview

This repository contains the code used to implement and evaluate AACO. The main files are as follows:

- ``aaco_rollout.py``: Main script that runs the AACO on a given dataset and configuration.
- ``src/classifier.py``: Contains different classifier classes.
- ``src/mask_generator.py``: Defines the two mask generation strategies (an exhaustive search or random masks).
- ``config.yaml``: Configuration file used to specify the dataset, hyperparameters, and other settings.


## Requirements

The repository uses the following key libraries:

- ``numpy``
- ``torch``
- ``xgboost``
- ``yaml``


## Running the Code

To run the AACO, make sure that the configuration file (config.yaml) is properly set up. Then, execute the following command:

```bash
python src/aaco_rollout.py
```

Results will be saved in the ``results/ directory`` as a .pt file.



