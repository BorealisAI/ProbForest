# Status

This repository is fully operational on Ubuntu 19.10. On MacOSX only CPU functionality has been tested.

# Install ForestRelaxation package

```  
pip install prob_forest
```

# Development installation

```
conda env create --file env.yml
conda activate prob_forest
```


# Requirements

This package was developed using torch 1.3.1 and numpy 1.17.4 under python 3.7.5.

This package uses the latest development version of sklearn, which can be installed from source [here](https://github.com/scikit-learn/scikit-learn).

```pytest``` is required to run tests.

```pandas``` and ```tqdm``` required for certain scripts.

# Installation

```pip install -e .```

# Tests

```
make test
```

# Data

We use the same datasets examined in the following papers:

* [Robust Decision Trees Against Adversarial Examples](https://arxiv.org/abs/1902.10660)
* [Robustness Verification of Tree-based Models](https://arxiv.org/abs/1906.03849)

This [repo](https://github.com/chenhongge/RobustTrees) contains all the necessary scripts for downloading and processing the data.

# Naming convention:

There are two considerations, which are motivated by keeping to the sklearn API:

1. All parameter names are given in the singular (ex. 'split' instead of 'splits')
2. Snake case, with separate prefix (ex. n_feature instead of nfeature)


