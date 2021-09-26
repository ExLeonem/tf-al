

[![PyPI version](https://badge.fury.io/py/tf-al.svg)](https://badge.fury.io/py/tf-al)
[![PyPi license](https://badgen.net/pypi/license/tf-al/)](https://pypi.org/project/tf-al/)
![Python Version: ^3.6](https://img.shields.io/badge/python-%5E3.6-blue)
<a href="https://codeclimate.com/github/ExLeonem/tf-al/maintainability"><img src="https://api.codeclimate.com/v1/badges/50b2389c5a7ce33298be/maintainability" /></a>
[![Coverage Status](https://coveralls.io/repos/github/ExLeonem/tf-al/badge.svg?branch=master)](https://coveralls.io/github/ExLeonem/tf-al?branch=master)


# Active learning with tensorflow

<sup>*Currently only supports bayesian active learning tasks.</sup>

Perform active learning in tensorflow/keras with extendable parts. 



# Index

1. [Installation](#Installation)
2. [Full Documentation](https://exleonem.github.io/tf-al/)
3. [Model Wrapper](#Model-wrapper)
4. [Development](#Development)
    1. [Setup](#Setup)
    2. [Scripts](#Scripts)
5. [Contribution](#Contribution)
6. [Issues](#Issues)



# Dependencies

```toml
python="^3.6"
tensorflow="^2.0.0"
scikit-learn="^0.24.2"
numpy="^1.0.0"
tqdm="^4.62.6"
```

# Installation


```shell
$ pip install tf-al
```

<sup>*To use a specific version of tensorflow or if you want gpu support you should manually install tensorflow. Else this package automatically will install the lastest version of tensorflow described in the [dependencies](#Dependencies).</sup>



# Model wrapper

Currently there are 3 model wrapper implemented.

| Wrapper | Description
| --- | ---
| Model | Basic wrapper for regular deep learning networks
| McDropout | Wrapps MC Dropout models for bayesian inference.
| Ensemble | Wrapps an ensemble of models.


# Development


## Setup

1. Fork and clone the forked repository
2. Create a virtual env (optional)
3. [Install and Setup Poetry](https://python-poetry.org/docs/#installation)
4. Install package dependencies [using poetry](https://python-poetry.org/docs/cli/#install) or set them up manually
5. Start development


## Scripts

### Create documentation

To create documentation for the `./tf_al` directory. Execute following command
in `./docs`

```shell
$ make html
```

To clear the generated documentation use following command.

```shell
$ make clean
```


### Run tests

To perform automated unittests run following command in the root package directory.

```shell
$ pytest
```

To generate additional coverage reports run.

```shell
$ pytest --cov
```


# Contribution


# Issues