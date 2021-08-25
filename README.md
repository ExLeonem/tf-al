
# Active learning with tensorflow


# Index

1. [Installation](#Installation)
2. [Getting started](#Getting-started)
    1. [Model wrapper](#Model-wrapper)
    2. [Acquisition functions](#Acquisition-functions)
    1. [Basic active learning loop](#Basic-active-learning-loop)
3. [Development](#Development)
    1. [Setup](#Setup)
    2. [Scripts](#Scripts)
4. [Contribution](#Contribution)
5. [Issues](#Issues)



# Installation


```shell
$ pip install tf-al
```


# Getting started

The library 


## Model wrapper

Model wrappers are use to create interfaces to the active learning loop.



## Acquisition functions





## Basic active learning loop


```
import tensorflow as tf
from tf_al import ActiveLearningLoop, Dataset, Pool



```







# Development

## Setup

1. Create a virtual env (optional)
2. [Install and Setup Poetry](https://python-poetry.org/docs/#installation)
3. Install package dependencies using poetry


## Scripts

### Create documentation

To create documentation for the `./active_leanring` directory. Execute following command
in `./docs`

```shell
$ make html
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