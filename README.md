
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


```python
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Input, Flatten

from tf_al import ActiveLearningLoop, Dataset, Pool


#
base_model = Sequential([
    Conv2D(32, 3, activation=tf.nn.relu, padding="same", input_shape=input_shape),
    Conv2D(64, 3, activation=tf.nn.relu, padding="same"),
    MaxPooling2D(),
    Dropout(.25),
    Flatten(),
    Dense(128, activation=tf.nn.relu),
    Dropout(.5),
    Dense(output, activation="softmax")        
])



```


# Development

## Setup

1. Create a virtual env (optional)
2. [Install and Setup Poetry](https://python-poetry.org/docs/#installation)
3. Install package dependencies using poetry


## Scripts

### Create documentation

To create documentation for the `./tf_al` directory. Execute following command
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