

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
2. [Documentation](https://exleonem.github.io/tf-al/)
3. [Getting started](#Getting-started)
    1. [Model wrapper](#Model-wrapper)
    2. [Acquisition functions](#Acquisition-functions)
    1. [Basic active learning loop](#Basic-active-learning-loop)
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

# Getting started


Following the active learning paradigm the most essential parts are the model and the pool of labeled/unlabeled data.


To enable modularity tensorflow models are wrapped. The model wrapper acts as an interface between the active learning loop and the model.
In essence the model wrapper defines methods which are called at different steps in the active learning loop.
To manage the labeled and unlabeled datapoints the pool class can be used. Which offers methods to label and select datapoints, labels and indices.


Other parts provided by the library easy the setup of active learning loops. The active learning loop class uses a dataset and model to creat an iterator, which then can be used to perform active learning over a single experiment.(model and query strategy combination)

The experiment suit can be used to perform a couple of experiments in a row, which is useful if for example you want to compare differnt acquisition functions.


## Model wrapper

Model wrappers are used to create an interface between the tensorflow model and the active learning loop.
Currently there are two wrappers defined. `Model` and `McDropout` for bayesian active learning. 
The `Model` wrapper can be used to create custom model wrappers.


Here is an example of how to create and wrap a basic McDropout model.

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Input, Flatten
from tf_al.wrapper import McDropout

# Define and wrap model (here McDropout)
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

# Wrap, configure and compile
model_config = Config(
    fit={"epochs": 200, "batch_size": 10},
    query={"sample_size" 25},
    eval={"batch_size": 900, "sample_size": 25}
)
model = McDropout(base_model, config=model_config)
model.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy", 
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)
```


### Basic methods


The model wrapper in essence can be used like a regular tensorflow model.

```python
model = McDropout(base_model)
model.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy", 
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

# Fit model to data
model.fit(inputs, targets, batch_size=25, epochs=100)

# Use model to predict output values
model(inputs)

# Evaluate model returning loss and accuracy
model.evaluate(some_inputs, some_targets)
```

To define a custom  custom model wrapper, simply extend your own class using the `Model` class and 
overwrite functions as needed. The regular tensorflow model can be accessed via `self._model`.

To provide your model wrappers as a package you can simply use the [template on github](https://github.com/ExLeonem/tf-al-ext), which already offers a poetry package setup.



```python
from tf_al import Model


class CustomModel(Model):

    def __init__(self, model, **kwargs):
        super().__init__(model, , model_type="custom", **kwargs)


    def __call__(self, *args, **kwargs):
        # Custom __call__ or standard tensorflow __call__


    def predict(self, inputs, **kwargs):
        # Custom prediction method or the standard tensorflow call model(inputs)
        

    def evaluate(self, inputs, targets, **kwargs):
        # Defining custom evaluate method
        # else standard evaluate method of tensorflow used.
        return {"metric_1": some_value, "metrics_2": some_other_value}


    def fit(self, *args, **kwargs):
        # Custom fitting procedure, else tensorflow .fit() method is used. 
        

    def compile(self, *args, **kwargs):
        # Custom compile method else using tensorflow .compile(**kwargs)
        

    def reset(self, pool, dataset):
        # In Which way to reset the network after each active learning round
        # standard is re-loading weights when enabled
```


## Acquisition functions



## Basic active learning loop


```python

import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Input, Flatten

from tf_al import ActiveLearningLoop, Dataset
from tf_al.wrapper import McDropout

# Load dataset and pack into dataset
(x_train, y_train), test_set = keras.datasets.mnist.load()
inital_pool_size = 20
dataset = Dataset(x_train, y_train, test=test_set, init_size=initial_pool_size)

# Create and wrap model
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

mc_model = McDropout(base_model)
mc_model.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy", 
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

# Create and start experiment suit (Collection of different experiments model + query_strategy)
query_strategy = "random"
active_learning_loop = ActiveLearningLoop(
    mc_model,
    dataset,
    query_strategy,
    step_size=10, # Number of new datapoints to select after each round
    max_rounds=100 # How many active learning rounds per experiment?
)

# To completely run through the active learning loop
active_learning_loop.run()

# Manually iterate over active learning loop
for step in active_learning_loop:

    # Dict with accumulated metrics 
    # ["train", "train_time", "query_time", "optim", "optim_time", "eval", "eval_time", "indices_selected"]
    step["train"]


# Alternativly iterate step inside the loop
num_rounds = 10
for i in range(num_rounds):

    metrics = active_learning_loop.step()
    # ... do something with the metrics
```

## Basic experiment suit setup

```python
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Input, Flatten

from tf_al import ActiveLearningLoop, Dataset, Config, ExperimentSuit, AcquisitionFunction
from tf_al.wrapper import McModel

# Split data and put into a dataset
x_train, x_test, y_train, y_test = train_test_split(some_inputs, some_targets, test_size=test_set_size)

# Number of initial datapoints in pool of labeled data
initial_pool_size = 20 
dataset = Dataset(
    x_train, y_train,
    test=(x_test, y_test),
    init_size=initial_pool_size
)

# Define and wrap model (here McDropout)
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

model_config = Config(
    fit={"epochs": 200, "batch_size": 10}, # Passed to fit() of the wrapper
    query={"sample_size" 25}, # Configuration passed to acquisition function during query step
    eval={"batch_size": 900, "sample_size": 25} # Parameters passed to evaluation method of the wrapper
)
model = McDropout(base_model, config=model_config)
model.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy", 
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

# Over which model to perform experiments single or list [model_1, ..., model_n]
models = model

# Define which acquisition functions to apply in separate runs either single one or a list [acquisition_1, ...] 
acquisition_functions = ["random", AcqusitionFunction("max_entropy", batch_size=900)]
experiments = ExperimentSuit(
    models,
    acquisition_functions,
    step_size=10, # Number of new datapoints to select after each round
    max_rounds=100 # How many active learning rounds per experiment?
)

```

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