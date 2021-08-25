
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




## Model wrapper

Model wrappers are used to create interfaces to the active learning loop. 

Currently there are two wrappers defined `Model` and `McDropout` for bayesian active learning using Dropout as bayesian approximation.

```python
import tensorflow as tf
from tensorflow.keras import Model, Sequential
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
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=[keras.metrics.SparseCategoricalAccuracy()])
```

You can easily define a custom model wrapper, simply extend your own class using the `Model` class and 
overwrite functions as needed. 

```python
from tf_al import Model


class CustomModel(Model):

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        # Custom 


    def predict(self, inputs, **kwargs):
        # Custom prediction method or the standard tensorflow call model(inputs)


    def evaluate(self, inputs, targets, **kwargs):
        """
            Defining custom evaluate method, else standard evaluate method of tensorflow used.
        """
        # Some operations
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



## Basic experiment setup

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
    fit={"epochs": 200, "batch_size": 10},
    query={"sample_size" 25},
    eval={"batch_size": 900, "sample_size": 25}
)
model = McDropout(base_model, config=model_config)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=[keras.metrics.SparseCategoricalAccuracy()])

# Over which model to perform experiments single or list [model_1, ..., model_n]
models = model

# Define which acquisition functions to apply in separate runs either single one or a list [acquisition_1, ...] 
acquisition_functions = AcqusitionFunction("random", batch_size=900)

experiments = ExperimentSuit(
    models,
    acquisition_functions,
    step_size=10, # How many new datapoints to select per round
    max_rounds=25, # Setting limit on the rounds per experiment
)

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