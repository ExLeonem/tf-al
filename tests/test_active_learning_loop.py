from logging import disable
from math import exp
import numpy as np
import pytest

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Softmax

from tf_al import ActiveLearningLoop, Pool, Dataset
from tf_al.wrapper import Model
from tf_al.utils import setup_growth, disable_tf_logs



def base_model(output=10):
    return Sequential([
        Dense(output, activation=tf.nn.relu),
        Softmax()
    ])


class MockFitResult(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class MockModel:

    def __init__(self, output_shape):
        self.output_shape = output_shape

    def __call__(self, *args, **kwargs):
        return np.random.randn(self.output_shape) 

    def evaluate(self, *args, **kwargs):
        return {"loss": 10, "acc": 0.1}

    def predict(self, *args, **kwargs):
        return np.random.randn(self.output_shape)

    def fit(self, x_inputs, y_inputs, **kwargs):
        return MockFitResult(history={})

    def load_weights(self, path):
        pass


class TestActiveLearningLoopIteration:

    def test_base_iteration(self):
        inputs = np.random.randn(10)
        targets = np.random.choice([0, 1, 2], 10)

        model = MockModel(10)
        mock_model = Model(model, None)
        dataset = Dataset(inputs, targets, test=(inputs, targets))
        loop = ActiveLearningLoop(mock_model, dataset, "random")

        res = None
        for i in loop:
            res = i

        keys = set(i.keys())
        expected_keys = ["eval_time", "indices_selected", "optim"]
        assert i["eval"] == {} and \
            all([key in keys for key in expected_keys])

    
    def test_base_iteration_generic_model(self):
        d_length = 10
        inputs = np.random.randn(d_length)
        targets = np.random.choice([0, 1, 2], d_length)

        model = base_model(output=d_length)
        mock_model = Model(model, None)
        mock_model.compile(loss="mean_absolute_error", optimizer="sgd", metrics=[keras.metrics.SparseCategoricalAccuracy()])

        dataset = Dataset(inputs, targets, test=(inputs, targets))
        loop = ActiveLearningLoop(mock_model, dataset, "random")

        res = None
        for i in loop:
            res = i
            break

        keys = res.keys()
        expected_keys = ["train", "train_time", "optim", "optim_time", "eval", "eval_time", "indices_selected"]
        assert all([key in keys for key in expected_keys])