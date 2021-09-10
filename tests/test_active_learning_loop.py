import numpy as np
import pytest

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Softmax

from tf_al import ActiveLearningLoop, Pool, Dataset
from tf_al.wrapper import Model
from tf_al.utils import setup_growth


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
        return [10, 0.1]

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

        expected_loss, expected_acc = model.evaluate()
        assert i["eval"]["loss"] == expected_loss and \
            i["eval"]["accuracy"] == expected_acc and \
            i.keys == None

    
    def test_iteration_with_model(self):
        d_length = 10
        inputs = np.random.randn(d_length)
        targets = np.random.choice([0, 1, 2], d_length)

        model = base_model(output=d_length)
        mock_model = Model(model, None)
        mock_model.compile(loss="mean_absolute_error", optimizer="sgd")

        dataset = Dataset(inputs, targets, test=(inputs, targets))
        loop = ActiveLearningLoop(mock_model, dataset, "random")

        res = None
        for i in loop:
            res = i

        expected_loss, expected_acc = model.evaluate()
        assert i["eval"]["loss"] == expected_loss and \
            i["eval"]["accuracy"] == expected_acc and \
            i.keys == None