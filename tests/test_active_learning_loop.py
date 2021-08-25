import numpy as np
import pytest

from tf_al import ActiveLearningLoop, Pool, Dataset
from tf_al.wrapper import Model
from tf_al.utils import setup_growth


class MockFitResult(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class MockModel:

    def __init__(self, output_shape):
        self.output_shape = output_shape

    def __call__(self, *args, **kwargs):
        return np.random.randn(self.output_shape) 

    def predict(self, *args, **kwargs):
        return np.random.randn(self.output_shape)

    def fit(self, x_inputs, y_inputs, **kwargs):
        return MockFitResult(history={})

    def load_weights(self, path):
        pass


class TestActiveLearningLoopIteration:

    def test_simple_iteration(self):
        inputs = np.random.randn(10)
        targets = np.random.choice([0, 1, 2], 10)

        mock_model = Model(MockModel(10), None)
        dataset = Dataset(inputs, targets)
        loop = ActiveLearningLoop(mock_model, dataset, "random")

        for i in loop:
            print(i)

        assert True

    
    def test_some(self):
        assert True