import numpy as np
from tensorflow.keras.metrics import get


class Accuracy:

    def __init__(self, name="accuracy"):
        self.name = name
        self.__fn = get(name)

    
    def __call__(self, true_targets, predictions, **kwargs):
        exp, _var = predictions
        return np.mean(self.__fn(true_targets, exp))
