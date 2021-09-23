import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import get


class Accuracy:

    def __init__(self, name="accuracy"):
        self.name = name
        self.__fn = get(name)

    
    def __call__(self, true_targets, predictions, **kwargs):
        exp, _var = predictions
        true_targets = tf.convert_to_tensor(true_targets)
        output = self.__fn(true_targets, exp)
        return np.mean(output)
