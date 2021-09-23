import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import get

class Accuracy:

    def __init__(self, name="accuracy"):
        self.name = name

    
    def __call__(self, true_targets, predictions, **kwargs):
        """
            Calculate the accuracy for the sampling procedure.
        """
        pred_mean = np.mean(predictions, axis=1)
        acc_fn = get(self.name)
        true_targets = tf.convert_to_tensor(true_targets)
        output = acc_fn(true_targets, pred_mean)
        return np.mean(output)

        # pred_targets = np.argmax(predictions, axis=-1)
        # if len(targets.shape) == 2:
        #     targets = np.argmax(targets, axis=1)

        # if sample_size > 1:
        #     targets = np.vstack([targets]*sample_size).T
        
        # return np.mean(pred_targets == targets)


    def __str__(self):
        return self.name