import numpy as np
import tensorflow as tf


class Loss:
    """
        Calculate the loss using a predefined tensorflow function.
    """

    def __init__(self, name="loss"):
        self.name = name

    def __call__(self, true_target, predictions, **kwargs):
        """
            Parameters:
                true_target (numpy.ndarray): The true targets to check against.
                pred_target (numpy.ndarray): The targets predicted by the network. Shape (num datapoints, num samples, ...)
            
            Returns:
                (int) the loss calculated.
        """

        # if sample_size is None:
        #     raise ValueError("Error in tf_al.stats.sampling.Loss(). Missing kwarg parameter sample_size. Seems like you arent using a McDropout model.")

        pred_mean = np.mean(predictions, axis=1)
        loss_fn = tf.keras.metrics.get(self.name)
        return np.mean(loss_fn(true_target, pred_mean).numpy())