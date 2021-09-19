import numpy as np
import tensorflow as tf


class SamplingStats:

    """
        Callback functions for stats for sampling based models.
    """


    @staticmethod
    def loss(predictions, inputs, targets, **kwargs):
        """
            Hook called upon evaluating loss in evaluation step.
        """

        loss = kwargs.pop("loss")
        expectation = predictions
        if len(predictions.shape) == 3:
            expectation = np.average(predictions, axis=1)

        loss_fn = tf.keras.losses.get(loss)
        loss = loss_fn(targets, expectation)
        return {"loss": loss}


    @staticmethod
    def accuracy(predictions, inputs, targets, **kwargs):
        """
            Hook called upon evaluating to get model accuracy.
        """
        extended = self.extend_binary_predictions(predictions)
        pred_targets = np.argmax(extended, axis=-1)

        sample_size = kwargs.get("sample_size", 10)
        targets = SamplingStats.__prepare_targets(targets, sample_size)
        acc = np.mean(pred_targets == targets)
        return {"acc": acc}


    @staticmethod
    def auc(predictions, inputs, targets, **kwargs):
        # tf.keras.metrics.AUC
        pass


    @staticmethod
    def __prepare_targets(targets, sample_size):
        targets = SamplingStats.__select_one_hot_index(targets)
        if sample_size > 1:
            targets = np.vstack([targets]*sample_size).T

        return targets


    @staticmethod
    def __select_one_hot_index(targets):
        # Potentially one-hot-vector, select index for target comparison
        if len(targets.shape) == 2:
            return np.argmax(targets, axis=-1)

        return targets