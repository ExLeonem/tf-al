import numpy as np


class StochasticStats:

    
    @staticmethod
    def on_evaluate_loss(predictions, inputs, targets, **kwargs):
        """
            Hook called upon evaluating loss in evaluation step.
        """
        loss = 12
        return {"loss": loss}


    @staticmethod
    def on_evaluate_acc(predictions, inputs, targets, **kwargs):
        """
            Hook called upon evaluating to get model accuracy.
        """
        exp, var = predictions

        acc = 13
        return {"acc": acc}


    @staticmethod
    def on_evaluate_auc(predictions, inputs, targets, **kwargs):
        # tf.keras.metrics.AUC
        pass


    @staticmethod
    def __prepare_targets(targets, sample_size):
        targets = self.__select_one_hot_index(targets)
        if sample_size > 1:
            targets = np.vstack([targets]*sample_size).T

        return targets


    @staticmethod
    def __select_one_hot_index(targets):
        # Potentially one-hot-vector, select index for target comparison
        if len(targets.shape) == 2:
            return np.argmax(targets, axis=-1)

        return targets