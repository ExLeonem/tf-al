import numpy as np

class TruePositive:

    def __init__(self, multi_class=True, labels=None):
        self.name = "true_positive"
        self._multi_class = multi_class
        self._labels = labels

    
    def __call__(self, true_targets, predictions, **kwargs):
        """
        
        """
        pass


        # Binary case
        # predictions = np.where(predictions>.5, 1, 0)
        # stacked_targets = np.vstack([true_targets]*num_samples).T
        # predicted_labels = np.argmax(predictions, axis=-1)
        # return np.sum(stacked_targets == predicted_labels]
