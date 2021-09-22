import numpy as np

class Confusion:

    def __init__(self, multi_class=True, labels=None):
        self.name = "true_positive"
        self._multi_class = multi_class
        self._labels = labels

    
    def __call__(self, true_targets, predictions, **kwargs):
        """
        
        """


        num_classes = predictions.shape[-1]
        class_labels = self._get_labels(num_classes)
        
        true_positive_scores = np.zeros(num_classes)
        for idx in range(num_classes):
            label = class_labels[idx]

            selector = true_targets == label
            
        


        # Binary case
        # predictions = np.where(predictions>.5, 1, 0)
        # stacked_targets = np.vstack([true_targets]*num_samples).T
        # predicted_labels = np.argmax(predictions, axis=-1)
        # return np.sum(stacked_targets == predicted_labels]

    
    def _get_labels(self, num_classes):
        if self._labels is not None:
            return self._labels

        return list(range(num_classes))