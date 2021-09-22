import numpy as np

class TrueNegative:

    def __init__(self):
        self.name = "true_negative"
        self._multi_label = multi_label

    
    def __call__(self, true_targets, predictions, sample_size=None, **kwargs):
        num_samples = predictions.shape[1]
        stacked_targets = np.vstack([true_targets]*num_samples).T
        predicted_labels = np.argmax(predictions, axis=-1)

        (stacked_targets == predicted_labels)