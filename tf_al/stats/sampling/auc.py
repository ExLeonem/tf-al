from scipy.special import comb
from sklearn.preprocessing import label_binarize
import numpy as np


class AUC():

    def __init__(self, num_thresholds=10, multi_label=False, from_logits=False, labels=None):
        self.name = "auc"

        if num_thresholds < 1:
            raise ValueError("Error in AUC(). Number of thresholds needs to be > 1.")
        self._thresholds = np.linspace(0, 1, num_thresholds)
        self._from_logits = from_logits
        self._multi_label = multi_label
        self._labels = labels

    
    def __call__(self, true_target, predictions, sample_size=None):
        """
            Calculate the AUC metric.
        """
        # https://towardsdatascience.com/roc-and-auc-how-to-evaluate-machine-learning-models-in-no-time-fb2304c83a7f
        # https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826#:~:text=False%20Positive%20(FP)%3A%20It,the%20negative%20class%20as%20positive.&text=It%20is%20also%20known%20as,or%20(1%2DAccuracy).

        if self._multi_label:
            return self.__multiclass(true_target, predictions)

                        

        return 12


    def __multiclass(self, true_target, predictions):

        
        num_classes = predictions.shape[-1]
        class_labels = self._get_labels(num_classes)
        true_target = label_binarize(true_target)
        num_thresholds = len(self._thresholds)

        shape = (num_classes, num_thresholds)
        tp = np.zeros(shape)
        tn = np.zeros(shape)
        fp = np.zeros(shape)
        fn = np.zeros(shape)

        for idx in range(num_classes):
            label = class_labels[idx]

            class_prediction = np.take(predictions, idx, axis=-1)
            for th_idx in range(num_thresholds):
                
                threshold = self._thresholds[th_idx]
                selector = true_target == label

                current_tp = np.sum(true_target[selector] == pred_targets[selector])


                other_selector = np.logical_not(selector)
                # current_fn = np.sum(true_target[other_selector] == )
                tp[idx, th_idx] = current_tp



    def true_positive(self, true_targets, predictions):

        pred_targets = np.where(np.mean(class_prediction, axis=1)>=threshold, label, -1)

    
    def __binary(self, true_target, predictions):

        pass



    def _get_labels(self, num_classes):
        if self._labels is not None:
            return self._labels

        return list(range(num_classes))


    