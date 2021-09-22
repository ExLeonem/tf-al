import numpy as np


class ProblemUtils:

    def __init__(self, classification: bool=True, is_binary: bool=False):
        self.classification = classification

        self.is_binary = is_binary
        if not self.classification:
            self.is_binary = False


    def extend_binary_predictions(self, predictions, num_classes=2):
        """
            In MC Dropout case always predictions of shape
            (batch_size, sample_size, classes) for classification 
            or (batch_size, sample_self.is_binary = is_binarysize) for binary/regression case
        """

        # Don't modify predictions shape in regression case
        if not self.classification:
            return predictions


        # Binary case: calculate complementary prediction and concatenate
        if self.is_binary:
            bin_alt_class = (1 + np.zeros(predictions.shape)) - predictions

            # Expand dimensions for predictions to concatenate. Is this needed?
            bin_alt_class = np.expand_dims(bin_alt_class, axis=-1)
            predictions = np.expand_dims(predictions, axis=-1)

            # Concatenate predictions
            predictions = np.concatenate([predictions, bin_alt_class], axis=len(predictions.shape)-1)
        
        return predictions

    
    def alt_extend_binary_predictions(self, predictions):
        """
            Extend dimensions for binary classification problems.
 
            Parameters:
                predictions (numpy.ndarray): Predictions made by the network.

            Returns:
                (numpy.ndarray) The predictions with extended dimension.
        """
        # Don't modify predictions shape in regression case
        if not self.classification:
            return predictions

        # Binary case: calculate complementary prediction and concatenate
        if self.is_binary():
            bin_alt_class = (1 + np.zeros(predictions.shape)) - predictions

            # Expand dimensions for predictions to concatenate. Is this needed?
            # bin_alt_class = np.expand_dims(bin_alt_class, axis=-1)
            # predictions = np.expand_dims(predictions, axis=-1)

            # Concatenate predictions
            class_axis = len(predictions.shape) + 1
            predictions = np.concatenate([predictions, bin_alt_class], axis=len(predictions.shape)-1)
        
        return predictions