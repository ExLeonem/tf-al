import os, math
import time
import numpy as np
import logging as log
from sklearn.metrics import accuracy_score
import tensorflow.keras as keras

from . import  Model, ModelType, Mode
import tensorflow as tf



class McDropout(Model):
    """
        Wrapper class for neural networks.

    """

    def __init__(self, model, config=None, **kwargs):
        super().__init__(model, config=config, model_type=ModelType.MC_DROPOUT, **kwargs)

        # disable batch norm
        # super().disable_batch_norm()


    def __call__(self, inputs, sample_size=10, batch_size=None, callback=None, **kwargs):
        """
            
            Parameters:
                inputs (numpy.ndarray): Inputs going into the model
                sample_size (int): How many times to sample from posterior?
                batch_size (int): In how many batches to split the data?
        """

        if batch_size is None:
            batch_size = len(inputs)
        
        if batch_size < 1:
            raise ValueError("Error in McDropout.__call__(). Can't select negative amount of batches.")

        if sample_size < 1:
            raise ValueError("Error in McDropout.__call__(). Can't sample negative amount.")


        total_len = len(inputs)
        num_batches = math.ceil(total_len/batch_size)
        batches = np.array_split(inputs, num_batches, axis=0)
        predictions = []

        for batch in batches:
            # Sample from posterior
            posterior_samples = []
            for i in range(sample_size):
                posterior_samples.append(self._model(batch, training=True))
                
            # Omit sample dimension, when only sampled single time?
            if sample_size > 1:
                stacked = np.stack(posterior_samples, axis=1)
                predictions.append(stacked)
            else:
                predictions.append(posterior_samples[0])


        if len(predictions) == 1:
            return predictions[0]

        return np.vstack(predictions)


    def evaluate(self, inputs, targets, sample_size=10, **kwargs):
        """
            Evaluate a model on given input data and targets.
        """
        
        if len(inputs) != len(targets):
            raise ValueError("Error in McDropout.evaluate(). Targets and inputs not of equal length.")

        # Returns: (batch_size, sample_size, target_len) or (batch_size, target_len)
        self.logger.info("evaluate/call")
        predictions = self.__call__(inputs, sample_size=sample_size, **kwargs)
        self.logger.info("evaluate/predictions.shape: {}".format(predictions.shape))
        loss, acc = self.__evaluate(predictions, targets, sample_size)
        return {"loss": loss, "accuracy": acc}


    def __evaluate(self, predictions, targets, sample_size):
        """

            Parameters:
                predictions (numpy.ndarray): The predictions made by the network of shape (batch, targets) or (batch, samples, targets)
                targets (numpy.ndarray): The target values
                sample_size (int): The number of samples taken from posterior.

            Returns:
                (list()) of values representing the accuracy and loss
        """
        
        expectation = predictions
        if len(predictions.shape) == 3:
            expectation = np.average(predictions, axis=1)

        # Will fail in regression case!!!! Add flag to function?
        loss_fn = tf.keras.losses.get(self._model.loss)
        loss = loss_fn(targets, expectation)

        # Extend dimension in binary case 
        extended = self.extend_binary_predictions(predictions)
        pred_targets = np.argmax(extended, axis=-1)

        # One-hot vector passed
        if len(targets.shape) == 2:
            targets = np.argmax(targets, axis=1)
        
        # Extend target dimension (multiple sample in prediction)
        if sample_size > 1:
            targets = np.vstack([targets]*sample_size).T
        
        acc = np.mean(pred_targets == targets)
        return [np.mean(loss.numpy()), acc]


    def expectation(self, predictions):
        """
            Calculate the mean of the distribution
            output distribution.

            Returns:
                (numpy.ndarray) The expectation per datapoint
        """
        # predictions -> (batch_size, num_predictions)
        predictions = self.extend_binary_predictions(predictions)
        return np.average(predictions, axis=1)


    def variance(self, predictions):
        """
            Calculate the variance of the distribution.

            Returns:
                (numpy.ndarray) The variance per datapoint and target
        """
        predictions = self.extend_binary_predictions(predictions)
        return np.var(predictions, axis=1)


    def std(self, predictions):
        """
            Calculate the standard deviation.

            Returns:
                (numpy.ndarray) The standard deviation per datapoint and target
        """
        predictions = self.extend_binary_predictions(predictions)
        return np.std(predictions, axis=1)


    def just_return(self, predictions):
        return predictions


    def extend_binary_predictions(self, predictions, num_classes=2):
        """
            In MC Dropout case always predictions of shape
            (batch_size, sample_size, classes) for classification 
            or (batch_size, sample_size) for binary/regression case
        """

        # Don't modify predictions shape in regression case
        if not self.is_classification():
            return predictions


        # Binary case: calculate complementary prediction and concatenate
        if self.is_binary():
            bin_alt_class = (1 + np.zeros(predictions.shape)) - predictions

            # Expand dimensions for predictions to concatenate. Is this needed?
            bin_alt_class = np.expand_dims(bin_alt_class, axis=-1)
            predictions = np.expand_dims(predictions, axis=-1)

            # Concatenate predictions
            predictions = np.concatenate([predictions, bin_alt_class], axis=len(predictions.shape)-1)
        
        return predictions



    # -----
    # Acquisition functions
    # ---------------------------

    def get_query_fn(self, name):

        if name == "max_entropy":
            return self.__max_entropy
        
        if name == "bald":
            return self.__bald
        
        if name == "max_var_ratio":
            return self.__max_var_ratio

        if name == "std_mean":
            return self.__std_mean

        if name == "margin_sampling":
            return self.__margin_sampling

        return None


    def __max_entropy(self, data, sample_size=10, **kwargs):
        """
            Select datapoints by using max entropy.

            Parameters:
                model (tf.Model) The tensorflow model to use for selection of datapoints
                unlabeled_pool (Pool) The pool of unlabeled data to select
        """
        self.logger.info("----------Max-Entropy-------------")
        
        # Create predictions
        predictions = self.__call__(data, sample_size=sample_size)
        expectation = self.expectation(predictions)
        
        # Absolute value to prevent nan values and + 0.001 to prevent infinity values
        log_post = np.log(np.abs(expectation) + .001)

        # Calculate max-entropy
        return -np.sum(expectation*log_post, axis=1)


    def __bald(self, data, sample_size=10, **kwargs):
        # TODO: dimensions do not line up in mutli class
        
        self.logger.info("------------ BALD -----------")
        # predictions shape (batch, num_predictions, num_classes)
        self.logger.info("_bald/data-shape: {}".format(data.shape))
        predictions = self.__call__(data, sample_size=sample_size)

        self.logger.info("_bald/predictions-shape: {}".format(predictions.shape))
        posterior = self.expectation(predictions)
        self.logger.info("_bald/posterior-shape: {}".format(posterior.shape))

        first_term = -np.sum(posterior*np.log(np.abs(posterior) + .001), axis=1)

        # Missing dimension in binary case?
        predictions = self.extend_binary_predictions(predictions)
        
        inner_sum = np.sum(predictions*np.log(np.abs(predictions) + .001), axis=1)
        self.logger.info("_bald/inner-shape: {}".format(inner_sum.shape))

        second_term = np.sum(inner_sum, axis=1)/predictions.shape[1]

        self.logger.info("_bald/first-term-shape: {}".format(first_term.shape))
        self.logger.info("_bald/second-term-shape: {}".format(second_term.shape))
        return first_term + second_term


    def __max_var_ratio(self, data, sample_size=10, **kwargs):
        """
            Select datapoints by maximising variation ratios.

            # (batch, predictions, classes) reduce to (batch, predictions (max-class))
            # 1 - (count of most common class / num predictions)
        """
        self.logger.info("----------Max-Var-Ratio--------------")

        # (batch, sample, num classses)
        # (batch, num_classes)
        predictions = self.__call__(data, sample_size=sample_size)
        posterior = self.expectation(predictions)

        # Calcualte max variation rations
        return 1 - posterior.max(axis=1)


    def __std_mean(self, data, sample_size=10,  **kwargs):
        """
           Maximise mean standard deviation.
           Check std mean calculation. Depending the model type calculation of p(y=c|x, w) can differ.
           (Kampffmeyer et al. 2016; Kendall et al. 2015)

           Todo:
            Implement distinction for different model types.
        """
        self.logger.info("----------Std-Mean-------------")

        # TODO: generalize for n-classes For binary classes
        predictions = self.__call__(data, sample_size=sample_size)

        # Calculate variance/standard deviation from samples
        variance = self.variance(predictions)
        std = np.square(variance)

        # Mean over target variables
        return np.mean(std, axis=-1)


    def __margin_sampling(self, data, sample_size=10, **kwargs):
        """
            Select sample which minimize distance between two most probable labels.
            Margin Sampling (MS).
        """

        self.logger.info("----------Margin-Sampling-------------")

        predictions = self.__call__(data, sample_size=sample_size)
        expectation = self.expectation(predictions)

        indices = np.argsort(expecation)[:, :-2]

    
    def __least_confidence(self, dtaa, sample_size=10, **kwargs):
        """
            Select sample which minimize distance between two most probable labels.
            Margin Sampling (MS).
        """

        self.logger.info("----------Margin-Sampling-------------")

        predictions = self.__call__(data, sample_size=sample_size)
        expectation = self.expectation(predictions)

        return np.argmin(expecation, axis=1)


