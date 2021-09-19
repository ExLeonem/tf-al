import math
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow.keras as keras

from . import  Model
import tensorflow as tf



class McDropout(Model):
    """
        Wrapper class for neural networks.

    """

    def __init__(self, model, config=None, **kwargs):
        super().__init__(model, config=config, model_type="mc_dropout", **kwargs)

        # disable batch norm
        # super().disable_batch_norm()


    def __call__(self, inputs, sample_size=10, batch_size=None, **kwargs):
        """
            Perform a prediction using mc dropout as bayesian approximation.

            Parameters:
                inputs (numpy.ndarray): Inputs going into the model
                sample_size (int): Number of samples to acquire from bayesian model. (default=10)
                batch_size (int): In how many batches to split the data? (default=None)
        """

        if batch_size is None:
            batch_size = len(inputs)
        
        if batch_size < 1:
            raise ValueError("Error in McDropout.__call__(). Can't select negative amount of batches.")

        if sample_size < 1:
            raise ValueError("Error in McDropout.__call__(). Can't sample negative amount of times.")

        total_len = len(inputs)
        num_batches = math.ceil(total_len/batch_size)
        batches = np.array_split(inputs, num_batches, axis=0)
        predictions = []

        for batch in batches:

            # Sample n_times for given batch
            posterior_samples = []
            for i in range(sample_size):
                posterior_samples.append(self._model(batch, training=True))
                
            # Sampled single time or multiple times?
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
        predictions = self.__call__(inputs, sample_size=sample_size, **kwargs)

        if self.is_classification():
            loss, acc = self.__evaluate(predictions, targets, sample_size)
            return {"loss": loss, "accuracy": acc}

        loss_fn = keras.losses.get(self._model.loss)
        loss = loss_fn(predictions, targets).numpy()
        return {"loss": np.mean(loss, axis=-1), "accuracy": []}
    

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
        extended = self._problem.extend_binary_predictions(predictions)
        pred_targets = np.argmax(extended, axis=-1)

        # One-hot vector passed
        if len(targets.shape) == 2:
            targets = np.argmax(targets, axis=1)
        
        # Extend target dimension (multiple sample in prediction)
        if sample_size > 1:
            targets = np.vstack([targets]*sample_size).T
        
        acc = np.mean(pred_targets == targets)
        return [np.mean(loss.numpy()), acc]


    # ---------------
    # Metric hooks
    # -------------------------

    def _on_evaluate_loss(self, predictions, inputs, targets, **kwargs):
        """
            Hook called upon evaluating loss in evaluation step.
        """

        expectation = predictions
        if len(predictions.shape) == 3:
            expectation = np.average(predictions, axis=1)

        loss_fn = tf.keras.losses.get(self._model.loss)
        loss = loss_fn(targets, expectation)
        return {"loss": loss}


    def _on_evaluate_acc(self, predictions, inputs, targets, **kwargs):
        """
            Hook called upon evaluating to get model accuracy.
        """
        extended = self._problem.extend_binary_predictions(predictions)
        pred_targets = np.argmax(extended, axis=-1)

        sample_size = kwargs.get("sample_size", 10)
        targets = self.__prepare_targets(targets, sample_size)
        acc = np.mean(pred_targets == targets)
        return {"acc": acc}


    def _on_evaluate_auc(self, predictions, inputs, targets, **kwargs):
        pass


    def __prepare_targets(self, targets, sample_size):
        targets = self.__select_one_hot_index(targets)
        if sample_size > 1:
            targets = np.vstack([targets]*sample_size).T

        return targets


    def __select_one_hot_index(self, targets):

        # Potentially one-hot-vector, select index for target comparison
        if len(targets.shape) == 2:
            return np.argmax(targets, axis=-1)

        return targets


    # -----
    # Acquisition functions
    # ---------------------------

    def get_query_fn(self, name):

        fn = None
        if name == "max_entropy":
            fn = self.__max_entropy
        
        elif name == "bald":
            fn = self.__bald
        
        elif name == "max_var_ratio":
            fn = self.__max_var_ratio

        elif name == "std_mean":
            fn = self.__std_mean

        elif name == "margin_sampling":
            fn = self.__margin_sampling

        return fn


    def __max_entropy(self, data, sample_size=10, **kwargs):
        """
            Select datapoints by using max entropy.

            Parameters:
                model (tf.Model) The tensorflow model to use for selection of datapoints
                unlabeled_pool (Pool) The pool of unlabeled data to select
        """        
        # Create predictions
        predictions = self.__call__(data, sample_size=sample_size)
        expectation = self.expectation(predictions)
        
        # Absolute value to prevent nan values and + 0.001 to prevent infinity values
        log_post = np.log(np.abs(expectation) + .001)

        # Calculate max-entropy
        return -np.sum(expectation*log_post, axis=1)


    def __bald(self, data, sample_size=10, **kwargs):
        # TODO: dimensions do not line up in mutli class
        # predictions shape (batch, num_predictions, num_classes)
        predictions = self.__call__(data, sample_size=sample_size)
        posterior = self.expectation(predictions)

        entropy = -self.__shannon_entropy(posterior)
        # first_term = -np.sum(posterior*np.log(np.abs(posterior) + .001), axis=1)

        # Missing dimension in binary case?
        predictions = self._problem.extend_binary_predictions(predictions)
        inner_sum = self.__shannon_entropy(predictions)
        # inner_sum = np.sum(predictions*np.log(np.abs(predictions) + .001), axis=1)
        disagreement = np.sum(inner_sum, axis=1)/predictions.shape[1]
        return entropy + disagreement


    def __max_var_ratio(self, data, sample_size=10, **kwargs):
        """
            Select datapoints by maximising variation ratios.

            # (batch, predictions, classes) reduce to (batch, predictions (max-class))
            # 1 - (count of most common class / num predictions)
        """
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
        predictions = self.__call__(data, sample_size=sample_size)
        expectation = self.expectation(predictions)
        indices = np.argsort(expectation)[:, :-2]


    
    def __least_confidence(self, data, sample_size=10, **kwargs):
        """
            Select sample which minimize distance between two most probable labels.
            Margin Sampling (MS).
        """
        predictions = self.__call__(data, sample_size=sample_size)
        expectation = self.expectation(predictions)

        return np.argmin(expectation, axis=1)



    # --------------
    # Utils
    # --------------------

    def __shannon_entropy(self, values):
        """
            Calculate the shannon entropy for given values.
        """
        return np.sum(values*np.log(values + .001), axis=1)

    
    def expectation(self, predictions):
        """
            Calculate the mean of the distribution
            output distribution.

            Returns:
                (numpy.ndarray) The expectation per datapoint
        """
        # predictions -> (batch_size, num_predictions)
        predictions = self._problem.extend_binary_predictions(predictions)
        return np.average(predictions, axis=1)


    def variance(self, predictions):
        """
            Calculate the variance of the distribution.

            Returns:
                (numpy.ndarray) The variance per datapoint and target
        """
        predictions = self._problem.extend_binary_predictions(predictions)
        return np.var(predictions, axis=1)


    def std(self, predictions):
        """
            Calculate the standard deviation.

            Returns:
                (numpy.ndarray) The standard deviation per datapoint and target
        """
        predictions = self._problem.extend_binary_predictions(predictions)
        return np.std(predictions, axis=1)
