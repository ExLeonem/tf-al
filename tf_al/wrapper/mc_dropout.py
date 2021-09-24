import math
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.special import digamma, beta
import tensorflow.keras as keras
import tensorflow as tf

from . import  Model
from ..utils import beta_approximated_upper_joint_entropy



class McDropout(Model):
    """
        Wrapper class for neural networks.


    """

    def __init__(self, model, config=None, **kwargs):
        super().__init__(model, config=config, model_type="mc_dropout", **kwargs)
        self._approximation_type = "sampling"

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

        output_metrics = {}
        for metric in self.eval_metrics:

            metric_name = None
            if hasattr(metric, "__name__"):
                metric_name = metric.__name__

            else:
                metric_name = metric.name

            output_metrics[metric_name] = metric(targets, predictions, sample_size=sample_size)

        return output_metrics

        # if self.is_classification():
        #     loss, acc = self.__evaluate(predictions, targets, sample_size)
        #     return {"loss": loss, "accuracy": acc}

        # loss_fn = keras.losses.get(self._model.loss)
        # loss = loss_fn(predictions, targets).numpy()
        # return {"loss": np.mean(loss, axis=-1), "accuracy": []}
    

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


    def compile(self, *args, **kwargs):
        self._model.compile(**kwargs)
        metrics = self._create_init_metrics(kwargs)        
        metric_names = self._extract_metric_names(metrics)

        self.eval_metrics = self._init_metrics("sampling", metric_names)
        self._compile_params = kwargs


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

        elif name == "baba":
            fn = self.__baba

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


    def __baba(self, data, sample_size=10, **kwargs):
        """
            Normalized mutual information

            Implementation of acquisition function described in:
            BABA: Beta Approximation for Bayesian Active Learning, Jae Oh Woo
        """
        # predictions shape (batch, num_predictions, num_classes)
        predictions = self.__call__(data, sample_size=sample_size)
        sample_mean = self.expectation(predictions)
        entropy = -self.__shannon_entropy(sample_mean)
        disagreement = self.__disagreement(predictions)
        bald_term = self.__mutual_information(entropy, disagreement)
        
        # Beta approximation parameters
        sample_var = self.variance(predictions)
        a = ((np.power(sample_mean, 2)*(1-sample_mean))/(sample_var+.0001))-sample_mean
        b = ((1/sample_mean)-1)*a
        upper_joint_entropy = beta_approximated_upper_joint_entropy(a, b)
        return bald_term/np.abs(upper_joint_entropy)



    # --------------
    # Utils
    # --------------------

    def __disagreement(self, predictions):
        predictions = self._problem.extend_binary_predictions(predictions)
        inner_sum = self.__shannon_entropy(predictions)
        return np.sum(inner_sum, axis=1)/predictions.shape[1]


    def __mutual_information(self, entropy, disagreement):
        return entropy + disagreement

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
