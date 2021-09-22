import os, math
import uuid
import numpy as np
from enum import Enum
import tensorflow as tf

from . import Checkpoint
from ..utils import setup_logger, ProblemUtils
from ..stats import get


class Model:
    """
        Base wrapper for deep learning models to interface
        with the active learning environment. 
        
        Attributes:
            _model (tf.Model): Tensorflow or pytorch module.
            _config (Config): Model configuration
            _mode (Mode): The mode the model is in 'train' or 'test'/'eval'.
            _model_type (str): The model type
            _checkpoints (Checkpoint): Created checkpoints.

        Parameters:
            model (tf.Model): The tensorflow model to be used.
            config (Config): Configuration object for the model. (default=None)
            is_binary (bool): 
            classification (bool): 
    """

    def __init__(
        self, 
        model, 
        config=None, 
        name=None,
        model_type=None, 
        checkpoint=None,
        verbose=False,
        checkpoint_path=None,
        **kwargs
    ):

        self.__verbose = verbose
        self.logger = setup_logger(verbose, "Model Logger")
        self.__id = uuid.uuid1()
        self._model = model
        self._config = config
        self._model_type = model_type
        self._name = name
        self._compile_params = None
        self._problem = ProblemUtils(
            kwargs.get("classification", True),
            kwargs.get("is_binary", False)
        )

        self.eval_metrics = []

        # Checkpoints path set?
        if checkpoint_path is None:
            checkpoint_path = os.getcwd()
        self._checkpoints = Checkpoint(checkpoint_path) if checkpoint is None else checkpoint



    def __call__(self, inputs, **kwargs):
        return self._model(inputs, **kwargs)


    def predict(self, inputs, **kwargs):
        """
            Approximate predictive distribution.

            Parameter:
                inputs (numpy.ndarray): The inputs for the approximation

        """
        return self._model.predict(inputs, **kwargs)
    

    def evaluate(self, inputs, targets, **kwargs):
        """
            Evaluate a model on given input data and targets.

            Parameters:
                inputs (numpy.ndarray):
                targets (numpy.ndarray):

            Returns:
                (list) A list with two values. [loss, accuracy]  
        """

        pred_targets = self.predict(inputs, **kwargs)
        output_metrics = {}
        for metric in self.eval_metrics:
            
            if isinstance(metric, str):
                metric = tf.keras.metrics.get(metric)

            metric_name = None
            if hasattr(metric, "__name__"):
                metric_name = metric.__name__
            
            else:
                metric_name = metric.name

            output_metrics[metric_name] = metric(targets, pred_targets)
            
        return output_metrics


    def fit(self, *args, **kwargs):
        """
            Fit the model to the given data.

            Args:
                x (numpy.ndarray): The inputs to train the model on. (default=None)
                y (numpy.ndarray): The targets to fit the model to. (default=None)
                batch_size (int): The size of each individual batch

            Returns:
                () a record of the trianing procedure
        """

        if self._config is not None and "fit" in self._config and isinstance(self._config["fit"], dict):
            fit_params = self._config["fit"]
            kwargs.update(fit_params)

        return self._model.fit(*args, **kwargs)


    def compile(self, *args, **kwargs):
        """
            Compile the model if needed
        """
        self._compile_params = kwargs
        self._model.compile(**kwargs)

        # Create evaluation metrics from compile parameters
        metrics = self._create_init_metrics(kwargs)
        if self.eval_metrics == []:
            self.eval_metrics = metrics


    def disable_batch_norm(self):
        """
            Disable batch normalization for activation of dropout during prediction.

            Parameters:
                - model (tf.Model) Tensorflow neural network model.
        """
        
        disabled = False
        for l in self._model.layers:
            if l.__class__.__name__ == "BatchNormalization":
                disabled = True
                l.trainable = False

        if disabled:
            self.logger.info("Disabled BatchNorm-Layers.")


    def clear_session(self):
        tf.keras.backend.clear_session()



    # ------
    # Model runtime configurations
    # -----------------------------

    def reset(self, pool, dataset):
        """
            Use to reset states, weights and other stuff after each active learning loop iteration.

            Parameters:
                pool (Pool): The pool managing labeled and unlabeled indices.
                dataset (Dataset): The dataset containting the different splits.
        """
        self._model = tf.keras.models.clone_model(self._model)
        self._model.compile(**self._compile_params)


    def optimize(self, inputs, targets):
        """
            Use to perform optimization during active learning loop.
        """
        pass


    # ----------------------
    # Utilities
    # ------------------------------

    def batch_prediction(self, inputs, batch_size=1, **kwargs):
        """
            
            Parameters:
                inputs (numpy.ndarray): Inputs going into the model
                n_times (int): How many times to sample from posterior?
                batch_size (int): In how many batches to split the data?
        """
    
        if batch_size < 1:
            raise ValueError("Error in McDropout.__call__(). Can't select negative amount of batches.")

        total_len = len(inputs)
        num_batches = math.ceil(total_len/batch_size)
        batches = np.array_split(inputs, num_batches, axis=0)

        predictions = []
        for batch in batches:
            predictions.append(self._model(batch, training=True))

        return np.vstack(predictions)


    def _create_init_metrics(self, kwargs):
        """
            Extract loss and other metrics passed to Model.compile()
            and transfer them to metrics which should be used for evaluation.
        """

        loss = kwargs.get("loss", None)
        metrics = kwargs.get("metrics", [])

        if not isinstance(metrics, list):
            metrics = [metrics]

        if loss is not None:
            metrics = [loss] + metrics
        
        return metrics


    def _extract_metric_names(self, metrics):
        """
            Get the metric names from a list of metric classes, 
            passing already existing metric names on.

            Parameters:
                metrics (list()): A list can contain strings and metric objects.
            
            Returns:
                (list(str)) a list of metric names
        """
        
        all_metrics = []
        for metric in metrics:

            metric_name = metric
            if not isinstance(metric, str):
                metric_name = metric.name
            
            all_metrics.append(metric_name)
        
        return all_metrics
        

    def _init_metrics(self, prefix, metrics):
        """
            Initializes metrics passed to the object during compilation.
            Using model specific metrics instead. 

            Parameters:
                prefix (str): The sub-package of tf_al.stats to be used (sampling, stochastic, ...)
                metrics (list(str)): A list of metric names.
            
            Returns:
                (list) of initialized metrics
        """
        initialized_metrics = []
        for metric in metrics:
            initialized_metrics.append(get(prefix, metric))

        return initialized_metrics




    # --------------
    # Checkpoint creation/loading
    # ------------------------------

    def empty_checkpoint(self):
        return self._checkpoints.empty()

    def new_checkpoint(self):
        self._checkpoints.new(self._model)

    
    def load_checkpoint(self, iteration=None):
        self._checkpoints.load(self._model, iteration)


    def clear_checkpoints(self):
        self._checkpoints.clean()


    def save_weights(self):
        path = self._checkpoints.PATH
        self._model.save_weights(path)


    def load_weights(self):
        path = self._checkpoints.PATH
        self._model.load_weights(path)
        

    def has_save_state(self):
        try:
            self.load_weights()
            return True

        except:
            return False



    # -----------
    # Access Configuration
    # ----------------------------

    def get_config_for(self, name):
        if (self._config is not None) and name in self._config:
            return self._config[name]
        
        return {}


    def get_config(self):
        return self._config.kwargs


    # --------------
    # Access important flags for predictions
    # ----------------------------

    def is_classification(self):
        return self._problem.classification


    def is_binary(self):
        return self._problem.is_binary


    # ---------------
    # Acquisition functions
    # --------------------------

    def get_query_fn(self, name):
        """
            Get model specific acquisition function.

            Parameters:
                name (str): The name of the acquisition function to return.

            Returns:
                (function) the acquisition function to use.
        """
        pass


    def __max_entropy(self, data, **kwargs):
        pass

    def __bald(self, data, **kwargs):
        pass

    def __max_var_ratio(self, data, **kwargs):
        pass

    def __std_mean(self, data, **kwargs):
        pass
    

    # -------------
    # Metric hooks
    # -----------------

    # def _on_evaluate_loss(self, **kwargs):
    #     pass

    
    # def _on_evaluate_acc(self, **kwargs):
    #     pass


    # -----------------
    # Setter/-Getter
    # --------------------------

    def get_id(self):
        return str(self.__id)

    def get_model_name(self, prefix=True):
        """
            Returns the model name.

            Parameters:
                prefix (bool): Prefix the model name with model type?

            Returns:
                (str) the model name.
        """

        # Model name and model type?
        model_name_exists = self._name is not None
        if self._model_type is not None:
            if model_name_exists and prefix:
                return self._model_type.lower() + "_" + self._name

            return self._model_type.lower()
        
        # Only model name given
        if model_name_exists:
            return self._name

        # Default model name
        return "model"
        

    def get_model(self):
        return self._model

    def get_metric_names(self):
        return self._model.metrics_names

    def get_base_model(self):
        return self._model
        

    # ---------------
    # Dunder
    # ----------------------
    
    def __str__(self):
        return self.get_model_name()

    def __getstate__(self):
        """
            Called when pickeling the object.
            Remove logger, because no use to pickle logger.
        """
        d = dict(self.__dict__)
        del d["logger"]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        verbose = d["__verbose"]
        self.logger = setup_logger(verbose, "Model Logger")
