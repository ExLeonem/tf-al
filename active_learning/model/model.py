import os, sys, math
import logging
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from . import Checkpoint


class Mode(Enum):
    TRAIN=1,
    TEST=2,
    EVAL=3


class ModelType(Enum):
    """
        Different bayesian model types.
    """
    MC_DROPOUT=1,
    MOMENT_PROPAGATION=2,
    SWAG=3


class BayesModel:
    """
        Base class for encapsulation of a bayesian deep learning model. 

        Attributes:
            _model (tf.Model): Tensorflow or pytorch module.
            _config (Config): Model configuration
            _mode (Mode): The mode the model is in 'train' or 'test'/'eval'.
            _model_type (ModelType): The model type
            _checkpoints (Checkpoint): Created checkpoints.
    """

    def __init__(
        self, 
        model, 
        config=None, 
        mode=Mode.TRAIN, 
        name=None,
        model_type=None, 
        classification=True, 
        is_binary=False,
        verbose=False,
        **kwargs
    ):

        self.setup_logger(verbose)
        self._model = model
        self._config = config
        self._mode = mode
        self._model_type = model_type
        self._name = name

        self._checkpoints = Checkpoint()

        self.__classification = classification
        if not self.__classification:
            # Binary classification always false, when regression problem
            self.__is_binary = False
        else:
            self.__is_binary = is_binary


    def __call__(self, *args, **kwargs):
        return self._model(inputs, training=self.in_mode(Mode.TRAIN))


    def predict(self, inputs, **kwargs):
        """
            Approximate predictive distribution.

            Parameter:
                inputs (numpy.ndarray): The inputs for the approximation

        """
        return self._model(inputs, training=self.in_mode(Mode.TRAIN))
    

    def evaluate(self, inputs, targets, **kwargs):
        """
            Evaluate a model on given input data and targets.

            Parameters:
                inputs (numpy.ndarray):
                targets (numpy.ndarray):

            Returns:
                (list) A list with two values. [loss, accuracy]  
        """
        return self._model.evaluate(inputs, targets, **kwargs)


    def fit(self, *args, **kwargs):
        """
            Fit the model to the given data.

            Args:
                x (numpy.ndarray): The inputs to train the model on. (default=None)
                y (numpy.ndarray): The targets to fit the model to. (default=None)
                batch_size (int): The size of each individual batch

            Returns:

        """

        if self._config is not None and "fit" in self._config and isinstance(self._config["fit"], dict):
            fit_params = self._config["fit"]
            kwargs.update(fit_params)

        return self._model.fit(*args, **kwargs)


    def compile(self, *args, **kwargs):
        """
            Compile the model if needed
        """
        self._model.compile(**kwargs)


    def prepare_predictions(self, predictions):
        """
            Extend predictions for binary classification case.

            Parameters:
                predictions (numpy.ndarray): The predictions made by the model

            Returns:
                (numpy.ndarray) The extended numpy array
        """
        return predictions

    
    def map_eval_values(self, values):
        """
            Create a dictionary mapping for evaluation metrics.

            Parameters:
                values (any): Values received from model.evaluate

            Returns:
                (dict) The values mapped to a specific key.
        """
        metric_names = self._model.metrics_names
        return dict(zip(metric_names, values))


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

    def reset(self):
        """
            Use to reset states, weights and other stuff during each active learning loop iteration.
        """
        self.load_weights()


    def optimize(self, inputs, targets):
        """
        Use to optimize parameters during active learning loop        
        """
        pass


    # ----------------------
    # Utilities
    # ------------------------------

    def setup_logger(self, debug):
        """
            Setup a logger for the active learning loop

            Parameters:
                propagate (bool): activate logging output in console?
        """

        logger = logging.Logger("Runner")
        log_level = logging.DEBUG if debug else logging.CRITICAL

        logger.handler = logging.StreamHandler(sys.stdout)
        logger.handler.setLevel(log_level)
        
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        logger.handler.setFormatter(formatter)
        logger.addHandler(logger.handler)

        dir_name = os.path.dirname(os.path.realpath(__file__))
        log_path = os.path.join(dir_name, "..", "logs", "model.log")

        fh = logging.FileHandler(log_path)
        fh.setLevel(log_level)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
        self.logger = logger


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


    # ---------------
    # Loss function
    # -----------------------

    def nll(self, predictions, targets):
        """
            Calculate the negative log likelihood per element.

            NLL: -np.log(true_class_prob)

            Parameters:
                prediction (numpy.ndarray): 

            Returns:
                (numpy.ndarray) the NLL values.
        """

        num_datapoints = len(predictions)
        true_preds = np.zeros(num_datapoints)
        for i in range(num_datapoints):
            true_target_index = targets[i]
            true_preds[i] = predictions[i][true_target_index]

        return -np.log(true_preds)


    def entropy(self, predictions):
        """
            Calculate the shannon entropy per datapoint.

            Parameters:
                prediction (numpy.ndarray): The predictions made by the network.
            
            Returns:
                (numpy.ndarray) the entropy values.
        """
        return np.sum(-(predictions*np.log2(predictions+1e-10)), axis=-1)


    # -----------
    # Access Configuration
    # ----------------------------

    def get_fit_config(self):
        if (self._config is not None) and hasattr(self._config, "fit"):
            return self._config["fit"]
        
        return {}


    def get_query_config(self):
        if (self._config is not None) and hasattr(self._config, "query"):
            return self._config["query"]

        return {}


    # --------------
    # Access important flags for predictions
    # -----------------------------

    def in_mode(self, mode):
        return self._mode == mode


    def is_classification(self):
        return self.__classification


    def is_binary(self):
        return self.__is_binary


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
        

    # -----------------
    # Setter/-Getter
    # --------------------------

    def get_model_name(self, prefix=True):
        """
            Returns the model name.
        """

        model_type = None
        if not (self._model_type is None) and prefix:
            _pre, model_type = str(self._model_type).split(".")
            model_typ3 = model_type.lower()


        # 
        if not (self._name is None): 
            if not (model_type is None) and prefix:
                return model_type + "_" + sel._name

            return self._name

        # No specific name set
        if not (model_type is None):
            return model_type

        return "model"
        


    def get_model_type(self):
        return self._model_type

    def get_model(self):
        return self._model

    def get_mode(self):
        return self._mode

    def get_metric_names(self):
        return self._model.metrics_names

    def set_mode(self, mode):
        self._mode = mode

    # ---------------
    # Dunder
    # ----------------------

    def __eq__(self, other):
        return other == self._model_type


    def __str__(self):
        return self.get_model_name()

