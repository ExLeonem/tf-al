import math
import numpy as np

import tensorflow as tf
from .utils import setup_logger


class AcquisitionFunction:
    """
        Query a model for next datapoints that should be labeled.
        Already implemented bayesian models implement the following acquisition
        functions:

        - Max Entropy (\"max_entropy\")
        - Bald (\"bald\")
        - Variation Ratios (\"max_var_ratio\")
        - Mean standard deviation (\"mean_std\")
        - Randomized selection (\"random\")

        Parameters:
            fn_name (str): The name of the acquisition function to use.
            batch_size (int): The number of batches to split the data into for processing. (default=None)
            verbose (bool): Apply debugging? (default=False)

        Attributes:
            name (str): The name of the acquisition function to apply.
            fn (function): The acquisition function to execute.
            batch_size (int): to configure the processing in batches (default=None)
    """

    def __init__(self, fn_name, batch_size=None, verbose=False, **kwargs):
        self.logger = setup_logger(verbose, "Acquisition Function Logger")

        self.name = fn_name
        self.fn = None
        self.batch_size = batch_size

        self.kwargs = kwargs

        # Set passed kwargs as additional attributes, not overwriting existing
        # for key, value in kwargs.items():
        #     if not hasattr(self, key):
        #         setattr(self, key, value)


    def __call__(self, model, pool, step_size=20, **kwargs):
        """
            
            Parameter:
                model (Model): The model to use for the computation of acquistion functions.
                pool (Pool): The pool of unlabeled data.
                step_size (int): Number of datapoints to collect for next active learning iteration.

            Returns:
                (numpy.ndarray) Indices
        """
        self.logger.info("Step-size: {}".format(step_size))
        self.logger.info("Function: {}".format(self.name))
        self.logger.info("Kwargs: {}".format(kwargs))

        # Set initial acquistion function
        if self.fn is None:
            self.fn = self._set_fn(model)

        # No need for batch processing, random selection of datapoints.
        data, indices = pool.get_unlabeled_data()
        if self.name == "random":
            return self.fn(indices, data, step_size=step_size, **kwargs)            

        # Iterate throug batches of data
        results = None
        num_datapoints = len(data)
        self.logger.info("Unlabeled Pool size: {}".format(num_datapoints))
           
        # Single batch processing
        if self.batch_size is None:
            self.batch_size = len(data)

        # Multi batch Processing
        num_batches = math.ceil(num_datapoints/self.batch_size)
        batches = np.array_split(data, num_batches, axis=0)
        results = []
        for batch in batches:
            
            # Fix some of tf memory leak issues
            input_tensor = tf.convert_to_tensor(batch)
            output_tensor = self.fn(input_tensor, **kwargs)
            results.append(output_tensor)
        
        stacked = np.hstack(results)
        num_of_elements_to_select = self._adapt_selection_num(len(stacked), step_size)
        return self.__select_first(stacked, indices, num_of_elements_to_select)


    def _adapt_selection_num(self, num_indices, num_to_select):
        """
            Check if n datapoints are available at all, else adapt the number of datapoints to select.

            Parameters:
                num_indices (int): The amount of total indices to select from
                num_to_select (int): The number of datapoints to select

            Returns:
                (int) the adapted number datapoints to select from a collection.
        """
        if num_indices == 0:
            raise ArgumentError("Can't select {} datapoints, all data is labeled.".format(num_to_select))

        if num_indices < num_to_select:
            return num_indices
        
        return num_to_select


    def _set_fn(self, model):
        """
            Set the function to use for acquisition.
            
            Parameters:
                name (str): The name of the acquisition to use.

            Returns:
                (function): The function to use for acquisition.
        """
        query_fn = model.get_query_fn(self.name)
        if query_fn is None:
            self.logger.debug("Set acquisition function: random baseline.")
            self.name = "random"
            return self._random

        return query_fn
    

    def _random(self, indices, data, step_size=5, **kwargs):
        """
            Randomly select a number of datapoints from the dataset.
            Baseline for comparison purposes.

            FIX: Random selection is wrong!!!

            Parameters:
                model (BayesianModel): The model to perform active learning on.
                pool (DataPool): The pool of data to use.
                num (int): Numbers of indices to draw from unlabeled data.
           
            Returns:
                (numpy.ndarray): Randomly selected indices for next training.
        """
        available_indices = np.linspace(0, len(data)-1, len(data), dtype=int)
        step_size = self._adapt_selection_num(len(available_indices), step_size)
        selected = np.random.choice(available_indices, step_size, replace=False).astype(int)
        return indices[selected], data[selected]


    def __select_first(self, predictions, indices, n):
        """
            Select n biggest elements from k- predictions.

            Parameters:
                predictions (numpy.ndarray): The predictions made by the network

            Returns: 
                (numpy.ndarray) indices of n-biggest predictions.
        """
        
        sorted_keys = np.argsort(predictions)
        n_biggest_keys = sorted_keys[-n:]
        return indices[n_biggest_keys], predictions[n_biggest_keys]


    # ----------
    # Dunder
    # ------------------

    def __str__(self):
        return self.name


    # ----------
    # Getter-/Setter
    # ------------------
    
    def set_fn(self, fn):
        self.fn = fn

    def get_name(self):
        return self.name