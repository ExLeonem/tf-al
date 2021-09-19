from types import FunctionType



class BaseMetric:
    """
        A simple metric object. Encapsulating the callback function
        for a specific custom metric as well as the prefix to be applied to the 
        returned metric keys.
    """

    def __init__(self, callback: FunctionType, prefix: str):

        if not isinstance(callback, FunctionType) and not hasattr(callback, "__call__"):
            raise ValueError("Error in BaseMetric.__init__(callback, prefix). Expected callback to be a function.")

        self.__callback = callback
        self.prefix = prefix

    
    def __call__(self, *args) -> dict:
        return self.__callback(*args)



class ModelSpecificMetric:
    """
        Specify metrics that will be applied only to specific types of model types.

        Parameters:
            prefix (str): a prefix that will prepended to the metric returned by the callback.
    """

    def __init__(self, prefix: str=None):
        self.prefix = prefix
        self.__callbacks = {}

    
    def __call__(self, model_type: str, *args):
        """
            Applies a callback function on the arguments when registered for given
            model type. 

            Following parameters will be passed down in order as args to the callback.

            Parameters:
                    prediction (numpy.ndarray): The predictions made by the model.
                    inputs (numpy.ndarray): The inputs values to the network.
                    true_targets (numpy.ndarray): The true target values of the dataset.
        """
        if model_type in self.__callbacks.keys():
            return self.__callbacks[model_type](*args)

        return  {}
    

    def add_callback(self, model_type: str, callback: FunctionType):
        """
            Register callback methods to be applied to models of a specific type.
        
            Parameters:
                model_type (str): The model type the callback function will be applied on.
                callback (function): 
        """

        if not isinstance(callback, FunctionType):
            raise ValueError("Error in ModelSpecificMetric.add_metric(). Expected parameter callback to be a function.")

        self.__callbacks[model_type] = callback



class MetricsAccumulator:
    """
        Collect metrics from the active learning loop to be added to
        the output per iteration.

        Allows definition of custom callback functions to be applied to
        the evaluation step of the active learning loop.

        Predefined metrics that will be collected are:
            - train_time: The time it took to train the model.
            - optim_time: The time used to execute the optimization
            - eval_time: The time it took to perform the evaluation step 

        Parameters:
            metrics (str|list(str)): A name or a list of predefined metric names. (default=None) 
    """

    def __init__(self):
        self.__metrics = []

        
    def __call__(self, *args, **kwargs) -> dict:
        """
            Applies the callback functions and passes the
            parameters down as kwargs.

            Following parameters are getting passed down to the metric callback as 
            in order as args.

            Parameters:
                model_type (str): The model type in lower case letters. 
                prediction (numpy.ndarray): The predictions made by the model.
                inputs (numpy.ndarray): The inputs values to the network.
                true_targets (numpy.ndarray): The true target values of the dataset.

            Returns:
                (dict) all metrics accumulated
        """
        all_metrics = {}
        for metric in self.__metrics:
            outputs = metric(*args, **kwargs)
            outputs = self.__add_prefix(metric.prefix, outputs)
            all_metrics.update(outputs)

        return all_metrics

    
    def track(self, callback: FunctionType) -> None:
        """
            Adds a callback method for metric accumulation. The callback function
            should receive a dictionary of values and return a dictionary with unique keys.

            Pre-defined keys that should not be used are:

                - train_loss
                - train_<accuracy_name>
                - train_time
                - query_time

            Parameters:
                callback (function|BaseMetric|ModelSpecificMetric): A function return a dictionary of metrics.
        """

        if isinstance(callback, BaseMetric):
            self.__metrics.append(callback)
            return
        
        elif isinstance(callback, ModelSpecificMetric):
            self.__metrics.append(callback)
            return

        elif not isinstance(callback, FunctionType) and not hasattr(callback, "__call__"):
            raise ValueError("Error in MetricAccumulator.add_metric(). Expected parameter callback to be of type BaseMetric, ModelSpecificMetric or function. Got {}".format(type(callback)))


        # Function passed
        new_metric = BaseMetric(callback, None)
        self.__metrics.append(new_metric)


    # def add_filter(self, names):
    #     """
    #         Pre-filter metrics to outputed per active learning iteration.

    #         Parameters:
    #             names (str|list(str)): A single metric name or list of names to be remove from the iteration output.
    #     """
    #     pass



    # --------------
    # Utilties
    # -------------------

    def __add_prefix(self, prefix: str, outputs: dict) -> dict:
        """
            Adds the given prefix to the metric names.

            Parameters:
                prefix (str): A string to be prepended to the metric names.

            Returns:
                (dict) output of previous metric callback with added prefix to the metric keys.
        """

        if prefix is None:
            return outputs

        new_outputs = {}
        for key, value in outputs.items():
            new_key = prefix + "_" + key
            new_outputs[new_key] = value
        
        return new_outputs