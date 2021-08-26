
def name(function_name):
    """
        Set a name for the acquisition function.

        Parameters:
            name (str): The name used for the acquisition function.
    """

    print(function_name)

    if isinstance(function_name, object):
        raise ValueError("Error in @name(). Missing value for function_name.")

    if not isinstance(function_name, str):
        raise ValueError("Error in @name(fn_name). Expected a string for fn_name.")

    def decorator(cls):

        def get_default_name(self):
            return function_name

        cls.__get_default_name = get_default_name
        return cls

    return decorator        



class AcquisitionFunction:
    """
        
    """

    def __init__(self, fn_name="acquisition_function", **kwargs):
        
        # Default acquision function name overwritten by decorator?
        if hasattr(self, "__get_default_name"):
            __get_default_name = getattr(self, "__get_default_name")
            fn_name = __get_default_name()

        self._fn_name = fn_name
        
        # Set every kwarg as single object attribute
        for key, value in kwargs.items():

            if hasattr(self, key):
                raise ValueError("Error in AcquisitionFunction.__init__(). Can't set attribute \"{}\", attribute already exists.".format(key))

            setattr(self, key, value)


    def __call__(self, predictions, **kwargs):
        pass


    def select_max(self, values, num_to_select=1):
        """
            Select the first n-datapoints that are max. 

            Parameters:
                values (numpy.ndarray): Calculated values after execution of the acquisition function.
                num_to_select (int): The number of datapoints to select.
            
            Returns:
                (numpy.ndarray) the indices of datapoints selected out of original unlabeled datapoints.
        """
        pass

    
    def select_min(self, values, num_to_select=1):
        """
            Select the first n-datapoints that are min. 

            Parameters:
                values (numpy.ndarray): Calculated values after execution of the acquisition function.
                num_to_select (int): The number of datapoints to select.
            
            Returns:
                (numpy.ndarray) the indices of datapoints selected out of original unlabeled datapoints.
        """
        pass


    # ------------
    # Setter/-Getter
    # ---------------------

    def get_fn_name(self):
        return self._fn_name

    def _set_fn_name(self, fn_name):
        self._fn_name = fn_name
    

    def _get_param(self, key, default=None):

        if self._kwargs is None:
            return default

        if not isinstance(self._kwargs, dict):
            raise ValueEror
        
        result = self._kwargs.get(key)        
        return result if result is not None else default

