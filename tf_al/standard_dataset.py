import tensorflow as tf
import tensorflow.keras as keras

from .dataset import Dataset


class KerasDataset(Dataset):
    """
        Load standard datasets of tensorflow/keras, split and 

        Supported names:
            - mnist
            - cifar10
            - cifar100
            - reuters
            - imdb
            - fashion_mnist
            - boston_housing

        Parameters:
            name (str): The name of the keras dataset to load.
            init_size (int): Initial labeled pool size.
            init_indices (list|numpy.ndarray): List or array of initial labeled pool indices.
            transform_inputs (function): Transformation applied on train and test inputs (x_train, x_test).
            transform_outputs (function): Transformation applied on train and test targets (y_train, y_test).
            **kwargs: Additional parameters, passed on to the load_data function
    """

    def __init__(self, name, init_size=0, init_indices=None, transform_inputs=None, transform_outputs=None, **kwargs):
        (x_train, y_train), (x_test, y_test) = self.__load_dataset(name, **kwargs)

        x_train = self.__transform_data(transform_inputs, x_train)
        y_train = self.__transform_data(transform_outputs, y_train)

        x_test = self.__transform_data(transform_inputs, x_test)
        y_test = self.__transform_data(transform_outputs, y_test)

        super().__init__(
            x_train, 
            y_train,
            test=(x_test, y_test),
            init_size=init_size,
            init_indices=init_indices
        )

    
    def __transform_data(self, fn_transform, data):
        """
            Apply transformation on the dataset.
        """
        if fn_transform is None:
            return

        return fn_transform(data)


    def __load_dataset(self, name, **kwargs):
        """
            Load a standard keras dataset by it's name. A list of datasets can be found [here](https://keras.io/api/datasets/).

            Supported names:
            - mnist
            - cifar10
            - cifar100
            - reuters
            - imdb
            - fashion_mnist
            - boston_housing

            Parameters:
                name (str): The name of the dataset
                **kwargs (dict): Dataset specific parameters. Depending on the dataset loaded.
        """

        name = name.lower()
        if name == "mnist":
            path = kwargs.get("path", "mnist.npz")
            return keras.datasets.mnist.load_data(path=path)

        elif name == "cifar10":
            return keras.datasets.cifar10.load_data()
        
        elif name == "cifar100":
            label_mode = kwargs.get("label_mode", "fine")
            return keras.datasets.cifar100.load_data(label_mode=label_mode)

        elif name == "imdb":
            return keras.datasets.imdb.load_data(**kwargs)

        elif name == "reuters":
            return keras.datasets.reuters.load_data(**kwargs)

        elif name == "fashion_mnist":
            return keras.datasets.fashion_mnist.load_data()

        elif name == "boston_housing": 
            params = self.__filter_kwargs(["path", "test_split", "seed"], kwargs)
            return keras.datasets.boston_housing.load_data(**params)

        raise ValueError("Error in StandardDataset.__init__(). Can't find keras dataset for name {}.".format(name))


    # -------------------
    # Utilities
    # -------------------------
    
    def __filter_kwargs(self, names, kwargs):
        """
            Collect specific set of names from **kwargs.

            Parameters:
                names (list(str)): A list of parameter names.
                kwargs (dict): A dictionary of parameters.

            Returns:
                (dict) the collected parameters.
        """

        parsed = {}
        for key, value in kwargs.items():

            if key in names:
                parsed[key] = value

        return parsed


    def __collect_and_set_defaults(self, names_and_defaults, **kwargs):
        """
            Deprecated

            Parameters:
                names_and_defaults (dict): A dictionary of parameter names and defaults.
                **kwargs (dict): Parameters received from user.

            Returns:
                (dict) with loaded parameters, while eventually setting default params.
        """

        parsed = {}
        for name, default in names_and_defaults.items():
            parsed[name] = kwargs.get(name, default)

            if name in kwargs:
                del kwargs[name]
            
        
        return parsed, kwargs