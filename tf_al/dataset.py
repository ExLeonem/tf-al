from enum import Enum
from sklearn.model_selection import train_test_split
from . import Pool


class Dataset:
    """
        Splits a dataset into tree parts. Train/Test/validation.
        The train split is used for selection of 
        

        Parameters:
            inputs (numpy.ndarray): The model inputs.
            targets (numpy.ndarray): The targets, labels or values.
            init_size (int): The initial size of labeled inputs in the pool.
            train_size (float|int): Size of the train split.
            test_size (float|int): Size of the test split.
            val_size (float|int): Size of the validation split.
    """


    def __init__(
        self, 
        inputs,
        targets,
        test=None,
        val=None,
        init_size=0,
        init_indices=None
        # train_size=.75, 
        # test_size=None, 
        # val_size=None
    ):

        self.pseudo = True
        self.init_size = init_size

        self.x_train = inputs
        self.y_train = targets
        
        if test is not None:
            self.x_test, self.y_test = test
        
        if val is not None:
            self.x_test, self.y_test = val
        

        # if len(inputs) != len(targets):
        #     raise ValueError("Error in Dataset.__init__(). Can't initialize dataset. Length of inputs and targets are not equal.")

        # train_size, test_size, val_size = self.__init_sizes(len(inputs), train_size, test_size, val_size)

        # if train_size == 1 and test_size == 0 and val_size == 0:
        #     self.x_train = inputs
        #     self.y_train = targets

        # else:
        #     self.x_train, x_test, self.y_train, y_test = train_test_split(inputs, targets, train_size=train_size)

        #     if test_size != 0 and test_size + train_size == 1:
        #         self.x_test = x_test
        #         self.y_test = y_test

        #     else:
        #         pass


    def __init_sizes(self, data_size, train_size, test_size, val_size):
        """
            Initialize sizes for the different sets.
            Throwing errors when catching sight of disallowed situations.

            Parameters:
                data_size (int): the size of the dataset.
                train_size (int): the size of the training set.
                test_size (int): the size of the test set.
                val_size (int): the size of the evaluate set.

            Return:
                (tuple) of sizes (train_size, test_size, val_size)
        """

        # 1. Cast everything to procentual proportion of dataset
        train_size = self.__cast_to_float(data_size, train_size)
        test_size = self.__cast_to_float(data_size, test_size, "test")
        val_size = self.__cast_to_float(data_size, val_size, "evaluate")

        # [(train, .75), (test, .25), (eval, .0)]
        sets = []
        if train_size is not None:
            sets.append(("train", train_size))
        
        if test_size is not None:
            sets.append(("test", test_size))

        if val_size is not None:
            sets.append(("eval", val_size))

        # No splits of data?
        num_of_sets = len(sets)
        if num_of_sets == 0:
            raise ValueError("Error in Dataset.__init__(). No ")

        # Sum percentages
        # for set_idx in range(num_of_sets):
        #     set_name, set_size = 


    def __cast_to_float(self, total_size, part_size, set_name="train"):
        """
            Transform integer set size into float set size.

            Parameters:
                total_size (int): The dataset size.
                part_size (int|float): The amount of datapoints to select from the dataset.
                set_name (str): The set for which to transform the set size.
            
            Returns:
                (float) the transformed set size, as percentual part of the dataset.
        """

        is_float = isinstance(part_size, float)
        is_integer = isinstance(part_size, int)
        if (is_float or is_integer) and part_size < 0:
            raise valueError("Error in Dataset.__init__(). Can't select negative number of datapoints for {} set.".format(set_name))

        percentage = 0
        if is_float:
            percentage = part_size
        
        elif is_integer:
            percentage = part_size/total_size
            return part_size/total_size
            
        else:
            return .0

        # More than 100% to select?
        if percentage > 1.0:
                raise ValueError("Error in Dataset.__init__(). Can't select more than datapoints than available for {} set.".format(set_name))
        
        return percentage

    # ----------
    # Utilities
    # -------------------

    def is_pseudo(self):
        return self.pseudo

    def has_test_set(self):
        return hasattr(self, 'x_test') and hasattr(self, 'y_test')

    def has_eval_set(self):
        return hasattr(self, 'x_val') and hasattr(self, 'y_val')


    def get_split_ratio(self):
        """

            Returns:
                (int, int, int) the split ratio between (train, test, eval) sets.  
        """

        train_size = len(self.x_train)
        test_size = 0
        if hasattr(self, "x_test"):
            test_size = len(self.x_test)
        
        eval_size = 0
        if hasattr(self, "x_eval"):
            eval_size = len(self.x_eval)
        
        return (train_size, test_size, eval_size)


    def percentage_of(self, total_number, part):
        """
            Calculates the percentage a part takes from given total number.

            Parameters:
                total_number (int): The total number from which to calculate the percentual part.
                part (int): The part of which to calculate the percentage.

            Returns:
                (float) representing the percentage of given part im total number.
        """
        return part/total_number

    
    def check_int_in_range(self, value):
        """

        """
        pass


    def check_float_range(self, value):
        """
            Is float in procentual range?

            Parameters:
                value (float): The value to perform the check on.
        """
        return (value < 1) and (value > 0 )


    # -------------
    # Setter/-Getter
    # -------------------

    def get_init_size(self):
        return self.init_size

    def get_train_split(self):
        return (self.x_train, self.y_train)

    def get_train_inputs(self):
        return self.x_train

    def get_train_targets(self):
        return self.y_train

    def get_test_split(self):
        return (self.x_test, self.y_test)

    def get_eval_split(self):
        return (self.x_val, self.y_val)

