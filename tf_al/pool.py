import math
from copy import deepcopy
import numpy as np


class Pool:
    """
        Pool that holds information about labeled and unlabeld inputs.
        The attribute 'indices' holds information about the labeled inputs.
        
        Each value of self.indices can take the following states:
        (value==-1) Corresponding input is labeld
        (value!=-1) Corresponding input is not labeled 

        Parameters:
            inputs (numpy.ndarray): Inputs to the network.
    """

    def __init__(self, inputs, targets=None):
        self.__inputs = inputs
        self.__true_targets = targets
        self.__indices = np.linspace(0, len(inputs)-1, len(inputs), dtype=int)
        self.__targets = np.zeros(len(inputs))


    def init(self, size):
        """
            Initialize the pool with specific number of labels.
            Only applicable when pool in pseudo mode.

            Parameters:
                size (int|list|np.ndarray): Either the number of datapoints to initialized or an explicit list or array of indices to initialize.
        """
        
        is_int = isinstance(size, int)
        is_list = isinstance(size, list)
        is_np_array = isinstance(size, np.ndarray)

        if not self.is_pseudo():
            raise ValueError("Error in Pool.init(size). Can't initialize pool using init(size) when not in pseudo mode. Initialize pool with targets, to put Pool in pseudo mode.")

        if is_int and size < 1:
            raise ValueError("Error in Pool.init(size). Can't initialize pool with {} targets. Use a positive integer > 1.".format(size))

        if is_int and len(self.__indices) < size:
            raise ValueError("Error in Pool.init(size). Can't initialize pool, not enough targets. {} targets required, {} are available.".format(size, len(self.__indices)))

        if not (is_int or is_list or is_np_array):
            raise ValueError("Error in Pool.init(size). Expected size to be an integer, list or numpy array of indices.")

        # Initialize explicit indices
        if is_list or is_np_array:
            self.__init_explicit_indices(size)
            return

        # WARNING: Will only work for categorical targets
        unique_targets = np.unique(self.__true_targets)

        # Initialize n-datapoints per class
        num_to_select = 1
        num_unique_targets = len(unique_targets)
        if num_unique_targets < size:
            num_to_select = math.floor(size/num_unique_targets)    
        
        # Annotate samples in round robin like schme
        while size > 0:
            
            # Select 
            for target in unique_targets:

                unlabeled_indices = self.get_unlabeled_indices()
                true_targets = self.__true_targets[unlabeled_indices]
                selector = (true_targets == target)

                indices = unlabeled_indices[selector]
                targets = true_targets[selector]

                # Move to next target, when none available of this type
                if len(indices) == 0:
                    continue
                
                adapted_num_to_select = self.__adapt_num_to_select(targets, num_to_select)
                selected_indices = np.random.choice(indices, adapted_num_to_select, replace=False)
                
                # Update pool
                selected_targets = self.__true_targets[selected_indices]
                self.annotate(selected_indices, selected_targets)
                size -= num_to_select

                if size < 1:
                    break
    

    def __init_explicit_indices(self, indices):
        """
            Initializes the pool with ex

            Parameters:
                indices (list|numpy.ndarray): A list of indices which to use 
        """        
    
        try:
            unlabeled_indices = self.get_unlabeled_indices()
            true_targets = self.__true_targets[unlabeled_indices]

            selected_indices = unlabeled_indices[indices]
            selected_targets = true_targets[indices]
            self.annotate(selected_indices, selected_targets)

        except IndexError as e:
            raise IndexError("Error in Pool.init(size). " + str(e).capitalize() + ".")

    
    def __adapt_num_to_select(self, available, num_to_select):
        """
            Adapts the number of elements to select next.

            Parameters:
                available (numpy.ndarray): The available elements.
                num_to_select (int): The number of elements to select.

            Returns:
                (int) the adapted number of elements selectable.
        """

        num_available = len(available)
        if num_available < num_to_select:
            return num_available

        return num_to_select


    def get_inputs_by(self, indices):
        """
            Get inputs by indices.

            Parameters:
                indices (numpy.ndarray): The indices at which to access the data.

            Returns:
                (numpy.ndarray) the data at given indices.
        """
        return self.__inputs[indices]


    def get_targets_by(self, indices):
        """

        """
        return self.__targets[indices]


    def __setitem__(self, indices, targets):
        """
            Shortcut to annotate function.

            Parameters:
                indices (numpy.ndarray): For which indices to set values.
                targets (numpy.ndarray): The targets to set.
        """
        self.annotate(indices, targets)

    
    def annotate(self, indices, targets=None):
        """
            Annotate inputs of given indices with given targets.

            Parameters:
                indices (numpy.ndarray): The indices to annotate.
                targets (numpy.ndarray): The labels to set for the given annotations.
        """

        if targets is None:
            if self.__targets is None:
                raise ValueError("Error in Pool.annotate(). Can't annotate inputs, targets is None.")

            targets = self.__true_targets[indices]
            

        # Create annotation
        self.__indices[indices] = -1
        self.__targets[indices] = targets


    # ---------
    # Utilities
    # -------------------

    def has_unlabeled(self):
        """
            Has pool any unlabeled inputs?

            Returns:
                (bool) true or false depending whether unlabeled data exists.
        """
        selector = np.logical_not(self.__indices == -1)
        return np.any(selector)


    def has_labeled(self):
        """
            Has pool labeled inputs?

            Returns:
                (bool) true or false depending whether or not there are labeled inputs.
        """
        selector = self.__indices == -1
        return np.any(selector)


    def is_pseudo(self):
        """
            Is the pool in pseudo mode?
            Meaning, true target labels are already known?

            Returns:
                (bool) indicating whether or not true labels are existent.
        """
        return self.__true_targets is not None


    def __deepcopy__(self, memo):
        return Pool(self.__inputs, self.__true_targets)

    # ---------
    # Setter/Getter
    # -------------------

    def get_indices(self):
        """
            Returns the current labeling state.

            Returns:
                (numpy.ndarray) the indices state. (-1) indicating a labeled input.
        """
        
        return self.__indices

    def get_labeled_indices(self):
        """
            

            Returns:
                (numpy.ndarray) of datapoints that already has been labeled.
        """
        selector = self.__indices == -1
        indices = np.linspace(0, len(self.__inputs)-1, len(self.__inputs), dtype=int)
        return indices[selector]


    def get_unlabeled_indices(self):
        """
            Get all unlabeled indices for this pool.

            Returns:
                (numpy.ndarray) an array of indices.
        """
        selector = self.__indices != -1
        return self.__indices[selector]


    def get_length_labeled(self):
        """
            Get the number of labeled inputs.

            Returns:
                (int) The number of labeled inputs.
        """
        return np.sum(self.__indices == -1)

    
    def get_length_unlabeled(self):
        """
            Get the number of unlabeld inputs.

            Returns:
                (int) the number of unlabeled inputs
        """
        return np.sum(np.logical_not(self.__indices == -1))   


    def get_labeled_data(self):
        """
            Get inputs, target pairs of already labeled inputs.

            Returns:
                (tuple(numpy.ndarray, numpy.ndarray)) inputs and corresponding targets.
        """
        selector = self.__indices == -1
        inputs = self.__inputs[selector]
        targets = self.__targets[selector]
        return inputs, targets

    
    def get_unlabeled_data(self):
        """
            Get inputs whire are not already labeled with their indices.

            Returns:
                (tuple(numpy.ndarray, numpy.ndarray)) The inputs and their indices in the pool
        """
        selector = self.__indices != -1
        inputs = self.__inputs[selector]
        indices = self.__indices[selector]
        return inputs, indices