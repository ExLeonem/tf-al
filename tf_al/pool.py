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
            targets (numpy.ndarray): Already known targets, used for experimental runs. (default=None)
            target_shape (tuple()): The shape of the target, if None equals the len(inputs). (default=None)
    """

    def __init__(self, inputs, targets=None, target_shape=None):
        self.__inputs = inputs
        self.__true_targets = targets
        self.__indices = np.linspace(0, len(inputs)-1, len(inputs), dtype=int)

        if targets is not None:
            self.__targets = np.zeros(targets.shape)
        
        elif target_shape is None:
            self.__targets = np.zeros(len(inputs))

        else:
            self.__targets = np.zeros(target_shape)    


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

        # Initialize pool with one-hot-vector labels
        true_target_shape = self.__true_targets.shape
        if len(true_target_shape) > 1 and true_target_shape[-1] > 1:
            self.__init_with_one_hot_vectors(size)
            return

        # WARNING: Will only work for categorical targets
        # Initialize n-datapoints per class
        unique_targets = np.unique(self.__true_targets)
        num_to_select = self.__adapt_init_size(size, len(unique_targets))
        while size > 0:
            # Annotate samples in round robin like schme
            
            # TODO: unique targets may be one-hot vector or float in regression case
            for target in unique_targets:

                unlabeled_indices = self.get_unlabeled_indices()
                true_targets = self.__true_targets[unlabeled_indices]
                selector = (true_targets == target)

                # Selector may be multi-dimensional array -> needs to be flattened 
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
                size -= adapted_num_to_select

                if size < 1:
                    break
    
                
    def __init_with_one_hot_vectors(self, size):
        true_label_index = np.argmax(self.__true_targets, axis=-1)
        num_labels = np.unique(true_label_index)
        num_to_select = self.__adapt_init_size(size, len(num_labels))

        while size > 0:

            for target in num_labels:

                # Get one-hot encoded labels
                unlabeled_indices = self.get_unlabeled_indices()
                true_targets = np.argmax(self.__true_targets, axis=-1)[unlabeled_indices]
                selector = (true_targets == target)

                # 
                indices = unlabeled_indices[selector]
                targets = true_targets[selector]

                # No datapoints for current target available
                if len(indices) == 0:
                    continue
            
                adapted_num_to_select = self.__adapt_num_to_select(targets, num_to_select)
                selected_indices = np.random.choice(indices, adapted_num_to_select, replace=False)

                # Update pool
                selected_targets = self.__true_targets[selected_indices]
                self.annotate(selected_indices, selected_targets)
                size -= adapted_num_to_select
                
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

    def __adapt_init_size(self, size, available):
        num_to_select = 1
        if available < size:
            num_to_select = math.floor(size/available)   
        
        return num_to_select


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
            Get the indices of labeled datapoints.
            
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
            Get data and indices of datapoints which are currently labeled.

            Returns:
                (tuple(numpy.ndarray, numpy.ndarray)) inputs and corresponding targets.
        """
        selector = self.__indices == -1
        inputs = self.__inputs[selector]
        targets = self.__targets[selector]
        return inputs, targets

    
    def get_unlabeled_data(self):
        """
            Get data and their indices of datapoints which are currently not labeled. 

            Returns:
                (tuple(numpy.ndarray, numpy.ndarray)) The inputs and their indices in the pool
        """
        selector = self.__indices != -1
        inputs = self.__inputs[selector]
        indices = self.__indices[selector]
        return inputs, indices