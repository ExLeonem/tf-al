import numpy as np

def init_pools(unlabeled_pool, labeled_pool, targets, num_init_per_target=10):
    """
        Initialize the pool with randomly selected values.

        Paramters:
            unlabeled_pool (UnlabeldPool): Pool that holds information about unlabeld datapoints.
            labeled_pool (LabeledPool): Pool that holds information about labeled datapoints.
            targest (numpy.ndarray): The labels of input values.
            num_init_per_target (int): The initial labels used per target. 

        Todo:
            - Make it work for labels with additional dimensions (e.g. bag-of-words, one-hot vectors)
    """

    # unlabeled_pool.update(indices)
    # labeled_pool[indices] = labels

    # Use initial target values?
    if num_init_per_target <= 0:
        return

    # Select 'num_init_per_target' per unique label 
    unique_targets = np.unique(targets)
    for idx in range(len(unique_targets)):

        # Select indices of labels for unique label[idx]
        with_unique_value = targets == unique_targets[idx]
        indices_of_label = np.argwhere(with_unique_value)

        # Set randomly selected labels
        selected_indices = np.random.choice(indices_of_label.flatten(), num_init_per_target, replace=True)

        # WILL NOT WORK FOR LABELS with more than 1 dimension
        unlabeled_pool.update(selected_indices)
        new_labels = np.full(len(selected_indices), unique_targets[idx])
        labeled_pool[selected_indices] = new_labels