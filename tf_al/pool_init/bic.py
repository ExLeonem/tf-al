from functools import reduce
import numpy as np
from sklearn.cluster import k_means


def __cosine_distance(from_point, to_points):
    """
        Calculate the cosine distance from a single points
        to multiple other points.
    """
    euc_from = np.sqrt(np.sum(np.power(from_point, 2)))
    euc_to = np.sqrt(np.sum(np.power(to_points, 2), axis=-1))
    return np.sum(from_point*to_points, axis=-1)/(euc_from*euc_to)


def __small_cluster(centeroid, x_flattened):
    euc_norm_distance = np.sqrt(np.sum(np.power(centeroid, 2)))
    euc_norm_samples = np.sqrt(np.sum(np.power(x_flattened, 2), axis=-1))
    distance = np.sum(centeroid * x_flattened, axis=-1)/(euc_norm_distance*euc_norm_samples)
    sample_to_select = np.argmin(distance, axis=0)
    return sample_to_select


def __assign_to_center(centeroids, x_flattened):
    """
        Assign each datapoints of x_flattened to one of the past centeroids.

        Parameters:
            centeroids (list(int)): Indices of centeroids.
            x_flattened (numpy.ndarray): The flattened data.
    """

    





def __bic_estimate(centeroid_idx, centeroid, x_flattened, labels):

    # Select two points fartest away from one another
    max_distance = -100
    center_1 = None
    center_2 = None
    for dp_idx in range(len(x_flattened)):

        distances = __cosine_distance(x_flattened[dp_idx], x_flattened)
        max_idx = np.argmax(distances, axis=0)
        
        if distances[max_idx] > max_distance:
            max_distance = distances[max_idx]
            center_1 = dp_idx
            center_2 = max_idx


    # Assign rest of datapoints to of two centers
    ds_1, ds_2 = __assign_to_center([center_1, center_2], x_flattened)


    


def __big_cluster():
    pass



def bic(initial_size, x_unlabeled, y_labels=None, threshold=100):
    
    x_flat_shape = (len(x_unlabeled), reduce(lambda x, y: x*y, x_unlabeled.shape[1:], 1))
    x_flattened = x_unlabeled.reshape(x_flat_shape)
    centeroids, labels, _inertia = k_means(x_flattened, initial_size)
    
    all_indices = np.arange(len(x_flattened))
    indices = []
    for idx in range(initial_size):

        # Small cluster, select closest to centeroid
        num_samples = np.sum(labels == idx)
        if num_samples < threshold:
            selected = __small_cluster(centeroids[idx], x_flattened)
            indices.append(all_indices[selected])
            all_indices = np.delete(all_indices, selected, axis=0)
            x_flattened = np.delete(x_flattened, selected, axis=0)
            continue

        # Big cluster
        bic = __bic_estimate(idx, centeroids[idx], x_flattened, labels[all_indices])




