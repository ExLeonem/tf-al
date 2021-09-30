from functools import reduce
import numpy as np
from sklearn.cluster import k_means



def sbc(initial_size, x_unlabeled, y_labels=None):
    x_flat_shape = (len(x_unlabeled), reduce(lambda x, y: x*y, x_unlabeled.shape[1:], 1))
    x_flattened = x_unlabeled.reshape(x_flat_shape)
    centeroid, _label, _intertia = k_means(x_flattened, initial_size)

    all_indices = np.arange(len(x_unlabeled))
    indices_selected = []
    for idx in range(initial_size):
        euc_norm_distance = np.sqrt(np.sum(np.power(centeroid[idx], 2)))
        euc_norm_samples = np.sqrt(np.sum(np.power(x_flattened, 2), axis=-1))
        distance = np.sum(centeroid[idx] * x_flattened, axis=-1)/(euc_norm_distance*euc_norm_samples)

        sample_to_select = np.argmin(distance)
        indices_selected.append(all_indices[sample_to_select])
        x_flattened = np.delete(x_flattened, sample_to_select, axis=0)
        all_indices = np.delete(all_indices, sample_to_select, axis=0)

    return np.array(indices_selected)




