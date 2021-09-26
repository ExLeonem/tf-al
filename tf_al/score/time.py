import numpy as np


def runtime(times, pool_sizes=None):
    """
        Calculate the runtime for given inputs. If an array with pool sizes is given
        for each timestep, the times will get normalized showing runtime per datapoint.

        Parameters:
            times (numpy.ndarray): The times for a specific operation per step x
            pool_sizes (numpy.ndarray): The sizes of the unlabeled or labeled pool. Used to clear dataset dependence. (default=None)

        Returns:
            (tuple(numpy.array, numpy.array)) the mean runtime and standard deviation.
    """


    if pool_sizes is not None:
        times = times/pool_sizes

    time_mean = np.mean(times)
    time_std = np.std(times)
    return time_mean, time_std



