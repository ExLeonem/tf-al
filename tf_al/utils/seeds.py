import numpy as np


def gen_seeds(num, min=10_000, max=99_999):
    """
        Generates a specific amount of random seeds to use for
        experiments.

        Parameters:
            num (int): The amount of seeds to generate.
            min (int): Min seed 
    """
    return np.random.choice(range(min, max), num, replace=False)
    