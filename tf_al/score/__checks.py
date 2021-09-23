import numpy as np

"""
    A collection of helper functions to do check ups for score calculation.
"""

def has_shape_len(input_arr: np.ndarray, shape_length: int) -> bool:
    return len(input_arr.shape) == shape_length


def has_same_len(first_arr: np.ndarray, second_arr: np.ndarray) -> bool:
    return len(first_arr) == len(second_arr)


def of_type(_type, value_1, *args) -> bool:
    """
        Check if a collection of values are of the same type.

        Parameters:
            _type (any): The type to check for.
            value_1 (any): The first value to check.
            *args (any): Rest of values to check against given type.

        Returns:
            (bool) whether or not all inputs of given type. 
    """

    all_of_type = isinstance(value_1, _type)
    i = len(args)

    while i > 0 and all_of_type != False:
        all_of_type = isinstance(args[i-1], _type)
        i -= 1

    return all_of_type
    