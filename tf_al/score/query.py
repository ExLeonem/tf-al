import numpy as np
from .__checks import has_same_len, has_shape_len, of_type


def __single_experiment_qeff(main:np.ndarray, baseline: np.ndarray) -> tuple:
    main_mean = np.mean(main)
    main_std = np.std(main)

    base_mean = np.mean(baseline)
    base_std = np.std(baseline)
    return (main_mean-base_mean)/max(main_mean, base_mean), \
        (main_std-base_std)/max(main_std, base_std)


def __multi_experiment_qeff(main: np.ndarray, baseline: np.ndarray) -> tuple:
    main_exp_mean = np.mean(np.mean(main, axis=0))
    main_exp_std = np.mean(np.std(main, axis=0))

    base_exp_mean = np.mean(np.mean(baseline, axis=0))
    base_exp_std = np.mean(np.std(baseline, axis=0))

    return (main_exp_mean-base_exp_mean)/max(main_exp_mean, base_exp_mean), \
         (main_exp_std-base_exp_std)/max(main_exp_std, base_exp_std)


# def qeff(main, baseline, args*):
def qeff(main, baseline):
    """
        Calculate the query efficiency of main method to
        the baseline.

        Parameters:
            main (np.ndarray): The query time per timestep
            baseline (np.ndarray): The baseline times to check against


        Examples:
        > queff(1.3123, 5.32132)

        > queff(np.array([8.93912, 7.93938, 7.8848]), np.array([3.93912, 3.93938, 3.8848]))

        > mult_exp_1 = np.array([[1.313, 1.238, 1.134], [1.434, 1.313, 1.223]])
        > mult_exp_2 = np.array([[5.313, 4.238, 3.134], [5.434, 4.313, 3.223]])
        > queff(mult_exp_1, mult_exp_2)
    """

    if of_type(np.ndarray, main, baseline):

        if not has_same_len(main, baseline):
            raise ValueError("Error in qeff(). Expected length of input parameters to match.")
        
        # Output of single experiment was passed (query_time_at_al_round_n)
        if has_shape_len(main, 1) and has_shape_len(baseline, 1):
            return __single_experiment_qeff(main, baseline)

        # Output of multiple experiments was passed (n_th_experiment, query_time_at_al_round_n)
        elif has_shape_len(main, 2) and has_shape_len(main, 2):
            return __multi_experiment_qeff(main, baseline)

        raise ValueError("Error in qeff(). Invalid dimensionality. Inputs can't have more than 2 dimensions. Received shapes \"main={}\" and \"baseline={}\".".format(main.shape, baseline.shape))

    # Single value
    elif of_type(float, main, baseline) or of_type(int, main, baseline):
        return (main-baseline)/max(main, baseline)


    raise ValueError("Error in qeff(). Expected both inputs to be of type numpy.ndarray, float or int. Got \"main={}\" and \"baseline={}\"".format(type(main), type(baseline)))
