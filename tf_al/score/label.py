import numpy as np
from .__checks import has_same_len, has_shape_len, of_type


def __multi_experiment_leff(main: np.ndarray, baseline: np.ndarray) -> tuple:
    """
        Calcualte labeling efficiency for multi experiment inputs.

        Parameters:
            main (numpy.ndarray): Accuracy of different experiments run.
            baseline (numpy.ndarray): Accuracy of different experimental runs of the baseline function.

        Returns:
            (tuple(float, float)) the mean of the labeling efficiency and standard deviation.
    """
    
    # Do inputs have the same number of accuracies at round x
    if main.shape[-1] != baseline.shape[-1]:
        raise ValueError("Error in leff(). Input shape missmatch. Got shapes \"main={}\" and \"baseline={}\"".format(main.shape, baseline.shape))

    main_eff = main/baseline
    return np.mean(main_eff, axis=0), np.std(main_eff, axis=0)

    # main_mean = np.mean(main, axis=0)
    # main_std = np.std(main, axis=0)
    
    # base_mean = np.mean(baseline, axis=0)
    # base_std = np.std(baseline, axis=0)

    # return (main_mean/base_mean), (main_std/base_std)


def leff(main, baseline):
    """
        Calculate the labeling efficiency over n-active learning rounds.

        Parameters:
            main (numpy.ndarray|float|int): The accuracies of main acquisition function for which to compute relative values.
            baseline (numpy.ndarray|float|int): The accuraciesk of the baseline acquisition function.

        Returns:
            (numpy.ndarray|float|int|tuple) the computed labeling efficiency per active learning round. Alternativly mean and std of computed labeling efficiency on pasesd 2D numpy array.
    """


    if of_type(np.ndarray, main, baseline):

        if not has_same_len(main, baseline):
            raise ValueError("Error in qeff(). Expected length of input parameters to match. Received \"main={}\" and \"baseline={}\".".format(main.shape, baseline.shape))

        # Output of single experiment passed (accuracy_at_round_n)
        if has_shape_len(main, 1) and has_shape_len(baseline, 1):
            mean_leff = main/baseline
            return mean_leff, np.zeros(len(main))

        # Output of multiple experiments passed (nth_experiment, accuracy_at_round_n)
        elif has_shape_len(main, 2) and has_shape_len(baseline, 2):
            return __multi_experiment_leff(main, baseline)


        raise ValueError("Error in leff(). Expected main and baseline to have the same shape. Received shapes \"main={}\" and \"base={}\".".format(main.shape, baseline.shape))

    
    # Calculate labeling efficiency for single value pair (values of different functions at active learning step n)
    elif of_type(float, main, baseline) or of_type(int, main, baseline):
        return main/baseline
    

    raise ValueError("Error in leff(). Type missmatch. Expected parameters both to be of type np.ndarray, float or int.\
         Received types {} and {}.".format(type(main), type(baseline)))



