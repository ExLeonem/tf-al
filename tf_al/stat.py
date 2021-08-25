
class Stat:
    """
        Use active learning output to compute statistically interesting stuff.

    """

    def __init__(self):
        pass


    @staticmethod
    def jensen_shannon():
        pass

    
    @staticmethod
    def labeling_efficiency(baseline, alternative_method, lable_size):
        """
            Calculate the labeling efficiency for a baseline value (usually random selection) against
            an alternative query method, for a specific labeled pool size.

            Parameters:
                baseline (float|numpy.ndarray): The accuracy of the baseline. Either single value or multiple values for different label sizes.
                alternative_method (float|numpy.ndarray): The accuracy of the alterantive method. Either single value or multiple values.
                label_size (int|numpy.ndarray): The size of the labeled pool

            Returns:
                (float|numpy.ndarry) single value or multiple labeling efficiencies, depending the given inputs.
        """
        pass

    
    @staticmethod
    def auc():
        pass