from .acquisition_function import AcquisitionFunction, name


@name("max_entropy")
class MaxEntropy(AcquisitionFunction):

    def __call__(self, predictions):
        pass