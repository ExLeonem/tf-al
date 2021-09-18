from ..acquisition_function import AcquisitionFunction, name


@name("least_confidence")
class LeastConfidence(AcquisitionFunction):

    def __call__(self, predictions):
        pass