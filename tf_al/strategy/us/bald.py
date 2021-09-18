from ..acquisition_function import AcquisitionFunction, name


@name("bald")
class Bald(AcquisitionFunction):

    def __call__(self, predictions):
        pass