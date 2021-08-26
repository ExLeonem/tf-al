from .acquisition_function import AcquisitionFunction, name


@name("random")
class Random(AcquisitionFunction):
    
    def __call__(self, predictions):
        pass

