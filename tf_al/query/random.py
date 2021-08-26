from .acquisition_function import AcquisitionFunction, name


@name("random")
class Random(AcquisitionFunction):
    
    def select(self, predictions, **kwargs):
        """
        
        """
        pass
    

