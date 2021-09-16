

class MetricsAccumulator:
    """
        Collect and adds metrics from active learning to be
        written into an output file.

        Parameters:
            metrics (str|list(str)): A metric name or a list of metrics.
    """


    def __init__(self, metrics=None):
        self.__metrics = []
        self.__update_metrics(metrics)

        
    def __call__(self, **kwargs):
        
        output_metrics = {}
        for metric in self.__metrics:
            output_metrics.update(metric(kwargs))

        return output_metrics


    
    def get_metric(self, name):
        



    
    def add_metric(self, callback):
        """
            Adds another metric to be added to the output file.

            Parameters:
                callback (function): A function return a dictionary of metrics.
        """

        pass