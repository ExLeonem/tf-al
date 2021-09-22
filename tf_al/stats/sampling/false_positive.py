

class FalsePositive:

    def __init__(self, multi_label=False):
        self.name = "false_positive"
        self._multi_label = multi_label

    
    def __call__(self, true_targets, predictions, **kwargs):
        pass