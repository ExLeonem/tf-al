

class FalseNegative:

    def __init__(self, multi_label=False):
        self.name = "false_negative"
        self._multi_label = multi_label

    
    def __call__(self, true_targets, predictions, sample_size=None, **kwargs):
        pass