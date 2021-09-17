import numpy as np
from . import Model


class SWAG(Model):

    # https://github.com/DD2412-Final-Projects/swag-reproduced/blob/master/train_swag.py

    def __init__(self, model):
        super().__init__(model, config, model_type="swag", **kwargs)

        self.second_moment = None
        self.cov_mat = None

    
    def __call__(self, inputs, **kwargs):
        pass

    
    def evaluate(self, inputs, targets, **kwargs):
        pass

    
    def fit(self, inputs, targets, **kwargs):
        pass


    def get_query_fn(self, name):
    
        # if name == "max_entropy":
        #     return self.__max_entropy
        
        # if name == "bald":
        #     return self.__bald
        
        # if name == "max_var_ratio":
        #     return self.__max_var_ratio

        # if name == "std_mean":
        #     return self.__std_mean

        return None