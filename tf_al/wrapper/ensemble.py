import numpy as np
import tensorflow as tf
from . import Model


class Ensemble(Model):

    def __init__(self, model, num_ensembles=1, config=None, **kwargs):
        super().__init__(model, config, model_type="ensemble", **kwargs)
        self.__num_ensembles = num_ensembles
        self.__models = self.__init_models(model, num_ensembles)


    
    def __call__(self, inputs, **kwargs):
        pass

    
    def evaluate(self, inputs, targets, **kwargs):
        pass

    
    def fit(self, inputs, targets, **kwargs):
        pass


    def compile(self, *args, **kwargs):

        if self._compile_params is None:
            self._compile_params = kwargs

        for model in self.__models:
            model.compile(**self.kwargs)
        
        metrics = self._create_init_metrics(kwargs)
        if self.eval_metrics == []:
            self.eval_metrics = metrics


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


    # -----------
    # Utilities
    # -----------------


    def __init_models(self, base_model, num_ensembles):
        models = []
        for i in range(num_ensembles):
            models.append(tf.keras.models.clone_model(base_model))
        
