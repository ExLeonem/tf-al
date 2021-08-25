import os, sys, importlib
import numpy as np

from . import Model, ModelType, Mode

# import mp.MomentPropagation as mp
# import tensorflow as tf

# -----------------------
# Dependent on external package MomentPropagation !!!!!!!
# -----------------------

# class MomentPropagation(Model):
#     """
#         Takes a regular MC Dropout model as input, that is used for fitting.
#         For evaluation a moment propagation model is created an used 

#     """

#     def __init__(self, model, config=None, **kwargs):
#         model_type = ModelType.MOMENT_PROPAGATION

#         super(MomentPropagation, self).__init__(model, config, model_type=model_type, **kwargs)

#         self.__mp_model = self._create_mp_model(model)
#         self.__compile_params = None


#     def __call__(self, inputs, **kwargs):
#         return self.__mp_model.predict(inputs, **kwargs)

    
#     def fit(self, *args, **kwargs):
#         history = super().fit(*args, **kwargs)
#         self.__mp_model = self._create_mp_model(self._model)
#         return history


#     def compile(self, **kwargs):

#         if kwargs is not None and len(kwargs.keys()) > 0:
#             print("Set compile params")
#             self.__compile_params = kwargs

#         self._model.compile(**self.__compile_params)
#         # self.__base_model.compile(**self.__compile_params)


#     def evaluate(self, inputs, targets, **kwargs):
#         """
#             Evaluates the performance of the model.

#             Parameters:
#                 inputs (numpy.ndarray): The inputs of the neural network.
#                 targets (numpy.ndarray): The true output values.
#                 batch_size (int): The number of batches to use for the prediction. (default=1)
                

#             Returns:
#                 (list()) Loss and accuracy of the model.
#         """

#         self.logger.info("Evaluate kwargs: {}".format(kwargs))

#         self.set_mode(Mode.EVAL)
#         exp, var = self.__mp_model.predict(inputs, **kwargs)
#         loss, acc = self.__evaluate(exp, targets)
#         return {"loss": loss, "accuracy": acc}


#     def __evaluate(self, prediction, targets):
#         """
#             Calculate the accuracy and the loss
#             of the prediction.

#             Parameters:
#                 prediction (numpy.ndarray): The predictions made.
#                 targets (numpy.ndarray): The target values.

#             Returns:
#                 (list()) The accuracy and 
#         """

#         self.logger.info("Prediction shape: {}".format(prediction.shape))

#         loss_fn = tf.keras.losses.get(self._model.loss)
#         loss = loss_fn(targets, prediction)
        
#         prediction = self.extend_binary_predictions(prediction)

#         labels = np.argmax(prediction, axis=1)
#         acc = np.mean(labels == targets)
#         return [np.mean(loss.numpy()), acc]


#     def map_eval_values(self, values, custom_names=None):
#         """
#             Maps the values returned from evaluate(inputs, targets) to specific keys.
#         """
#         metric_names = ["loss", "accuracy"] if custom_names is None else custom_names
#         return dict(zip(metric_names, values))


#     def _create_mp_model(self, model):
#         """
#             Transforms the set base model into an moment propagation model.

#             Returns:
#                 (tf.Model) as a moment propagation model.
#         """
#         _mp = mp.MP()
#         return _mp.create_MP_Model(model=model, use_mp=True, verbose=True)


#     def variance(self, predictions):
#         expectation, variance = predictions

#         variance = self.extend_binary_predictions(variance)
#         return self.__cast_tensor_to_numpy(variance)


#     def expectation(self, predictions):
#         expectation, variance = predictions
        
#         expectation = self.extend_binary_predictions(expectation)
#         return self.__cast_tensor_to_numpy(expectation) 


#     # --------
#     # Weights loading
#     # ------------------

#     # def load_weights(self):
#     #     path = self._checkpoints.PATH
#     #     self.__base_model.load_weights(path)

#     # def save_weights(self):

#     #     path = self._checkpoints.PATH
#     #     self.__base_model.save_weights(path)


#     # --------
#     # Utilities
#     # ---------------

#     def extend_binary_predictions(self, predictions):
#         """
#             Extend dimensions for binary classification problems.

#             Parameters:
#                 predictions (numpy.ndarray): Predictions made by the network.

#             Returns:
#                 (numpy.ndarray) The predictions with extended dimension.
#         """
#         # Don't modify predictions shape in regression case
#         if not self.is_classification():
#             return predictions

#         # Binary case: calculate complementary prediction and concatenate
#         if self.is_binary():
#             bin_alt_class = (1 + np.zeros(predictions.shape)) - predictions

#             # Expand dimensions for predictions to concatenate. Is this needed?
#             # bin_alt_class = np.expand_dims(bin_alt_class, axis=-1)
#             # predictions = np.expand_dims(predictions, axis=-1)

#             # Concatenate predictions
#             class_axis = len(predictions.shape) + 1
#             predictions = np.concatenate([predictions, bin_alt_class], axis=len(predictions.shape)-1)
        
#         return predictions

    
#     def __cast_tensor_to_numpy(self, values):
#         """
#             Cast tensor objects of different libraries to
#             numpy arrays.
#         """

#         # Values already of type numpy.ndarray
#         if isinstance(values, np.ndarray):
#             return values

#         values = tf.make_ndarray(values)
#         return values


#     # ----------------
#     # Custom acquisition functions
#     # ---------------------------

#     def get_query_fn(self, name):

#         if name == "max_entropy":
#             return self.__max_entropy
        
#         if name == "bald":
#             return self.__bald
        
#         if name == "max_var_ratio":
#             return self.__max_var_ratio

#         if name == "std_mean":
#             return self.__std_mean


#     def __max_entropy(self, data, **kwargs):
#         # Expectation and variance of form (batch_size, num_classes)
#         # Expectation equals the prediction
#         predictions = self.__mp_model.predict(x=data)

#         # Need to scaled values because zeros
#         class_probs = self.expectation(predictions)
        
#         class_prob_logs = np.log(np.abs(class_probs) + .001)
#         return -np.sum(class_probs*class_prob_logs, axis=1)

    
#     def __bald(self, data, **kwargs):
#         """
#             [ ] Check if information about variance is needed here. Compare to mc dropout bald.
#         """
#         predictions = self.__mp_model.predict(x=data)
#         expectation = self.expectation(predictions)
#         variance = self.variance(predictions)

#         first_term = -np.sum(expectation * np.log(np.abs(expectation) + 1e-100), axis=1)
#         predictions = self.extend_binary_predictions(predictions)
#         second_term = np.sum(np.mean(predictions*np.log(np.abs(predictions) + 1e-100), axis=1), axis=1)
#         return first_term + second_term


#     def __max_var_ratio(self, data, **kwargs):
#         predictions = self.__mp_model.predict(x=data)
#         expectation = self.expectation(predictions)

#         col_max_indices = np.argmax(expectation, axis=1)        
#         row_indices = np.arange(len(data))
#         max_var_ratio = 1- expectation[row_indices, col_max_indices]
#         return max_var_ratio

    
#     def __std_mean(self, data, **kwargs):
#         predictions = self.__mp_model.predict(data, **kwargs)
#         variance = self.variance(predictions)
#         std = np.square(variance)
#         return np.mean(std, axis=-1)
