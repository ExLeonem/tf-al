import tensorflow as tf
import numpy as np
from tqdm import tqdm


def disable_batch_norm(model):
    """
        Disable batch normalization for activation of dropout during prediction.

        Parameters:
            - model (tf.Model) Tensorflow neural network model.
    """
    
    disabled = False
    for l in model.layers:
        if l.__class__.__name__ == "BatchNormalization":
            disabled = True
            l.trainable = False

    if disabled:
        print("Disabled BatchNorm-Layers.")


def predict_n(model, point, n_times=5, enable_tqdm=True):
    """
        Helper to perform n-predictions for given bayesian model.
        Returns predictions in numpy array.

        Parameters:
            - model (tf.Model | torch.Model) The model to be used for the prediction
            - point (np.ndarray) The datapoint to be used for prediction
            - n_times (int) The number of predictions to be made
            - enabel_tqdm (bool) Should tqdm be used for progress indication?

        TODO: 
            - [ ] extend to be useable with pytorch aswell
            - [ ] Test if first dimension of output can always be omitted
        
        Returns:
            - (np.ndarray)  of shape (ntimes, ...model output shape)
    """

    if n_times < 1 or not int(n_times):
        raise ArgumentError("At least single prediction needed. But for n_times {} given.".format(n_times))

    # Prepare layers of bayesian model for prediction
    disable_batch_norm(model)

    # Make n-predictions
    output = None
    iterator = tqdm(range(n_times)) if enable_tqdm else range(n_times)
    for i in iterator:

        # Set initial shape
        result = model(point, training=True)
        if output is None:
            output = np.zeros(tuple([n_times]+list(result.shape[1:])))

        output[i] = result

    tf.keras.backend.clear_session()
    return output


def batch_predict_n(model, data, n_times=5, enable_tqdm=True):
    """
        Perform n-predictions per data point in colletion of data points.

        Parameters:
            - model (tf.Model | torch.Model) The model to perform predictions on
            - data (np.ndarray) The data to perform the predictions on
            - n_times (int) total number of predictions per input point
            - fun_pred (function) 

        Returns:
            - (np.ndarray) of shape (num data points, n_times, ...model output shape)
    """

    disable_batch_norm(model)
    data_points = data.shape[0]
    output = None
    # batch_size = data_points if batch_size is None else batch_size
    iterator = tqdm(range(n_times)) if enable_tqdm else range(n_times)
    for i in iterator:

        # Set initial shape
        result = model(data, training=True)
        if output is None:
            output = np.zeros(tuple([n_times] + list(result.shape)))

        output[i] = result

    tf.keras.backend.clear_session()
    return output.reshape(tuple([data_points, n_times] + list(output.shape[2:])))



def measures(predictions, batch_dim=False):
    """
        Helper to calculate regular measures for bayesian neural networks.

        Parameters:
            - predictions (np.ndarray) Array with n-predictions.
            - batch_dim (bool) Existing batch dimension at first index.

        Returns:
            - (np.ndarray, np.ndarray) The mean and std of predictions
    """

    axis = 0
    if batch_dim:
        axis = 1

    avg = np.mean(predictions, axis=axis)
    var = np.var(predictions, axis=axis)

    return (avg, var)


def uncertainty():
    """
        Helper to calculate specific uncertainty measures.
    """
    pass

