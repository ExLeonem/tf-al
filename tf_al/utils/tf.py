import os
import tensorflow as tf

def setup_growth():
    """
        Setup memory to grow. Check tf.config.experimental.set_memory_growth for reference.
    """

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPU's, ", len(logical_gpus), "Logical GPU's")
            
        except RuntimeError as e:
            print(e)
            
    else:
        cpus = tf.config.experimental.list_physical_devices("CPU")
        try:
            logical_cpus = tf.config.experimental.list_logical_devices("CPU")
            print(len(cpus), "Physical CPU,", len(logical_cpus), "Logical CPU")
            
        except RuntimeError as e:
            print(e)


def set_tf_log_level(level="2"):
    """
        Set a log level for tensorflow logging messages.

        Parameters:
            level (str): The log level, one of [0, 1, 2, 3].
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = level


def disable_tf_logs():
    """
        Disable tensorflow log messages.
    """
    set_tf_log_level()