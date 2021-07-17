import tensorflow as tf


def setup_growth():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    # cpus = tf.config.experimental.list_physical_devices("CPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPU's, ", len(logical_gpus), "Logical GPU's")
            
        except RuntimeError as e:
            print(e)
            
    elif cpus:
        try:
            logical_cpus = tf.config.experimental.list_logical_devices("CPU")
            print(len(cpus), "Physical CPU,", len(logical_cpus), "Logical CPU")
            
        except RuntimeError as e:
            print(e)
            
    tfk = tf.keras