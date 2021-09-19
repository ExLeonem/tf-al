from enum import Enum
import numpy as np


class LabelType(Enum):
    CLASS_LABEL=1,
    ONE_HOT_VECTOR=2,
    VALUE=3

class OracleMode(Enum):
    PSEUDO=1,
    ANNOTATE=2


class Oracle:
    """
        Oracle handles the labeling process for input values.
        
        Parameters:
            callback (Callback): Function to call for user input for input values. Function receives (pool, indices)
            pseudo_mode (bool): Active learning environment in pseudo mode?
    """

    def __init__(self, callback=None, pseudo_mode=False):
        self.__annotation_callback = callback
        self.pseudo_mode = pseudo_mode


    def init(self, pool, size, pseudo_mode=None):
        """
            Initialize pool with given number of samples.

            Parameters:
                pool (Pool): holding information about already labeled targets.
                size (int): number of elements to initialize the pool with.
                pseudo_mode (bool): Whether or not pseudo labeling of inputs. (Only applicable when pool initialized with targets)
        """
        if self.is_pseudo(pseudo_mode) and pool.is_pseudo():
            pool.init(size)
            return
        

        if self.__annotation_callback is None:
            raise ValueError("Error in Oracle.init(). Can't initialize pool because no callback function was set.")

        unlabeled_indices = pool.get_unlabeled_indices()
        indices = np.random.choice(unlabeled_indices, size, replace=False)
        self.__annotation_callback(pool, indices)


    def annotate(self, pool, indices, pseudo_mode=None):
        """
            Create annotations for given indices and update the pool.

            Parameters:
                pool (Pool): The pool holding information about already annotated inputs.
                indices (numpy.ndarray|list(int)): Indices indicating which inputs to annotate.
        """
        
        # Pseudo mode, use already known labels
        if pool.is_pseudo() and self.is_pseudo(pseudo_mode): 
            pool.annotate(indices)
            return 

        if self.__annotation_callback is None:
            raise ValueError("Error in Oracle.annotate(). Oracle not in pseudo-mode and callback is None.")

        self.__annotation_callback(pool, indices)

    
    def is_pseudo(self, mode=None):
        """
            Is the oracle put into pseudo labeling mode?
            Meaning: when pool is also in pseudo mode, labels will automatically be set by using known labels.
        """

        if mode is not None:
            if not isinstance(mode, bool):
                raise ValueError("Error in Oracle. Value of pseudo_mode is not boolean.")

            return mode
            
        else:
            return self.pseudo_mode
        



        