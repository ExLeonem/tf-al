

class Experiment:

    """
        A single runable experiment.
    """


    def __init__(self, model, query_fn, dataset=None, **kwargs):
        self.__model = model
        self.__query_fn = query_fn
        self.__dataset = dataset


    def run(self, dataset=None):
        """
            Runs the experiment.

            Parameters:
                dataset (Dataset): The dataset on which to perform active learning.

            
            Returns:
                (Metrics)
        """

        if not self.has_dataset() and dataset is None:
            raise ValueError("Can't perform experiment without a dataset.")

        pass



    # ----------
    # Setter/-Getter
    # -----------------

    def get_dataset(self):
        return self.__dataset

    def get_query_fn(self):
        return self.__query_fn

    def get_model(self):
        return self.__model


    # ---------
    # Uitilities
    # ---------------

    def has_dataset(self):
        return self.__dataset is not None

    


