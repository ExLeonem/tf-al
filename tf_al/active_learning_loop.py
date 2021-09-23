import time, gc
from tqdm import tqdm

from .utils import setup_logger
from .wrapper import Model
from . import AcquisitionFunction, Pool, Oracle, \
    ExperimentSuitMetrics, Dataset


class ActiveLearningLoop:
    """
        Creates an active learning loop. The loop accumulates metrics during training in a dictionary
        that is returned.

        To use with tqdm: 
        ::
            for i in tqdm(my_iterable):
            do_something()

        Parameters:
            model (Model): A model wrapped into a Model type object.
            dataset (Dataset): The dataset to use (inputs, targets)
            query_fn (list(str)|str): The query function to use.
            step_size (int): How many new datapoints to add per active learning rounds. (default=1)
            max_rounds (int): The max. number of rounds to execute the active learning loop. If None apply until unlabeled data pool is empty. (default=None)
            pseudo (bool): Whether or not to execute loop in pseudo mode. Pseudo mode uses already existing labels to perform experiments. (default=True)
            verbose (bool): Wheter or not to generate logging output. (default=False)
    """

    def __init__(
        self,
        model: Model,
        dataset: Dataset,
        query_fn,
        step_size: int=1,
        max_rounds: int=None,
        pseudo: bool=True,
        verbose: bool=False,
        **kwargs
    ):
        
        self.verbose = verbose
        self.logger = self.logger = setup_logger(verbose, "ActiveLearningLoop-Logger")

        # Data and pools
        self.dataset = dataset
        x_train, y_train = dataset.get_train_split()
        self.initial_size = initial_pool_size = dataset.get_init_size()

        self.pool = Pool(x_train, y_train)
        if dataset.is_pseudo() and initial_pool_size > 0:
            self.pool.init(initial_pool_size)
        
        # Loop parameters
        self.step_size = step_size
        self.iteration_user_limit = max_rounds
        self.iteration_max = int(self.pool.get_length_unlabeled())
        self.i = 0

        # Active learning components
        self.model = model
        self.oracle = Oracle(pseudo_mode=pseudo)
        self.query_fn = self.__init_acquisition_fn(query_fn)

        self.query_config = self.model.get_config_for("query")
        self.query_config.update({"step_size": step_size})


    def __len__(self):
        """
            How many iterations until the active learning loop exits.

            Returns:
                (int) The number of iterations.
        """

        times, rest = divmod(self.iteration_max, self.step_size)
        if rest != 0:
            times += 1

        if self.iteration_user_limit is not None and self.iteration_user_limit < times:
            return self.iteration_user_limit
        
        return times



    # ---------
    # Iterator Protocol
    # -----------------

    def __iter__(self):
        self.i = 0
        return self


    def __next__(self):
        """
            Iterate over dataset and query for labels.

            Returns:
                (dict) Accumulated information during avtive learning iterations.
        """

        if not self.has_next():
            raise StopIteration

        # Load previous checkpoints/recreate model
        self.logger.info("++++++++++++++ (START) Iteration {} ++++++++++++++".format(self.i))
        self.model.reset(self.pool, self.dataset)
        
        # Optimiize model params
        optim_metrics, optim_time = self.__optim_model_params()

        # Fit model
        train_metrics, train_time = self.__fit_model()
        self.logger.info("\\---- (END) Model fitting")

        # Update pools
        acq_start = time.time()
        self.logger.info("\\---- (START) Acquisition")
        indices, _pred = self.query_fn(self.model, self.pool, **self.query_config)
        self.logger.info("\\---- (END) Acquisition")
        acq_time = time.time() - acq_start

        self.oracle.annotate(self.pool, indices)

        # Evaluate model
        eval_metrics, eval_time = self.__eval_model()
        self.i += 1

        # Fix some of tf memory leak issues
        gc.collect()
        self.model.clear_session()
        self.logger.info("++++++++++++++ (END) Iteration ++++++++++++++")

        return {
            "train": train_metrics,
            "train_time": train_time,
            "query_time": acq_time,
            "optim": optim_metrics,
            "optim_time": optim_time,
            "eval": eval_metrics,
            "eval_time": eval_time,
            "indices_selected": indices.tolist()
        }


    def __optim_model_params(self):
        """
            Perform parameter optimization using on a validation set.
        """
        metrics = None
        duration = None
        if hasattr(self.model, "optimize") and self.dataset.has_eval_set():
            e_inputs, e_targets = self.dataset.get_eval_split()
            start = time.time()
            metrics = self.model.optimize(e_inputs, e_target)
            duration = time.time() - start

        return metrics, duration

    
    def __fit_model(self):
        """
            Fit model to the labeled data.

            Returns:
                (tuple(dict(), float)) metrics and the time needed to fit the model.
        """
        history = None
        duration = None

        self.logger.info("//// (START) Model fitting")
        if self.pool.get_length_labeled() > 0:
            inputs, targets = self.pool.get_labeled_data()
            start = time.time()

            h = None
            if self.dataset.has_eval_set():
                x_eval, y_eval = self.dataset.get_eval_split()
                h = self.model.fit(inputs, targets, validation_data=(x_eval, y_eval), verbose=False)
            else:
                h = self.model.fit(inputs, targets, verbose=False)

            duration = time.time() - start
            history = h.history

        return history, duration


    def __eval_model(self):
        """
            Performan an evaluation of the model.

            Returns:
                (tuple(dict(), float)) metrics and the time needed to evaluate the model.
        """
        metrics = None
        duration = None
        config = self.model.get_config_for("eval")

        # print("Config: {}".format(self.model.get_config()))
        if self.dataset.has_test_set():
            x_test, y_test = self.dataset.get_test_split()
            start = time.time()
            
            # predictions = self.model(x_test, y_test, **config)
            # new_metrics = self.__metrics_acc(predictions, x_test, y_test, **config)
            metrics = self.model.evaluate(x_test, y_test, **config)
            duration = time.time() - start
        
        return metrics, duration


    # -------
    # Functions for diverse grades of control
    # ---------------------------------

    def run(self, experiment_name=None, metrics_handler=None):
        """
            Runs the active learning loop till the end.

            Parameters:
                experiment_name (str): The name of the file to write to
                metrics_handler (ExperimentSuitMetrics): Metrics handler for write/read operations.
        """

        if experiment_name is None:
            experiment_name = self.get_experiment_name()

        # Write meta information
        if metrics_handler is not None \
        and isinstance(metrics_handler, ExperimentSuitMetrics):
            model_name = self.get_model_name()
            query_fn = self.get_query_fn_name()
            params = self.collect_meta_params()
            metrics_handler.add_experiment_meta(experiment_name, model_name, query_fn, params)

        iteration = 0
        with tqdm(total=self.__len__()) as pbar:
            for metrics in self:
                iteration += 1
                pbar.update(1)

                metrics.update({
                    "iteration": iteration,
                    "labeled_pool_size": self.pool.get_length_labeled()-self.step_size,
                    "unlabeled_pool_size": self.pool.get_length_unlabeled()
                })

                # Write metrics to file
                if metrics_handler is not None \
                and isinstance(metrics_handler, ExperimentSuitMetrics):
                    metrics_handler.write_line(experiment_name, metrics)


    def step(self):
        """
            Perform a step of the active learning loop.
        """
        return next(self)

    
    def has_next(self):
        """
            Can another step of the active learning loop be performed?
        """

        # Limit reached?
        if (self.iteration_user_limit is not None) and not (self.i < self.iteration_user_limit):
            return False

        if self.i >= self.iteration_max:
            return False

        # Any unlabeled data left?
        if not self.pool.has_unlabeled():
            return False

        return True


    # ----------
    # Utils
    # ----------------

    def is_done(self):
        """
            The active learning has executed and is done.

            Returns:
                (bool) whether or not the loop has executed.
        """
        return not self.has_next()


    def collect_meta_params(self):
        """
            Collect meta information about experiment to be written into .meta.json.

            Returns:
                (dict) with all meta information.
        """

        iterations = self.iteration_max
        if self.iteration_user_limit is not None and self.iteration_user_limit < self.iteration_max:
            iterations = self.iteration_user_limit

        # fitting_params = model.get_compile_params()
        initial_indices = []
        if self.pool.has_labeled():
            initial_indices = self.pool.get_labeled_indices().tolist()


        return {
            "iterations": iterations,
            "step_size": self.step_size,
            "initial_size": self.initial_size,
            "initial_indices": initial_indices
        }

    
    # -----------
    # Initializers
    # --------------------

    def __init_acquisition_fn(self, functions):

        # Single acquisition function?
        if isinstance(functions, str):
            return AcquisitionFunction(functions)

        # Already acquisition function
        if isinstance(functions, AcquisitionFunction):
            return functions

        else:
            raise ValueError(
                "Error in ActiveLearningLoop.__init_acquisition_fn(). Can't initialize one of given acquisition functions. \
                Expected value of type str or AcquisitionFunction. Received {}".format(type(functions))
            )
    

    # --------
    # Getters-/Setter
    # ------------------
    
    def get_experiment_name(self):
        return self.get_model_name() + "_" + self.get_query_fn_name()

    def get_model_name(self):
        return self.model.get_model_name().lower()
    
    def get_query_fn_name(self):
        return self.query_fn.get_name()