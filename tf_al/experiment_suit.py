import os, sys, select
import numpy as np
import tensorflow as tf

from . import ActiveLearningLoop, AcquisitionFunction, ExperimentSuitMetrics
# from .wrapper import Model
from .utils import setup_logger

class ExperimentSuit:
    """
    Performs a number of experiments.
    Iterating over given models and methods.

    Parameters:
        models (list(Model)): The models to iterate over.
        query_fns (list(str)|list(AcquisitionFunction)|str|AcquisitionFunction): A list of query functions to use
        dataset (Dataset): A dataset for experiment execution.
        max_rounds (int): The max. number of rounds to query for datapoints per experiment run. If not set, perform query operation as long as there is data.
        acceptance_timeout (int): Timeout in seconds in which experiment can be proceeded or aborted, after successfull (model,query function) iteration. Setting None will automatically proceed. (default: None)
        metrics_handler (ExperimentSuitMetrics): A configured metrics handler to use.
        verbose (bool): Printing log messages?
    """

    def __init__(
        self, 
        models,
        query_fns,
        dataset,
        step_size=1,
        max_rounds=None,
        runs=1,
        seed=None,
        no_save_state=False,
        acceptance_timeout=None,
        metrics_handler=None,
        verbose=False
    ): 
        
        # if seed is not None and isinstance(seed, int):
        #     np.random.seed(seed)
        #     tf.random.set_seed(seed)

        self.verbose = verbose
        self.logger = setup_logger(verbose, name="ExperimentSuit Logger")

        self.dataset = dataset
        self.max_rounds = max_rounds
        self.runs = runs
        self.step_size = step_size
        self.acceptance_timeout = acceptance_timeout
        self.seed = seed

        self.models = self.__init_models(models)
        self.query_functions = self.__init_query_fns(query_fns)
        self.metrics_handler = metrics_handler
        self.no_save_state = no_save_state
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # To disable tensorflow output


    def start(self):
        """
            Starts the experiment suit. 
            Runs an experiment for each acquisition function and model combination.

            TODO:
                [x] Last iteration even when no other experiments to run, prompts proceeding request.
                [ ] Implement run/seed implementation. Run seeds experiments with seeds n-times.
        """
        
        if self.runs == 1:
            self.__iterate_seeds(self.runs-1)
            return


        for run in range(self.runs):
            self.__iterate_experiments(run)


    def __iterate_seeds(self, run):
        """
            Iterate through different seeds when a list of seeds where given.
        """

        if self.seed is not None:

            if isinstance(self.seed, int):
                self.__iterate_experiments(run, self.seed)
                return

            if not isinstance(self.seed, list):
                raise ValueError("Error initializing the active learning loop. Parameter seed of unknown type. Expected list of integers or single integer.")
            
            for idx in range(len(self.seed)):
                seed = self.seed[idx]
                self.__iterate_experiments(idx, seed)

            return
        
        # Run single experiment without seed
        self.__iterate_experiments(run)
    

    def __iterate_experiments(self, run, seed=None):
        """
            
            Parameters:
                run (int): The number of run for acquisition function model and model.

        """

        # Perform experiment for each model & query function combination
        exit_loop = False
        number_of_models = range(len(self.models))
        number_of_query_fns = range(len(self.query_functions))
        for i in number_of_models:
            model = self.models[i]
            
            if not self.no_save_state and not model.has_save_state():
                model.save_weights()

            # Group experiment output per model in terminal
            if i != 0:
                print("#"*10)

            metrics = None
            for j in number_of_query_fns:
                query_fn = self.query_functions[j]

                print("Running experiment (Run: {} | Model: {} | Query-Function: {})".format(run, model, query_fn))
                self.__run_experiment(run, model, query_fn, seed)

                if (j != (len(self.query_functions)-1) or i != (len(self.models)-1)) \
                and not self.__await_proceed():

                    exit_loop = True
                    break
            
            if exit_loop:
                break

    def __run_experiment(self, run, model, query_fn, seed):
        """
            Run a single experiment.

            Parameters:
                run (int): The number of experiment of this type (combination of acquisition funciton and model)
                model (Model): A model wrapper.
                query_ fn (str|AcquisitionFunction): The acquisition function to use.
        """

        # Quick fix, reset random state after each iteration
        # TODO: Adding outter loop and extend parameter list for seed
        if seed is not None and isinstance(seed, int):
            np.random.seed(seed)
            tf.random.set_seed(seed)

        active_learning_loop = ActiveLearningLoop(
            model, 
            self.dataset, 
            query_fn, 
            step_size=self.step_size,
            max_rounds=self.max_rounds,
            pseudo=True,
            verbose=self.verbose
        )

        experiment_name = str(run) + "_" + active_learning_loop.get_experiment_name()
        active_learning_loop.run(experiment_name, self.metrics_handler)


    def __await_proceed(self):
        """
            Waiting for user input to proceed or abort experiments.

            TOOD:
                [ ] Restart user input when failed input
        """

        if self.acceptance_timeout is not None and isinstance(self.acceptance_timeout, int):
            print("Proceed with next experiment? (y/n) ")
            while True:
                i, o, e = select.select([sys.stdin], [], [], 2)

                if i: 
                    value = sys.stdin.readline().strip().lower()
                    if value == "y" or value == "yes":
                        return True
                    elif value == "n" or value == "no":
                        return False
                    else:
                        print("Unknown value passed. Either input yes or no.")
                        continue

                else:
                    print("\033[F Time-out. Auto-proceed with experiments.")
                    return True
            
        return True



    # ------------
    # Utilities
    # ---------------

    def __init_models(self, models):
        """
            Iterate through passed models,
            raising an error when one of the models can't be processed.
        """

        if isinstance(models, BayesModel):
            return [models]

        verified_models = []
        if isinstance(models, list):
            for model in models:
                
                # Passed model can be used in context of ActiveLearningLoop?
                if not isinstance(model, BayesModel):
                    raise ValueError("Error in ExperimentSuit.__init__(). One of the passed models is no sub-class of BayesModel.")

        else:
            raise ValueError("Error in ExperimentSuit.__init__(). Can't parse models of type {}. Expected list of or single BayesModel.".format(type(models)))

        return models


    def __init_query_fns(self, query_fns):
        """
            Create AcquisitionFunction
        """

        if isinstance(query_fns, str) or isinstance(query_fns, AcquisitionFunction):
            return [query_fns]

        fns = []
        if isinstance(query_fns, list):
            for query_fn in query_fns:

                if isinstance(query_fn, str):
                    fns.append(AcquisitionFunction(query_fn))

                elif isinstance(query_fn, AcquisitionFunction):
                    fns.append(query_fn)
                
                else:
                    raise ValueError("Error in ExperimentSuit.__init__(). Can't initialize one of the given AcquisitionFunctions")
        
        else:
            raise ValueError("Error in ExperimentSuit.__init__(). Got type {} for qury_fns. Expected a list of strings, AcqusitionFunctions, singel strings or a single AcquisitionFunction.".format(type(query_fns)))

        return fns