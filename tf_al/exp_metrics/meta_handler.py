import os, json
import base64, pickle
import logging
from tf_al.acquisition_function import AcquisitionFunction
# from ..utils import setup_logger


class MetaHandler():
    """
        Write meta information about the active learning experiment out into
        a json file.

        Parameters: 
            base_path (str): The base path where to put the .meta.json file.
    """

    def __init__(self, base_path, metric_extesion=None, verbose=False):
        # self.__logger = setup_logger(verbose, name="MetaWriter", default_log_level=logging.WARN)
        self.__BASE_PATH = base_path        
        self.__META_FILE_PATH = os.path.join(base_path, ".meta.json")
        self.__METRIC_EXT = metric_extesion

        # What goes initialy into the .meta.json file
        self.__base_content = {
            "models": [], 
            "dataset": {}, 
            "params": {}, 
            "acquisition_function": [], 
            "run": []
        }


    def add_model(self, model):
        """
            Append model information to the meta file. 
            The passed model needs to be compiled. An uncompiled model will result in an error.

            Parameters:
                model (Model): the wrapped model, encapsulating all model information.
        """

        serialized_model = pickle.dumps(model)
        model_name = model.get_model_name()
        
        # https://stackoverflow.com/questions/60212925/is-there-a-keras-function-to-obtain-the-compile-options
        base_model = model.get_base_model()

        # Is model compiled?
        error_message = "Error in MetaHandler.add_model(). Can't add model information because model is not compiled. "
        if not hasattr(base_model, "loss"):
            raise ValueError(error_message + "Missing loss attribute.")

        if not hasattr(base_model, "optimizer"):
            raise ValueError(error_message, "Missing optimizer attribute")


        # Collect compilation parameters
        loss = base_model.loss
        optimizer_name = base_model.optimizer.__class__.__name__
        optimizer_config = base_model.optimizer.get_config()
        optimizer = {
            "name": optimizer_name,
            **optimizer_config
        }
        
        # Write data to the file
        data = {
            "id": model.get_id(),
            "name": model_name,
            "object": base64.b64encode(serialized_model).decode("ascii"),
            "loss": loss,
            "optimizer": optimizer,
        }

        data = self.__add_model_config(model, data)
        self.write("models", data, append=True)


    def add_dataset(self, dataset, name, url=None):
        """
            Add information about the used dataset.

            Parameters:
                dataset (Dataset): The dataset used.
                name (str): The name of the dataset.
                url (str): An optional url where to get the dataset from. (default=None)
        """

        data = {"name": name}

        if url is not None:
            data["url"] = url

        # Add dataset split information
        train_size, test_size, eval_size = dataset.get_split_ratio()
        if test_size != 0 or eval_size != 0:
            data["splits"] = {
                "train": train_size,
                "test": test_size,
                "eval": eval_size
            }                        

        self.write("dataset", data)


    def add_params(self, **kwargs):
        """
            Add active learning specific arguments to the meat file.
        """
        self.write("params", kwargs)

    
    def add_acquisition_function(self, acqusition_function):
        """
            Add information about a specific acquisition function to the meta file.

            Parameters:
                acquisition_function (AcquisitionFunction): The acquisition function for which to add meta information.
        """

        data = {
            "id": 0,
            "name": acqusition_function.name,
            "params": {
                **acqusition_function.kwargs
            }
        }
        
        self.write("acquisition_function", data, append=True)



    def add_run(self, num_run, model_id, acquisition_function, init_indices, seed=None):
        """
            Add information about an experimental run to the meta file.

            TODO: 
                - Add check if experiment already run

            Parameters:
                num_run (int): The number of this run.
                model_id (uuid): The model uuid.
                acquisition_function (AcquisitionFunction): The acquisition function to use.
                init_indices (numpy.ndarray): The initial indices used for this experiment.
                seed (int): The seed on which to experiment is performed. (default=None)

        """

        filename = model_id + "." + self.__METRIC_EXT
        file_path = os.path.join(self.__BASE_PATH, filename)

        data = {
            "n": n,
            "model": model_id,
            "acquisition_function": "test",
            "path": file_path,
            "initial_indices": init_indices
        }

        if seed is not None:
            data["seed"] = seed

        self.write("run", data, append=True)


    def write(self, key, data, append=False):
        """
            Write data into the meta file under given key.

            Parameters:
                path (str|list(str)): Single key or series of keys where to put data.
                data (any()): The data to put under this key.
                replace (bool): Wether or not replace existing data,
                create (bool): Create non-existent paths or keys.
        """
        content = self.read()
        if append:

            if isinstance(content[key], list):
                content[key].append(data)

            elif isinstance(content[key], dict):
                content[key] = {
                    **content[key],
                    **data
                }
            
            else:
                raise ValueError("Error in MetaHandler.write(). Can only append to data of type list or dict. Got type {}".format(type(content[key])))

        else:
            content[key] = data

        # Overwrite old json meta information
        with open(self.__META_FILE_PATH, "w") as json_file:
            json_file.write(json.dumps(content))


    
    def read(self):
        """
            Read informatio from the meta file.
        """
        content = self.__base_content
        with open(self.__META_FILE_PATH, "r") as json_file:
            content = json_file.read()

        return json.loads(content)



    # ------------------
    # Utilities
    # -------------------------

    def init_meta_file(self):
        """
            Create the base .meta file if it is none existing.

            Returns:
                (bool) True when file was initialized False when not.
        """
        if not os.path.exists(self.__META_FILE_PATH):      
            with open(self.__META_FILE_PATH, "w") as json_file:
                json_file.write(json.dumps(self.__base_content))
            
            return True

        return False
    

    def __append_new_data(self, key, content, new_data):
        
        if isinstance(content[key], list):
            content[key].append(new_data)

        elif isinstance(content[key], dict):
            content[key] = {
                **content[key],
                **new_data
            }
        
        else:
            raise ValueError("Error in MetaHandler.write(). Can only append to data of type list or dict. Got type {}".format(type(content[key])))

        return content


    def __add_model_config(self, model, data):
        """
            Adding model configurations to a dictionary.

            Returns:
                (dict) updated with model configurations for differen situations.
        """

        fit_config = model.get_fit_config()
        if fit_config != {}:
            data["fit"] = fit_config

        query_config = model.get_query_config()
        if query_config != {}:
            data["query"] = query_config
        
        eval_config = model.get_eval_config()
        if eval_config != {}:
            data["eval"] = eval_config
        
        return data


    # ---------------
    # Setter/-Getter
    # ------------------

    def get_models(self):
        json_data = self.read()
        key = "models"
        return json_data.get(key, self.__base_content[key])

    def get_dataset(self):
        json_data = self.read()
        key = "dataset"
        return json_data.get(key, self.__base_content[key])

    def get_params(self):
        json_data = self.read()
        key = "params"
        return json_data.get(key, self.__base_content[key])

    def get_acquisition_function(self):
        json_data = self.read()
        key = "acquisition_function"
        return json_data.get(key, self.__base_content[key])

    def get_run(self, run=None):
        """
            Access information of different experiment runs.

            Parameters:
                run (int): A specific run to be selected. Positive integer starting at 0 being the first run. (default=None)
            
            Returns:
                (dict) information about a specific run.
        """

        json_data = self.read()
        key = "run"

        run_data = json_data.get(key, self.__base_content[key])
        if run is None:
            return run_data

        if run >= 0 and run < len(run_data):
            return run_data[run]
        
        elif run < 0:
            raise ValueError("Error in MetaWriter.get_run(). Can't select negative experiment run. Expected positive integer for parameter run.")

        raise ValueError("Error in MetaWriter.get_run(). Can't select run {}, only {} runs available.".format(run, len(runs)))