import os, json, logging
# from ..utils import setup_logger


class MetaHandler:
    """
        Write meta information about the active learning experiment out into
        a json file.

        Parameters: 
            base_path (str): The base path where to put the .meta.json file.
    """

    def __init__(self, base_path, verbose=False):
        # self.__logger = setup_logger(verbose, name="MetaWriter", default_log_level=logging.WARN)
        self.__BASE_PATH = base_path        
        self.__META_FILE_PATH = os.path.join(base_path, ".meta.json")

        # What goes initialy into the .meta.json file
        self.__base_content = {
            "models": [], 
            "dataset": {}, 
            "params": {}, 
            "acquisition_function": [], 
            "run": []
        }


    def add_model(self, name, loss, optimizer, fit, query):
        """
            Add information about a specific model.
        """
        
        pass

    
    def add_dataset(
        self, 
        name, 
        url=None, 
        test_size=None, 
        train_size=None, 
        val_size=None
    ):
        """
            Add information about the used dataset.
        """

        split_ratio = {}
        data = {"name": name}

        if url is not None:
            data["url"] = url
        
        if split_ratio != {}:
            data["split"] = split_ratio

        self.write("dataset", data)


    def add_params(self, **kwargs):
        pass


    def write(self, path, data, replace=True, create=False):
        """
            Write data into the meta file under given key.

            Parameters:
                path (str|list(str)): Single key or series of keys where to put data.
                data (any()): The data to put under this key.
                replace (bool): Wether or not replace existing data,
                create (bool): Create non-existent paths or keys.
        """
        content = self.read()
        if isinstance(path, str):
            content[path] = data

        # Resolve path
        if isinstance(path, list):
            content = self.__resolve(path, data, content, replace, create)

        # Overwrite old json meta information
        with open(self.__META_FILE_PATH, "r") as json_file:
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
    

    def __resolve(self, path, new_data, old_content, replace, create):
        """

        """
        pass


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