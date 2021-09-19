import os
import csv, json
import logging

from .utils import setup_logger


class ExperimentSuitMetrics:
    """
        Uses the given path to write and read experiment metrics
        and meta information.

        If the last segment of the path is not existent it will be created.

        Creating a new object pointing to an already existing metrics path will
        reconstruct all metrics files that were written. 

        WARNING: The reconstructred files will be locked for appending and writing. Can be unlocked by using the unlock() method.

        Parameters:
            base_path (str): Where to save the experiments? No recursive creation of directories.
            verbose (bool): Set debugg mode?
    """

    def __init__(self, base_path, verbose=False):
        self.logger = setup_logger(verbose, name="ExperimentSuitMetrics", default_log_level=logging.WARN)

        self.BASE_PATH = base_path
        self.META_FILE_PATH = os.path.join(base_path, ".meta.json")
        self.__setup_dir(base_path)

        # Keep track of written experiment metrics (Code 0=File was loaded, 1=File created)
        self.experiment_files = {}

        # CSV Parameters
        self.delimiter = " "
        self.quotechar = "\""
        self.quoting = csv.QUOTE_MINIMAL
        self.__load_experiments()


    def __setup_dir(self, path):
        """
            Setup a directory for a suit of experiment metrics.
        """

        # Try to create directory if non existen
        if not os.path.exists(path):
            os.mkdir(path)
        
        # Create non-existent meta.json file
        if not os.path.exists(self.META_FILE_PATH):
            # base_content = {"models": [], "dataset": {}, "params": {}, "acquisition_function": [], "run": []}
            base_content = {"experiments": []}
            self.write_meta(base_content)



    def add_dataset_meta(self, name, path, train_size, test_size=None, val_size=None):
        """
            Adding meta information about the dataset used for the experiments

            Parameters:
                name (str): The name of the dataset.
                path (str): The path to the dataset used.
                train_size (float|int): Similiar to sklearn.model_selection.train_test_split.
                test_size (float|int): the size of the test set.
                val_size (float|int): the size of the validation set.
        """


        meta = self.read_meta()
        meta["dataset"] = {
            "name": name,
            "path": path,
            "train_size": train_size
        }

        if test_size is not None:
            meta["dataset"]["test_size"] = test_size
        

        if val_size is not None:
            meta["dataset"]["val_size"] = val_size

        self.write_meta(meta)


    def add_experiment_meta(self, experiment_name, model_name, query_fn, params):
        """
            Adding meta information about an experiment to the meta file.

            Parameters:
                experiment_name (str): The name of the experiment
                model_name (str): Name of the model used
                query_fn (str): Name of the acquisition function
                params (dict): Dictionary of additional parameters to be saved. Like step_size, iterations, ...
        """
        meta = self.read_meta()
        experiments = meta["experiments"]

        experiments.append({
            "experiment_name": experiment_name,
            "model": model_name,
            "query_fn": query_fn,
            "params": params
        })

        meta["experiments"] = experiments
        self.write_meta(meta)


    # ----------
    # Read/Write files
    # -------------------------

    def write_meta(self, content):
        """
            Writes a dictionary to .meta.json.

            Parameters:
                content (dict): The meta information to be written to .meta.json
        """

        with open(self.META_FILE_PATH, "w") as json_file:
            json_file.write(json.dumps(content, indent=4))


    def read_meta(self):
        """
            Reads the meta information from the .meta.json file.

            Returns:
                (dict) of meta information.
        """
        content = {}
        with open(self.META_FILE_PATH, "r") as json_file:
            content = json_file.read()

        return json.loads(content)

    
    def write_line(self, experiment_name, values, filter_keys=None, filter_nan=True):
        """
            Writes a new line into one of the experiment files. 
            Creating the experiment file if it not already exists.

            Parameter:
                experiment_name (str): The name of the experiment performed.
                values (dict): A dictionary of values to write to the experiment file.
                filter_keys (list(str)): A list of str keys to filter keys of given values dictionary.
        """

        # Filter out empty values
        if filter_nan and isinstance(values, dict):
            values = dict(filter(lambda elem: elem[1] is not None, values.items()))

        values = self._resolve_dict(values)
        
        # Filter specific keys
        if filter_keys is not None and isinstance(filter_keys, str):
            values = {key: values[key] for key in filter_keys}

        filename = self._add_extension(experiment_name, "csv")
        file_path = os.path.join(self.BASE_PATH, filename)

        # Was metrics file reconstructed and is locked?
        if experiment_name in self.experiment_files and self.experiment_files[experiment_name] == 0:
            error_msg = "File {} was reconstructed and is locked. Use .unlock(experiment_name) to open this file up for writing.".format(experiment_name)
            raise ValueError(error_msg)
        
        mode = self.__get_mode(experiment_name)
        with open(file_path, mode) as csv_file:
            fieldnames = list(values.keys())
            csv_writer = self.__get_csv_writer(csv_file, fieldnames)
            
            # Experiment file non-existent? Overwrite mode?
            if (experiment_name not in self.experiment_files) or (self.experiment_files[experiment_name] == 2):
                self.experiment_files[experiment_name] = 1
                csv_writer.writeheader()

            csv_writer.writerow(values)


    def read(self, experiment_name):
        """
            Read metrics from a specific experiment.

            Parameters:
                experiment_name (str): The experiment to read from.

            Returns:
                (list(dict)) of accumulated experiment metrics.
        """

        # .csv extension in filename? 
        experiment_name = self._add_extension(experiment_name, "csv")

        values = []
        experiment_file_path = os.path.join(self.BASE_PATH, experiment_name) 
        with open(experiment_file_path, "r") as csv_file:

            csv_reader = self.__get_csv_reader(csv_file)
            for row in csv_reader:
                values.append(row)
            
        return values


    def unlock(self, experiment_name):
        """
            Unlocks a reconstructed file to be available to write it again.

            Parameters:
                experiment_name (str): Name of the expierment to unlock for appending.
        """

        if not experiment_name in self.experiment_files:
            return

        self.experiment_files[experiment_name] = 1


    def unlock_all(self):
        """
            Unlocks all locked files, being able to append to files again.
        """

        for key, value in self.experiment_files.items():
            if value == 0:
                self.experiment_files[key] = 1


    def overwrite(self, experiment_name):
        """
            Mark reconstructed experiment metrics to be overwriten.

            Parameters:
                experiment_name (str): Name of the experiment to mark for overwriting.
        """

        if experiment_name not in self.experiment_files:
            return
        
        self.experiment_files[experiment_name] = 2



    # ---------
    # Utilities
    # --------------------
    
    def _resolve_dict(self, values, prefix=None):
        """
            Resolves a dictionary into a flat pandas dataframe like structure.
            Nested dictionaries are getting prefixed with parent key.

            Parameters:
                values (dict): A dictionary of keys. Can include dictionaries with single level nesting.

            Returns:
                (dict) a flattened dictionary.
        """
        
        flattened_dict = {}
        for key, value in values.items():
        
            # Copy flat values into flattened dictionary
            prefixed_key = key if prefix is None else (prefix + "_" + key)
            if not isinstance(values[key], dict):
                flattened_dict[prefixed_key] = value
                continue
            
            resolved = self._resolve_dict(values[key], prefix=prefixed_key)
            flattened_dict.update(resolved)

        return flattened_dict


    def _add_extension(self, filename, ext):
        """
            Adds an extension to a filename.

            Parameters:
                filename (str): The filename to check for the extension
                ext (str): The file extension to add and check for
            
            Returns:
                (str) the file name with a file extension appended. 
        """

        if ext not in filename:
            return filename + "." + ext
        
        return filename


    def __load_experiments(self):
        """
            Reconstrcut metrics from files available files.
        """

        if not os.path.exists(self.BASE_PATH):
            return

        dir_content = os.listdir(self.BASE_PATH)
        for element in dir_content:
            
            # Skip sub-directories
            element_path = os.path.join(self.BASE_PATH, element)
            if not os.path.isfile(element_path):
                continue

            # Skip meta file
            if ".meta.json" in element:
                continue
            
            # Strip extension off of filename
            name, ext = os.path.splitext(element)
            self.experiment_files[name] = 0
        


    # -----------
    # Getter/-Setter
    # ------------------

    def __get_mode(self, experiment_name):
        default_mode = "a"
        if experiment_name not in self.experiment_files:
            return default_mode

        status = self.experiment_files[experiment_name]
        if status == 2:
            return "w"

        return default_mode


    def __get_csv_params(self):
        return {
            "delimiter": self.delimiter,
            "quotechar": self.quotechar,
            "quoting": self.quoting
        }


    def __get_csv_writer(self, file, fieldnames):
        csv_params = self.__get_csv_params()
        return csv.DictWriter(file, fieldnames, **csv_params)

    
    def __get_csv_reader(self, file):
        csv_params = self.__get_csv_params()
        return csv.DictReader(file, **csv_params)


    def get_dataset_info(self):
        """
            Read 

            Returns:
                (dict) containing meta information about the used dataset for the experiment
        """
        
        meta = self.read_meta()
        return meta.get("dataset", None)


    def get_experiment_meta(self, experiment_name):
        """

            Parameter:
                experiment_name (self): The name of the experiment.
        """
        meta = self.read_meta()
        experiments = meta["experiments"]
        experiment_information = {}
        for experiment in experiments:
            
            if experiment["experiment_name"] == experiment_name:
                experiment_information = experiment

        return experiment_information
        