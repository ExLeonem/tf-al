import os, sys, csv

class Metrics:
    """
        Uses the given path to create 
        Prepares and writes metrics into a csv file.

        Parameters:
            base_path (str): The base path where to save the metrics.
            keys (list(str)): A list of keys.
    """

    def __init__(self, base_path, keys=["accuracy", "loss"]):
        self.metric_keys = keys
        self.BASE_PATH = base_path
        self.EXT = "csv"

        # CSV Parameters
        self.delimiter = " "
        self.quotechar = "\""
        self.quoting = csv.QUOTE_MINIMAL


    def write_line(self, filename, values):
        file_path = os.path.join(self.BASE_PATH, filename+"."+self.EXT)
        with open(file_path, "a", newline="") as csv_file:
            pass
    

    def write(self, filename, values):
        """
            Write given values into a csv file.

            Parameters:
                filename (str): The name of the file.
                values (list(dict)): A dictionary of metrics/values to write into a .csv file.
        """

        file_path = os.path.join(self.BASE_PATH, filename+"."+self.EXT)
        with open(file_path, "w", newline="") as csv_file:
            
            # Setup csv file
            file_writer = csv.DictWriter(
                csv_file, delimiter=self.delimiter, 
                quotechar=self.quotechar, quoting=self.quoting, fieldnames=self.metric_keys)

            # Create content of csv file
            file_writer.writeheader()
            for line in values:
                collected = self.collect(line)
                file_writer.writerow(collected)

    
    def read(self, filename):
        """
            Read a .csv file of metrics.

            Parameters:
                filename (str): The filename to read in.

            Returns:
                (list(dict)) a list of metric values, per trained iteration.
        """

        values = []

        if not ("."+self.EXT in filename):
            filename = filename + "." + self.EXT 

        file_path = os.path.join(self.BASE_PATH, filename)
        with open(file_path, "r") as csv_file:

            reader = csv.DictReader(
                filter(lambda row: row[0] != "#", csv_file), 
                delimiter=self.delimiter, 
                quotechar=self.quotechar
            )

            for row in reader:
                values.append(row)

        return values



    # -------------
    # Utilities
    # ------------------

    def collect(self, values, keys=None):
        """
            Collect metric values from a dictionary of values.

            Parameter:
                values (dict): A collection of values collected during training

            Returns:
                (dict) A subset of metrics extracted from the values. 
        """
        # Set default keys to use
        if keys is None:
            keys = self.metric_keys

        return {key: self.__prepare_value(value) for key, value in values.items() if key in keys}


    def __prepare_value(self, value):
        """
            Prevent's saving list of single values.
        """
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        
        return value


    # -------------
    # Setter/-Getter
    # ------------------

    def get_path(self):
        return self.BASE_PATH



def save_history(history, path, filename):
    """
        Saves values of history to the path.
    """
    metrics = Metric(path)
    metrics.write(history, filename)


def read_history(path, filename):
    """
        Reads values from the saved history.
    """
    metrics = Metric(path)
    return metric.read(filename)


def aggregates_per_key(history):
    """
        Aggregate values per key
    """
    
    if len(history) == 0:
        return history

    sample_entry = history[0]
    keys = list(sample_entry.keys())
    aggregates = {key: [] for key in keys}

    for entry in history:
        for key, value in entry.items():
            aggregates[key].append(value)

    return aggregates