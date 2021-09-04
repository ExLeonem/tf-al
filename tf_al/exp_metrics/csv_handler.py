import os, csv, logging


class CsvHandler:

    def __init__(self, base_path):
        self.__BASE_PATH = base_path
        self.__META_FILE_PATH = os.path.join(base_path, ".meta.json")

        # CSV Parameters
        self.delimiter = " "
        self.quotechar = "\""
        self.quoting = csv.QUOTE_MINIMAL
        self.__load_experiments()


    def write(self, data):
        pass
    
    
    # -----------
    # Utilities
    # ----------------

    