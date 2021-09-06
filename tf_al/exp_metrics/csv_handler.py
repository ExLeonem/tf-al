import os, csv, logging
from .file_handler import FileHandler

class CsvHandler(FileHandler):

    def __init__(self, base_path):
        self.__BASE_PATH = base_path
        # self.__CSV_FILE_PATH = os.path.join(base_path, ".meta.json")

        # CSV Parameters
        self.__delimiter = " "
        self.__quotechar = "\""
        self.__quoting = csv.QUOTE_MINIMAL


    def add(self, filename, data, ):
        """
            Writes a new line of data into the csv file. 

            Parameters:
                filename (str): The name of the csv file to put new data into.
                data (dict): The data 
        """
        
        filename = self._add_extension(filename, "csv")
        CSV_FILE_PATH = os.path.join(self.__BASE_PATH, filename)

        with open(CSV_FILE_PATH, "a") as csv_file:
            field_names = data.keys()
            csv_writer = self.__get_csv_writer(csv_file, field_names)
    
            # Sniff file
            if ()

    def read(self, filename):
        """
            Read csv data from given filename into python processable data.
            

            Parameters:
                filename (str): The filename to be read, with or without .csv extension.
            
            Returns:
                (list(dict)) of csv data.
        """

        filename = self._add_extension(filename, "csv")
        CSV_FILE_PATH = os.path.join(self.__BASE_PATH, filename) 

        values = []
        with open(CSV_FILE_PATH, "r") as csv_file:
            csv_reader = self.__get_csv_reader(csv_file)
            for row in csv_reader:
                values.append(row)
        
        return values


    
    # -----------
    # Utilities
    # ----------------

    def __get_csv_writer(self, file, fieldnames):
        csv_params = self.__get_csv_params()
        return csv.DictWriter(file, fieldnames, **csv_params)


    def __get_csv_reader(self, file):
        csv_params = self.__get_csv_params()
        return csv.DictReader(file, **csv_params)


    def __get_csv_params(self):
        return {
            "delimiter": self.__delimiter,
            "quotechar": self.__quotechar,
            "quoting": self.__quoting
        }