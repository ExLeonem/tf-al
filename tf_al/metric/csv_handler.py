import os, csv
import ast
import logging

from .file_handler import FileHandler

class CsvHandler(FileHandler):

    def __init__(self, base_path):
        self.__BASE_PATH = base_path
        self.__EXT = "csv"

        # CSV Parameters
        self.__delimiter = " "
        self.__quotechar = "\""
        self.__quoting = csv.QUOTE_MINIMAL


    def write(self, filename, data, mode="a+"):
        """
            Writes a new line of data into the csv file. 

            Parameters:
                filename (str): The name of the csv file to put new data into.
                data (dict): The data to write into the csv file.
        """
        
        filename = self._add_extension(filename, self.__EXT)
        CSV_FILE_PATH = os.path.join(self.__BASE_PATH, filename)
        data = self._resolve_dict(data)

        with open(CSV_FILE_PATH, mode) as csv_file:
            
            has_header = self.has_header(csv_file)
            field_names = data.keys()
            csv_writer = self.__get_csv_writer(csv_file, field_names)
    
            # Sniff file
            
            if not has_header:
                csv_writer.writeheader()
            
            csv_writer.writerow(data)


    def read(self, filename):
        """
            Read csv data from given filename into python processable data.
            

            Parameters:
                filename (str): The filename to be read, with or without .csv extension.
            
            Returns:
                (list(dict)) of csv data.
        """

        filename = self._add_extension(filename, self.__EXT)
        CSV_FILE_PATH = os.path.join(self.__BASE_PATH, filename) 

        values = []
        with open(CSV_FILE_PATH, "r") as csv_file:
            csv_reader = self.__get_csv_reader(csv_file)
            for row in csv_reader:
                
                # Parse row values into types int, list, ...
                row = self.__auto_parse_values(row)
                values.append(row)
        
        return values


    def read_all(self):
        """
            Read all files from a directory of experiments
        """

        data = {}
        dir_content = os.listdir(self.__BASE_PATH)
        for element in dir_content:
            
            if ("." + self.__EXT) in element:
                name, ext = os.path.splitext(element)
                data[name] = self.read(element)

        return data
 

    def has_header(self, csv_file):
        """
            Check if csv file already has an header.

            Parameters:
                csv_file (): Opened file.
        """

        # Read first line to check and reset
        csv_file.seek(0)
        first_row = csv_file.readline()
        csv_file.seek(0, 2)

        has_header = False
        if first_row != "":
            has_header = csv.Sniffer().has_header(first_row)
        
        return has_header

    
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

    
    def __auto_parse_values(self, row):
        """
            Try to parse each column value into datatype of list, dict, int or float.
            When failing keeping string value.

            Returns:
                (dict) with values parsed.
        """

        for key, value in row.items():
            try:
                # Try to auto-parse value into types list, dict, boolean, int, float
                row[key] = ast.literal_eval(value)

            except ValueError:
                # Keep string type for value
                continue
        
        return row


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