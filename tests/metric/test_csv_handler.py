import os, shutil
import copy
import random
import pytest
from tf_al.metric import CsvHandler, csv_handler


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
METRICS_PATH = os.path.join(BASE_PATH, "..", "metrics")

def generate_row(num_columns=1, max_value=10):
    """
        Generate a dictionary with random number of items.
    """
    data = {}
    for i in range(num_columns):
        col_name = "column_" + str(i)
        value = int(random.random()*max_value)
        data[col_name] = value

    return data


def row_from_data(*args):
    """
        Generate a insertable row from given data.
    """

    data = {}
    for i in range(len(args)):
        col_name = "column_" + str(i)
        data[col_name] = args[i]
    
    return data




class TestCsvHandler:

    def setup_method(self):
        if not os.path.exists(METRICS_PATH):
            os.mkdir(METRICS_PATH)


    def teardown_method(self):
        if os.path.exists(METRICS_PATH):
            shutil.rmtree(METRICS_PATH)


    def test_write_basic_file(self):
        csv_handler = CsvHandler(METRICS_PATH)
        sample_data = generate_row(2)

        filename = "test_file"
        csv_handler.write(filename, sample_data)
        written_data = csv_handler.read(filename)
        assert written_data == [sample_data]

    
    def test_header_recognized(self):
        csv_handler = CsvHandler(METRICS_PATH)
        first_row = generate_row(2)
        second_row = generate_row(2)

        filename = "test_file"
        csv_handler.write(filename, first_row)
        csv_handler.write(filename, second_row)

        written_data = csv_handler.read(filename)
        assert first_row == written_data[0] and second_row == written_data[1]


    def test_write_different_types(self):
        csv_handler = CsvHandler(METRICS_PATH)
        expected = row_from_data(
            12, 
            random.random(),
            "some_string",
            [0, 12, 4, 3, 2],
            True
        )

        filename = "test_file"
        csv_handler.write(filename, expected)
        written_data = csv_handler.read(filename)
        assert written_data == [expected]


    def test_resolve_nested_dicts_with_prefix(self):
        csv_handler = CsvHandler(METRICS_PATH)
        expected = generate_row(4)

        input_dict = copy.deepcopy(expected)
        nested = {"test": 12, "test_2": False}
        input_dict["nested"] = nested
        
        # Copy values from nested into 
        for key, value in nested.items():
            key = "nested_" + key
            expected[key] = value

        filename = "test"
        csv_handler.write(filename, input_dict)
        written_data = csv_handler.read(filename)

        assert written_data == [expected]

    
    def test_write_n_rows(self):
        csv_handler = CsvHandler(METRICS_PATH)
        filename = "test_file"
                
        n_rows = 6 + int(random.random()*19)
        for i in range(n_rows):
            data = generate_row(4)
            csv_handler.write(filename, data)

        written_data = csv_handler.read(filename)
        assert len(written_data) == n_rows
    