import os, shutil
import pytest
from tf_al.exp_metrics import CsvHandler


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
METRICS_PATH = os.path.join(BASE_PATH, "..", "metrics")

class TestCsvHandler:

    def setup_method(self):
        if not os.path.exists(METRICS_PATH):
            os.mkdir(METRICS_PATH)


    def teardown_method(self):
        if os.path.exists(METRICS_PATH):
            shutil.rmtree(METRICS_PATH)


    def test_read_data(self):
        assert False