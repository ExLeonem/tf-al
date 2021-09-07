import os
import numpy as np
import pytest


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
METRICS_PATH = os.path.join(BASE_PATH, "metrics")


class TestExperimentMetrics:

    def setup_method(self):
        pass

    def teardown_method(self):
        pass


    def test_initialize_dir(self):
        

        assert True