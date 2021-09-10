import os
import numpy as np
import pytest
from tf_al.exp_metrics import ExperimentSuitMetrics



BASE_PATH = os.path.dirname(os.path.realpath(__file__))
METRICS_PATH = os.path.join(BASE_PATH, "metrics")


class TestExperimentMetrics:

    def setup_method(self):
        pass

    def teardown_method(self):
        pass


    def test_initialize_dir(self):
        experiment_metrics = ExperimentSuitMetrics(METRICS_PATH)
        experiment_metrics.init_dir()

        assert True