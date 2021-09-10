import os, shutil
import numpy as np
import pytest
from tf_al.metric import ExperimentSuitMetrics


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
METRICS_PATH = os.path.join(BASE_PATH, "metrics")


class TestExperimentMetrics:

    def setup_method(self):
        if not os.path.exists(METRICS_PATH):
            os.mkdir(METRICS_PATH)

    def teardown_method(self):
        if os.path.exists(METRICS_PATH):
            shutil.rmtree(METRICS_PATH)


    def test_initialize_dir(self):
        experiment_metrics = ExperimentSuitMetrics(METRICS_PATH)
        experiment_metrics.init_dir()

        assert True