import os, shutil
import pytest

from active_learning import Metrics


dir_path = os.path.dirname(os.path.realpath(__file__))
METRICS_PATH = os.path.join(dir_path, "test_metrics")


class TestMetricsProcessing:

    def test_collect_base(self):
        metrics = Metrics(METRICS_PATH, keys=["accuracy"])
        collected = metrics.collect({"accuracy": 2.22, "loss": 2.2342, "other": 2342})
        keys = collected.keys()
        assert "accuracy" in keys and "loss" not in keys and "other" not in keys


    def test_collect_custom_keys(self):
        metrics = Metrics(METRICS_PATH)
        collected = metrics.collect({"accuracy": 2.22, "loss": 2.2323, "other": 323}, keys=["accuracy"])
        keys = collected.keys()
        assert ("accuracy" in keys) and ("loss" not in keys) and ("other" not in keys)


    
class TestMetricsWriteRead:

    def setup_method(self):
        """
            Create a metrics directory to test functionality
        """
        if not os.path.exists(METRICS_PATH):
            print("Create DIR: {}".format(METRICS_PATH))
            os.mkdir(METRICS_PATH)


    def teardown_method(self):
        """
            Remove generated files and directory.
        """
        if os.path.exists(METRICS_PATH) and os.path.isdir(METRICS_PATH):
            print("Remove Directory: {}".format(METRICS_PATH))
            shutil.rmtree(METRICS_PATH)


    def test_base_write(self):
        metrics_loader = Metrics(METRICS_PATH)
        assert True