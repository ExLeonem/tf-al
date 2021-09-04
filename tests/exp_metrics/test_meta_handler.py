import os, shutil
import pytest
from tf_al.exp_metrics import MetaHandler


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
METRICS_PATH = os.path.join(BASE_PATH, "..", "metrics")


class TestMetaHandler:

    def setup_method(self):
        if not os.path.exists(METRICS_PATH):
            os.mkdir(METRICS_PATH)


    def teardown_method(self):
        if os.path.exists(METRICS_PATH):
            shutil.rmtree(METRICS_PATH)

    def test_init_meta_file(self):
        meta_handler = MetaHandler(METRICS_PATH)
        meta_handler.init_meta_file()
        content = meta_handler.read()

        expected_keys = Set(["models", "dataset", "params", "acquisition_function", "run"])
        written_keys = Set(content.keys())
        assert expected_keys == written_keys


    def test_meta_file_init(self):
        assert False

    
    def test_some(self):
        assert False