import os, shutil
import uuid
import pytest
from tf_al.exp_metrics import MetaHandler
from tf_al.wrapper import Model


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
METRICS_PATH = os.path.join(BASE_PATH, "..", "metrics")


class MockOptimizer:

    def get_config(self):
        return {
            "lr": 0.001,
            "delta": 0.1
        }


class MockNet:

    def compile(self, *args, **kwargs):
        self.optimizer = MockOptimizer()
        self.loss = "sparse_categorical_entropy"        


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

        expected_keys = set(["models", "dataset", "params", "acquisition_function", "run"])
        written_keys = set(content.keys())
        assert expected_keys == written_keys

    def test_added_model_valid(self):
        mock_net = MockNet()
        model = Model(mock_net, name="test")
        model.compile()

        # Setup meta file and write model information
        meta_handler = MetaHandler(METRICS_PATH) 
        meta_handler.init_meta_file()
        meta_handler.add_model(model)

        # Get written meta information
        json_content = meta_handler.read()
        model_content = json_content["models"][0]

        equal_ids = model_content["id"] == model.get_id()
        equal_name = model_content["name"] == model.get_model_name()
        

        assert equal_ids and equal_name


    def test_added_model_uncompiled(self):
        pass