import os, shutil
import numpy as np
import pytest

from tf_al.exp_metrics import MetaHandler
from tf_al.wrapper import Model
from tf_al import Dataset

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


    # ------
    # Test adding model information
    # ----------------------------------

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


    def test_add_uncompiled_model(self):
        mock_net = MockNet()
        model = Model(mock_net, name="test")
        
        meta_handler = MetaHandler(METRICS_PATH)
        meta_handler.init_meta_file()

        with pytest.raises(ValueError) as e:
            meta_handler.add_model(model)

    

    # -----------
    # Adding dataset info tests
    # ------------------------

    def test_add_dataset_no_split(self):
        meta_handler = MetaHandler(METRICS_PATH)
        meta_handler.init_meta_file()

        dummy_data = np.random.randn(10)
        dataset = Dataset(dummy_data,dummy_data)

        ds_name ="mnist"
        url = "http://yann.lecun.com/exdb/mnist/"
        meta_handler.add_dataset(dataset, ds_name, url)
        meta_info = meta_handler.read()
        dataset_info = meta_info["dataset"]

        assert dataset_info["name"] == ds_name and \
            dataset_info["url"] == url and \
            "splits" not in dataset_info.keys()

        
    def test_add_dataset_with_split(self):
        meta_handler = MetaHandler(METRICS_PATH)
        meta_handler.init_meta_file()

        train_data = np.random.randn(10)
        test_data = np.random.randn(30)
        dataset = Dataset(train_data, train_data, test=(test_data, test_data))

        name = "test"
        meta_handler.add_dataset(dataset, name)
        meta_info = meta_handler.read()
        dataset_info =  meta_info["dataset"]

        ds_split = dataset.get_split_ratio()
        split_ratio = {
            "train": ds_split[0],
            "test": ds_split[1],
            "eval": ds_split[2]
        }

        assert dataset_info["name"] == name and \
            dataset_info["splits"] == split_ratio and \
            "url" not in dataset_info

    
    # ------------
    # Test active learning parameters write
    # ----------------------------------------

    def test_write_active_learning_params(self):
        meta_handler = MetaHandler(METRICS_PATH)
        
        expected_params = {
            "step_size": 12,
            "rounds": 100,
            "initial_pool_size": 20,
            "num_of_experiments": 1
        }
        meta_handler.init_meta_file()
        meta_handler.add_params(**expected_params)

        written_params = meta_handler.read()
        assert written_params["params"] == expected_params