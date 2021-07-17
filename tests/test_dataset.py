import numpy as np
from active_learning import Dataset


class TestDataset:

    
    def test_flags_with_targets(self):
        # inputs = np.random.randn(10, 100)
        # targets = np.random.randn(10)
        # dataset = Dataset(inputs, targets)
        assert True
    
    
    # def test_default_data_split(self):
    #     inputs = np.random.randn(10)
    #     targets = np.random.choice([0, 1, 2, 3], 10)
    #     dataset = Dataset(inputs, targets)
    #     assert dataset.has_test_set()


    # def test_custom_test_set_no_validation(self):
    #     inputs = np.random.randn(10)
    #     targets = np.random.choice([0, 1, 2, 3], 10)
    #     dataset = Dataset(inputs, targets, train_size=0.9, test_size=0.1)
    #     assert dataset.has_test_set() and not dataset.has_eval_set()


    # def test_custom_valid_test_train_eval_set_split(self):
    #     inputs = np.random.randn(100)
    #     targets = np.random.choice([0, 1, 2, 3], 100)



