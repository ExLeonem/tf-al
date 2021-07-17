import numpy as np
import pytest
from active_learning import Oracle, Pool


class TestOracle:

    def test_pseudo_annotation(self):
        test_inputs = np.random.randn(10)
        test_targets = np.random.choice([0, 1, 2], 10)
        pool = Pool(test_inputs, test_targets)

        indices = np.array([0, 5])
        oracle = Oracle(pseudo_mode=True)
        oracle.annotate(pool, indices)
        
        inputs, targets = pool.get_labeled_data()
        assert np.all(targets == test_targets[indices])

    
    def test_annotation(self):
        test_inputs = np.random.randn(10)
        test_targets = np.random.choice([0, 1, 2], 10)
        pool = Pool(test_inputs)

        indices = np.array([0, 1, 5])
        callback_mock = lambda p, i: p.annotate(i, test_targets[i])
        oracle = Oracle(callback_mock)
        oracle.annotate(pool, indices)
        
        inputs, targets = pool.get_labeled_data()
        assert np.all(targets == test_targets[indices])


    def test_annotation_without_callback(self):
        test_inputs = np.random.randn(10)
        # test_targets = np.random.choice([0, 1, 2], 10)
        pool = Pool(test_inputs)

        indices = np.array([0, 2, 5])
        oracle = Oracle()

        # No callback was set for the annotation procedure
        with pytest.raises(ValueError) as e:
            oracle.annotate(pool, indices)

        
    def test_init_pool_no_callback(self):
        test_inputs = np.random.randn(10)
        pool = Pool(test_inputs)

        oracle = Oracle()
        with pytest.raises(ValueError) as e:
            oracle.init(pool, 12)
            
        
    def test_init_pool_with_callback(self):
        test_inputs = np.random.rand(10)
        test_targets = np.random.choice([0, 1, 2], 10)
        pool = Pool(test_inputs)

        mock_callback = lambda p, i: p.annotate(i, test_targets[i])
        oracle = Oracle(callback=mock_callback)

        oracle.init(pool, 2)
        all_indices = pool.get_indices()
        selected_indices = all_indices == -1
        inputs, targets = pool.get_labeled_data()
        assert np.all(targets == test_targets[selected_indices]) and len(targets) == 2
