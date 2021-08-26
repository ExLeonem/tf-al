import pytest
import numpy as np
from tf_al import Pool


class TestPool:

    def test_has_unlabeled(self):
        test_inputs = np.random.randn(50, 28, 28, 1)
        new_pool = Pool(test_inputs)
        assert new_pool.has_unlabeled() and not new_pool.has_labeled()


    def test_has_labeled(self):
        test_inputs = np.random.randn(50, 28, 28, 1)
        new_pool = Pool(test_inputs)
        new_pool.annotate([0], 1)
        inputs, targets = new_pool.get_labeled_data()
        assert new_pool.has_labeled() and targets == np.array([1])    
    

    def test_annotate(self):
        test_inputs = np.random.randn(50, 28, 28, 1)
        new_pool = Pool(test_inputs)
        indices = [0, 2, 5, 12]
        new_pool.annotate(indices, [1, 0, 1, 1])
        inputs, targets = new_pool.get_labeled_data()
        assert len(inputs) == len(indices)


    def test_annotate_pseudo(self):
        test_inputs = np.random.randn(50)
        test_targets = np.random.choice([0, 1, 2], 50)
        new_pool = Pool(test_inputs,test_targets)

        indices = np.array([0, 2, 5, 12])
        new_pool.annotate(indices)
        inputs, targets = new_pool.get_labeled_data()
        assert np.all(test_targets[indices] == targets)


    def test_annotate_shortcut(self):
        test_inputs = np.random.randn(50)
        new_pool = Pool(test_inputs)
        indices = [0, 2, 5, 12]
        targets = [2, 5, 1, 0]
        new_pool[indices] = targets
        inputs, targets = new_pool.get_labeled_data()
        assert len(inputs) == len(indices)


    def test_get_by(self):
        test_inputs = np.array([0, 2, 5, 12])
        new_pool = Pool(test_inputs)
        indices = [0, 1]
        values = new_pool.get_inputs_by(indices)
        true_values = test_inputs[np.array(indices)]
        assert  np.all(values == true_values)
    
    
    def test_get_length_unlabeled(self):
        test_inputs = np.random.randn(50)
        new_pool = Pool(test_inputs)
        assert new_pool.get_length_labeled() == 0

        new_pool[1] = 0
        assert new_pool.get_length_labeled() == 1

        new_pool[[2, 5]] = [1, 0]
        assert new_pool.get_length_labeled() == 3

    
    def test_get_length_labeled(self):
        test_inputs = np.random.randn(50)
        new_pool = Pool(test_inputs)
        assert new_pool.get_length_unlabeled() == len(test_inputs)

        new_pool[0] = 1
        assert new_pool.get_length_unlabeled() != len(test_inputs)


    def test_get_labeld_data(self):
        test_inputs = np.random.randn(50)
        new_pool = Pool(test_inputs)
        inputs, targets = new_pool.get_labeled_data()
        assert len(targets) == 0

        value = np.array([1])
        new_pool[0] = value
        inputs, targets = new_pool.get_labeled_data()
        assert targets == value


    def test_get_unlabeled_data(self):
        test_inputs = np.random.randn(50)
        new_pool = Pool(test_inputs)
        inputs, indices = new_pool.get_unlabeled_data()
        assert len(indices) == len(test_inputs)

        new_pool[0] = 1
        inputs, indices = new_pool.get_unlabeled_data()
        assert len(indices) != len(test_inputs)


    def test_set_explicit_initial_indices(self):
        test_inputs = np.random.randn(50)
        test_targets = np.random.choice([0, 1, 2], 50)
        new_pool = Pool(test_inputs, test_targets)

        explicit_indices = [0, 6, 10, 15]
        new_pool.init(explicit_indices)

        inputs, targets = new_pool.get_labeled_data()

        equal_inputs = np.all(test_inputs[explicit_indices] == inputs)
        equal_targets = np.all(test_targets[explicit_indices] == targets)
        assert len(inputs) == len(explicit_indices) and equal_inputs and equal_targets

    
    def test_set_explicit_initial_indices_array_valid(self):
        test_inputs = np.random.randn(50)
        test_targets = np.random.choice([0, 1, 2], 50)
        new_pool = Pool(test_inputs, test_targets)

        explicit_indices = np.array([0, 7, 12, 19, 32])
        new_pool.init(explicit_indices)

        inputs, targets = new_pool.get_labeled_data()
        equal_inputs = np.all(test_inputs[explicit_indices] == inputs)
        equal_targets = np.all(test_targets[explicit_indices] == targets)
        assert len(inputs) == len(explicit_indices) and equal_inputs and equal_targets


    def test_set_explicit_initial_indices_array_invalid(self):
        test_inputs = np.random.randn(50)
        test_targets = np.random.choice([0, 1, 2], 50)
        new_pool = Pool(test_inputs, test_targets)

        explicit_indices = np.array([60, 120])        
        with pytest.raises(IndexError) as e:
            new_pool.init(explicit_indices)

        
    def test_set_explicit_initial_indices_array_invalid_empty(self):
        test_inputs = np.random.randn(50)
        test_targets = np.random.choice([0, 1, 2], 50)
        new_pool = Pool(test_inputs, test_targets)

        explicit_indices = np.array([])        
        with pytest.raises(IndexError) as e:
            new_pool.init(explicit_indices)
    

