import pytest, random
import numpy as np
from tf_al.score import leff


class TestLabelEfficiency:


    def test_valid_numeric_inputs(self):
        main = random.random()
        baseline = random.random()

        result = leff(main, baseline)
        expected = (main-baseline)

        assert result == pytest.approx(expected) and \
            result <= 1 and result >= -1

    
    def test_valid_array_inputs(self):
        num_rounds = random.randint(2, 20)
        main = np.random.random(num_rounds)
        baseline = np.random.random(num_rounds)

        out_mean, out_std = leff(main, baseline)
        expected = (main-baseline)
        assert len(out_mean) == len(expected)

    
    def test_valid_multi_experiment_inputs(self):
        num_rounds = random.randint(2, 20)
        num_experiments = random.randint(2, 10)
        main = np.random.random((num_experiments, num_rounds))
        baseline = np.random.random((num_experiments, num_rounds))

        exp_mean = np.mean(main, axis=0)-np.mean(baseline, axis=0)
        exp_std = np.std(main, axis=0)-np.std(baseline, axis=0)

        out_mean, out_std = leff(main, baseline)
        assert len(out_mean) == len(exp_mean)
        assert len(out_std) == len(exp_std)


    def test_missmatching_shapes(self):
        main = np.zeros(15)
        baseline = np.zeros(10)
        with pytest.raises(ValueError):
            output = leff(main, baseline)

    
    def test_type_missmatch(self):
        main = .93
        baseline = np.zeros(10)

        with pytest.raises(ValueError):
            output = leff(main, baseline)


    def test_invalid_dimensionality(self):
        main = np.zeros((2, 2, 2))
        baseline = np.zeros((2, 2, 2))
        with pytest.raises(ValueError):
            output = leff(main, baseline)