import pytest, random
import numpy as np
from tf_al.score import qeff

class TestQueryEfficency:


    def test_single_ints(self):
        main = random.randint(0, 20)
        baseline = random.randint(0, 20)
        result = qeff(main, baseline)
        expected = ((main-baseline)/max(main, baseline))
        assert expected == pytest.approx(result) and \
            result <= 1 and result >= -1


    def test_multi_round_experiment(self):
        num_rounds = random.randint(5, 20)
        main = np.random.randint(0, 20, num_rounds)
        baseline = np.random.randint(0, 20, num_rounds)

        main_mean = np.mean(main)
        main_std = np.std(main)
        base_mean = np.mean(baseline)
        base_std = np.std(baseline)

        out_mean, out_std = qeff(main, baseline)
        assert out_mean >= -1 and out_mean <= 1
        assert out_std >= -1 and out_std <= 1

        exp_mean = (main_mean-base_mean)/max(main_mean, base_mean)
        exp_std = (main_std-base_std)/max(main_std, base_std)
        assert out_mean == pytest.approx(exp_mean) and \
            out_std == pytest.approx(exp_std)

        
    def test_multi_experiment_qeff(self):
        num_rounds = random.randint(5, 20)
        num_experiments = random.randint(2, 10)

        main = np.random.randint(0, 20, (num_experiments, num_rounds))
        baseline = np.random.randint(0, 20, (num_experiments, num_rounds))

        main_mean = np.mean(np.mean(main, axis=0))
        main_std = np.mean(np.std(main, axis=0))

        base_mean = np.mean(np.mean(baseline, axis=0))
        base_std = np.mean(np.std(baseline, axis=0))
        

        out_mean, out_std = qeff(main, baseline)
        assert out_mean >= -1 and out_mean <= 1
        assert out_std >= -1 and out_std <= 1

        exp_mean = (main_mean-base_mean)/max(main_mean, base_mean)
        exp_std = (main_std-base_std)/max(main_std, base_std)
        assert exp_mean == pytest.approx(out_mean) and \
            exp_std == pytest.approx(out_std)


    def test_invalid_shapes(self):
        main = np.zeros(15)
        baseline = np.zeros(14)
        with pytest.raises(ValueError):
            output = qeff(main, baseline)

    
    def test_mixed_types(self):
        main = np.zeros(15)
        baseline = 14.13
        with pytest.raises(ValueError):
            output = qeff(main, baseline)
    

    def test_invalid_dimensionality(self):
        shape = (2, 2, 2)
        main = np.zeros(shape)
        baseline = np.zeros(shape)
        with pytest.raises(ValueError):
            output = qeff(main, baseline)
        
