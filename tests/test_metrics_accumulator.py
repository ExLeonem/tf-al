import pytest
from tf_al import MetricsAccumulator
from tf_al.metrics_accumulator import BaseMetric, ModelSpecificMetric


def mock_empty_callback(*args):
    return {}

def mock_callback_1(*args):
    return {"1": 1 , "2": 2}

def mock_callback_2(*args):
    return {"3": 3, "4": 4}


class TestBaseMetric:

    def test_invalid_callback_type(self):
        with pytest.raises(ValueError):
            metric = BaseMetric("hello", None)

        
    def test_valid_basic_metric(self):
        metric = BaseMetric(mock_callback_1, None)
        assert "1" in metric().keys() and metric.prefix is None



class TestModelSpecificMetric:

    def test_invalid_callback_type(self):
        metrics = ModelSpecificMetric()    
        with pytest.raises(ValueError):
            metrics.add_callback("hello", None)


    def test_valid_model_distinct_execution(self):
        metrics = ModelSpecificMetric()
        metrics.add_callback("mc_dropout", mock_callback_1)
        metrics.add_callback("moment_prop", mock_callback_2)

        mc_metrics = metrics("mc_dropout")
        mp_metrics = metrics("moment_prop")
        
        assert "4" in mp_metrics.keys() and "1" in mc_metrics.keys() \
            and metrics.prefix is None

    
    def test_non_registered_model(self):
        # Better to raise an error?
        metrics = ModelSpecificMetric()
        metrics.add_callback("mc_dropout", mock_callback_1)

        output = metrics("moment_prop")
        assert output == {}


class TestMetricsAccumulator:


    def test_invalid_callback_type(self):
        metrics_acc = MetricsAccumulator()
        with pytest.raises(ValueError):
            metrics_acc.track("hello")

    
    def test_valid_only_base_callback(self):
        metrics_acc = MetricsAccumulator()
        metrics_acc.track(mock_callback_2)
        metrics_acc.track(mock_callback_1)
        output = metrics_acc()
        output_keys = output.keys()
        assert "1" in output_keys and "4" in output_keys


    def test_valid_mixed_metrics(self):
        metrics_acc = MetricsAccumulator()
        metrics_acc.track(mock_callback_1)
        metrics_acc.track(mock_empty_callback),
        
        model_specific_metric = ModelSpecificMetric()
        model_specific_metric.add_callback("mc_dropout", mock_callback_2)
        metrics_acc.track(model_specific_metric)

        mp_metrics = metrics_acc("moment_propagation")
        mc_metrics = metrics_acc("mc_dropout")

        mp_keys = mp_metrics.keys()
        mc_keys = mc_metrics.keys()
        assert ("4" not in mp_keys) and ("1" in mp_keys) \
            and ("4" in mc_keys) and ("1" in mc_keys)


    def test_prefixed_metrics(self):
        metrics_acc = MetricsAccumulator()

        train_metrics = BaseMetric(mock_callback_1, "train")
        metrics_acc.track(train_metrics)

        eval_metrics = BaseMetric(mock_callback_1, "eval")
        metrics_acc.track(eval_metrics)

        output = metrics_acc()
        assert "train_1" in output.keys() and "eval_1" in output.keys()

        