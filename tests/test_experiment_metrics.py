import os, shutil
import pytest
from active_learning import ExperimentSuitMetrics


METRICS_PATH = os.path.join(os.getcwd(), "test_metrics")


class TestExperimentMetricsReadWrite:

    def setup_method(self):
        """
            Build Metrics directory to perform tests.
        """
        if not os.path.exists(METRICS_PATH):
            os.mkdir(METRICS_PATH)

    
    def teardown_method(self):
        """
            Clear test directory from metrics.
        """
        if os.path.exists(METRICS_PATH):
            shutil.rmtree(METRICS_PATH)


    def test_reconstruction(self):
        metrics = ExperimentSuitMetrics(METRICS_PATH)
        experiment = "mc_dropout_test"
        values = {"accuracy": .22, "loss": 1, "size": 10}
        metrics.write_line(experiment, values)

        values = {"accuracy": .50, "loss": 0.75, "size": 20}
        metrics.write_line(experiment, values)

        new_metrics = ExperimentSuitMetrics(METRICS_PATH, verbose=True)
        assert experiment in new_metrics.experiment_files


    def test_write_reconstructed(self):
        metrics = ExperimentSuitMetrics(METRICS_PATH)
        experiment = "mc_dropout_test"
        values = {"accuracy": .22, "loss": 1, "size": 10}
        metrics.write_line(experiment, values)

        values = {"accuracy": .50, "loss": 0.75, "size": 20}
        metrics.write_line(experiment, values)

        new_metrics = ExperimentSuitMetrics(METRICS_PATH, verbose=True)
        with pytest.raises(ValueError) as e:
            new_metrics.write_line(experiment, values)


    def test_write_reconstructed_and_unlocked(self):
        metrics = ExperimentSuitMetrics(METRICS_PATH)
        experiment = "mc_dropout_test"
        values = {"accuracy": .22, "loss": 1, "size": 10}
        metrics.write_line(experiment, values)

        values = {"accuracy": .50, "loss": 0.75, "size": 20}
        metrics.write_line(experiment, values)

        new_metrics = ExperimentSuitMetrics(METRICS_PATH, verbose=True)
        new_metrics.unlock(experiment)
        new_metrics.write_line(experiment, values)
        experiment_metrics = new_metrics.read(experiment)
        assert len(experiment_metrics) == 3


    def test_write_reconstructed_overwrite(self):
        metrics = ExperimentSuitMetrics(METRICS_PATH)
        experiment = "mc_dropout_test"
        values = {"accuracy": .22, "loss": 1, "size": 10}
        metrics.write_line(experiment, values)
        metrics.write_line(experiment, values)

        new_metrics = ExperimentSuitMetrics(METRICS_PATH, verbose=True)
        new_metrics.overwrite(experiment)
        new_metrics.write_line(experiment, values)
        new_metrics.write_line(experiment, values)
        new_metrics.write_line(experiment, values)
        experiment_metrics = new_metrics.read(experiment)
        assert len(experiment_metrics) == 3

    
    def test_valid_appending_to_experiment_file(self):
        metrics = ExperimentSuitMetrics(METRICS_PATH)

        experiment = "mc_dropout_test"
        values = {"accuracy": .22, "loss": 1, "size": 10}
        metrics.write_line(experiment, values)

        values = {"accuracy": .50, "loss": 0.75, "size": 20}
        metrics.write_line(experiment, values)

        experiment_metrics = metrics.read(experiment)
        assert len(experiment_metrics) == 2


    def test_read_experiment_metrics(self):
        metrics = ExperimentSuitMetrics(METRICS_PATH)

        experiment = "mc_dropout_test"
        with pytest.raises(FileNotFoundError) as e:
            empty_metrics = metrics.read(experiment)

        values = {"accuracy": .22, "loss": 1, "size": 10}
        metrics.write_line(experiment, values)
        experiment_metrics = metrics.read(experiment)
        assert len(experiment_metrics) > 0


    def test_dataset_meta_write(self):
        metrics = ExperimentSuitMetrics(METRICS_PATH)
        metrics.add_dataset_meta("mnist", "~/datasets/mnist", 0.75)
        meta = metrics.read_meta()
        assert meta.get("dataset") != None


    def test_dataset_meta_write_non_existent_meta(self):
        metrics = ExperimentSuitMetrics(METRICS_PATH)
        os.remove(os.path.join(METRICS_PATH, ".meta.json"))
        with pytest.raises(FileNotFoundError) as e:
            metrics.add_dataset_meta("mnist", "~/datasets/mnist", 0.75, 0.25)

    
    def test_base_experiment_meta_write(self):
        metrics = ExperimentSuitMetrics(METRICS_PATH)
        params = {
            "iterations": 10,
            "step_size": 100,
            "inital_size": 10
        }
        metrics.add_experiment_meta("mc_dropout_max_entropy", "mc_dropout", "query_fn", params)
        meta = metrics.read_meta()
        assert len(meta["experiments"]) == 1

    
    def test_experiment_meta_write_non_existent_meta(self):
        metrics = ExperimentSuitMetrics(METRICS_PATH)
        params = {
            "iterations": 10,
            "step_size": 100,
            "inital_size": 10
        }

        os.remove(os.path.join(METRICS_PATH, ".meta.json"))
        with pytest.raises(FileNotFoundError) as e:
            metrics.add_experiment_meta("mc_dropout_max_entropy", "mc_dropout", "query_fn", params)


    def test_get_meta_information_for_experiment(self):
        metrics = ExperimentSuitMetrics(METRICS_PATH)
        experiment = "mc_dropout_max_entropy"
        params = {
            "iterations": 10,
            "step_size": 100,
            "initial_size": 10
        }
        metrics.add_experiment_meta("other_experiment", "mc_droput", "random", {"iterations": 10, "step_size": 10, "initial_size": 50})
        metrics.add_experiment_meta(experiment, "mc_dropout", "max_entropy", params)
        meta_info = metrics.get_experiment_meta(experiment)
        assert (meta_info["experiment_name"] == experiment) and (meta_info["params"]["initial_size"] == 10)