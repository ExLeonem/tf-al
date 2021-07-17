
import pytest
from active_learning import Config, TrainConfig

class TestConfig:

    def test_config(self):
        first_key = 12
        second_key = 15
        config = Config(first_key=first_key, second_key=second_key)
        assert config["first_key"] == first_key
        assert config["second_key"] == second_key


    def test_valid_defaults(self):
        defaults = {
            "batch_size": 60,
            "train_size": 0.3,
        }
        train_size = 0.4
        config = Config(train_size=train_size, defaults=defaults)
        assert config["batch_size"] == defaults["batch_size"]
        assert config["train_size"] == train_size


    def test_invalid_defaults(self):
        defaults = [
            ("train_size", .6),
            ("batch_size", 40)
        ]
        with pytest.raises(ValueError) as e:
            Config(train_size=0.4, defaults=defaults)

        

class TestTrainConfig:

    def test_defaults_key_passed(self):
        defaults = {
            "batch_size": 60,
            "epochs": 12
        }
        
        with pytest.raises(ValueError) as e:
            config = TrainConfig(defaults=defaults)


    def test_only_defaults(self):
        config = TrainConfig()
        assert config["batch_size"] == 40
        assert config["optimizer"] == "adam"


    def test_overwrite_set_defaults(self):
        batch_size = 12
        config = TrainConfig(batch_size=batch_size)
        assert config["batch_size"] == batch_size
        assert config["optimizer"] == "adam"
        
