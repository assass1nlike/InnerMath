import json

import numpy as np
import pytest

from innermath.cyclic import ExperimentConfig, loss_and_gradients, modular_data, run_experiment


def test_backpropagation_matches_finite_differences():
    rng = np.random.default_rng(0)
    x, y = modular_data(5)
    weights = {"w1": rng.normal(size=(10, 3))*0.1, "b1": rng.normal(size=3)*0.1,
               "w2": rng.normal(size=(3, 5))*0.1, "b2": rng.normal(size=5)*0.1}
    _, gradients = loss_and_gradients(x[:9], y[:9], weights)
    epsilon = 1e-5
    for key, value in weights.items():
        for index in np.ndindex(value.shape):
            original = value[index]
            value[index] = original + epsilon
            plus, _ = loss_and_gradients(x[:9], y[:9], weights)
            value[index] = original - epsilon
            minus, _ = loss_and_gradients(x[:9], y[:9], weights)
            value[index] = original
            assert gradients[key][index] == pytest.approx((plus-minus)/(2*epsilon), abs=1e-8)


def test_training_probe_and_artifacts_are_reproducible(tmp_path):
    config = ExperimentConfig(modulus=5, hidden=16, steps=300, seed=3)
    first = run_experiment(config, tmp_path/"first")
    second = run_experiment(config, tmp_path/"second")
    assert first == second
    assert first["task"]["train_accuracy"] > 0.95
    splits = first["splits"]
    assert set(splits["task_train"]).isdisjoint(splits["task_test"])
    assert set(splits["probe_train_orbits"]).isdisjoint(splits["probe_test_orbits"])
    assert first["probes"]["exact_control"]["test_equivariance_error"] < 1e-4
    saved = json.loads((tmp_path/"first"/"report.json").read_text())
    assert saved == first
    with np.load(tmp_path/"first"/"artifacts.npz") as artifacts:
        assert artifacts["hidden"].shape == (5, 5, 16)


@pytest.mark.parametrize("config", [ExperimentConfig(modulus=2), ExperimentConfig(steps=0),
                                         ExperimentConfig(train_fraction=1), ExperimentConfig(ridge=0)])
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        config.validate()
