import numpy as np
import pytest

from innermath.structures import character_energy, exact_cyclic_orbits, probe_cyclic


def test_exact_group_action_generalizes_and_closes():
    states = exact_cyclic_orbits(11, 12, seed=7)
    report, _ = probe_cyclic(states[:8], states[8:], ridge=1e-10)
    assert report["test_equivariance_error"] < 1e-8
    assert report["test_mean_power_error"] < 1e-7
    assert report["ambient_cycle_error"] < 1e-7
    assert report["mean_eigenvalue_distance_to_roots"] < 1e-8
    energy = np.array(character_energy(states))
    assert energy.sum() == pytest.approx(1)
    assert energy[[1, 2, 9, 10]].sum() == pytest.approx(1)


def test_independent_shuffle_breaks_shared_action():
    states = exact_cyclic_orbits(11, 12, seed=7)
    rng = np.random.default_rng(9)
    shuffled = np.stack([orbit[rng.permutation(11)] for orbit in states])
    report, _ = probe_cyclic(shuffled[:8], shuffled[8:])
    assert report["test_equivariance_error"] > 0.5


def test_orthogonal_change_of_basis_preserves_probe():
    states = exact_cyclic_orbits(7, 10, seed=5)
    basis, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(4, 4)))
    left, generator = probe_cyclic(states[:7], states[7:])
    right, rotated = probe_cyclic(states[:7] @ basis, states[7:] @ basis)
    assert np.allclose(rotated, basis.T @ generator @ basis)
    assert left["test_equivariance_error"] == pytest.approx(right["test_equivariance_error"])


@pytest.mark.parametrize("bad", [np.zeros((2, 7, 4)), np.zeros((7, 4)), np.full((2, 7, 4), np.nan)])
def test_uninformative_or_invalid_states_rejected(bad):
    with pytest.raises(ValueError):
        probe_cyclic(bad, bad)
