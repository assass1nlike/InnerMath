"""Numerical probes for real representations of a cyclic group.

States have shape (orbit, group_element, feature). Row vectors act on the right:
h(g + 1) ~= h(g) @ T. No equivariance is imposed during fitting.
"""
from __future__ import annotations

import numpy as np


def _states(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.ndim != 3 or min(value.shape) < 1 or value.shape[1] < 2:
        raise ValueError("Expected (orbits, group_order >= 2, features) states.")
    if not np.isfinite(value).all():
        raise ValueError("States must be finite.")
    return value


def relative_error(predicted: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - expected) / max(np.linalg.norm(expected), 1e-12))


def fit_generator(states: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Fit a linear generator by ridge regression on complete training orbits.

    ridge is relative to the largest eigenvalue of X.T @ X, making the fit
    invariant to a common rescaling of the features.
    """
    states = _states(states)
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive.")
    x = states.reshape(-1, states.shape[-1])
    y = np.roll(states, -1, axis=1).reshape(x.shape)
    gram = x.T @ x
    scale = max(float(np.linalg.norm(gram, 2)), 1e-12)
    return np.linalg.solve(gram + ridge * scale * np.eye(x.shape[1]), x.T @ y)


def character_energy(states: np.ndarray) -> list[float]:
    """Energy in C_n characters exp(2*pi*i*k*g/n), including the trivial one.

    A DFT exists for ANY indexed orbit: this is descriptive, not evidence of
    an equivariant learned representation. Energies depend on feature metric.
    """
    states = _states(states)
    energy = np.sum(np.abs(np.fft.fft(states, axis=1, norm="ortho")) ** 2, axis=(0, 2))
    return (energy / max(float(energy.sum()), 1e-12)).tolist()


def probe_cyclic(train: np.ndarray, test: np.ndarray, ridge: float = 1e-6) -> tuple[dict, np.ndarray]:
    """Fit on training orbits, evaluate only on disjoint held-out orbits."""
    train, test = _states(train), _states(test)
    if train.shape[1:] != test.shape[1:]:
        raise ValueError("Train and test group orders and feature dimensions must agree.")
    if np.linalg.norm(train) < 1e-12 or np.linalg.norm(test) < 1e-12:
        raise ValueError("A zero representation is not an informative probe input.")
    order, width = train.shape[1:]
    generator = fit_generator(train, ridge)
    identity = np.eye(width)
    closure = np.linalg.matrix_power(generator, order)
    errors = [relative_error(test @ np.linalg.matrix_power(generator, k),
                             np.roll(test, -k, axis=1)) for k in range(1, order)]
    eigenvalues = np.linalg.eigvals(generator)
    roots = np.exp(2j * np.pi * np.arange(order) / order)
    report = {
        "train_equivariance_error": relative_error(train @ generator, np.roll(train, -1, axis=1)),
        "test_equivariance_error": errors[0],
        "test_power_errors": errors,
        "test_mean_power_error": float(np.mean(errors)),
        "test_cycle_error": relative_error(test @ closure, test),
        "ambient_cycle_error": relative_error(closure, identity),
        "train_rank": int(np.linalg.matrix_rank(train.reshape(-1, width))),
        "feature_dimension": width,
        "eigenvalues": [[float(v.real), float(v.imag)] for v in eigenvalues],
        "mean_eigenvalue_distance_to_roots": float(np.abs(eigenvalues[:, None] - roots).min(axis=1).mean()),
        "test_character_energy": character_energy(test),
    }
    return report, generator


def exact_cyclic_orbits(order: int, orbits: int, seed: int = 0) -> np.ndarray:
    """Positive control: two rotation irreps in a randomly rotated basis."""
    if order < 5 or orbits < 2:
        raise ValueError("Positive control requires order >= 5 and at least two orbits.")
    rng = np.random.default_rng(seed)
    starts = rng.normal(size=(orbits, 4))
    basis, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    result = []
    for g in range(order):
        action = np.zeros((4, 4))
        for block, frequency in enumerate((1, 2)):
            angle = 2 * np.pi * frequency * g / order
            c, s = np.cos(angle), np.sin(angle)
            action[2*block:2*block+2, 2*block:2*block+2] = [[c, -s], [s, c]]
        result.append(starts @ action @ basis)
    return np.stack(result, axis=1)
