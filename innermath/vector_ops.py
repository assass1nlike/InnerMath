from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


EPSILON = 1e-12


@dataclass(frozen=True)
class OperationConfig:
    """Parameters shared by vector operations."""

    t: float = 0.5
    alpha: float = 1.0
    beta: float = 1.0
    normalize_output: bool = True


def as_vector(value: np.ndarray | list[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"Expected a 1D vector, got shape {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Vector contains NaN or infinite values.")
    return vector


def ensure_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Vector shapes must match, got {a.shape} and {b.shape}.")


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def unit(vector: np.ndarray) -> np.ndarray:
    length = norm(vector)
    if length < EPSILON:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / length


def maybe_normalize(vector: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return vector
    length = norm(vector)
    if length < EPSILON:
        return vector
    return vector / length


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a_unit = unit(a)
    b_unit = unit(b)
    dot = float(np.clip(np.dot(a_unit, b_unit), -1.0, 1.0))

    if abs(dot) > 1.0 - 1e-7:
        return unit(lerp(a_unit, b_unit, t))

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (
        np.sin((1.0 - t) * theta) / sin_theta * a_unit
        + np.sin(t * theta) / sin_theta * b_unit
    )


def project(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denominator = float(np.dot(b, b))
    if denominator < EPSILON:
        raise ValueError("Cannot project onto a near-zero vector.")
    return float(np.dot(a, b)) / denominator * b


def reject(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a - project(a, b)


def reflect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 2.0 * project(a, b) - a


def cosine_bisector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return unit(unit(a) + unit(b))


def safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a), where=np.abs(b) >= EPSILON)


def _weighted_sum(a: np.ndarray, b: np.ndarray, config: OperationConfig) -> np.ndarray:
    return config.alpha * a + config.beta * b


OperationFn = Callable[[np.ndarray, np.ndarray, OperationConfig], np.ndarray]


OPERATIONS: dict[str, OperationFn] = {
    "add": lambda a, b, config: a + b,
    "subtract": lambda a, b, config: a - b,
    "average": lambda a, b, config: (a + b) / 2.0,
    "weighted-sum": _weighted_sum,
    "lerp": lambda a, b, config: lerp(a, b, config.t),
    "slerp": lambda a, b, config: slerp(a, b, config.t),
    "hadamard": lambda a, b, config: a * b,
    "divide": lambda a, b, config: safe_divide(a, b),
    "project-a-on-b": lambda a, b, config: project(a, b),
    "reject-a-from-b": lambda a, b, config: reject(a, b),
    "reflect-a-across-b": lambda a, b, config: reflect(a, b),
    "cosine-bisector": lambda a, b, config: cosine_bisector(a, b),
}


def list_operations() -> list[str]:
    return sorted(OPERATIONS)


def apply_operation(
    operation: str,
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    config: OperationConfig | None = None,
) -> np.ndarray:
    config = config or OperationConfig()
    a_vector = as_vector(a)
    b_vector = as_vector(b)
    ensure_same_shape(a_vector, b_vector)

    try:
        result = OPERATIONS[operation](a_vector, b_vector, config)
    except KeyError as exc:
        options = ", ".join(list_operations())
        raise ValueError(f"Unknown operation {operation!r}. Available operations: {options}.") from exc

    return maybe_normalize(as_vector(result), config.normalize_output)
