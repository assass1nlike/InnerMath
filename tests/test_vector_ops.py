import numpy as np
import pytest

from innermath.vector_ops import OperationConfig, apply_operation


def test_add_normalizes_by_default():
    result = apply_operation("add", [3.0, 0.0], [0.0, 4.0])
    assert np.allclose(np.linalg.norm(result), 1.0)
    assert np.allclose(result, np.array([0.6, 0.8]))


def test_weighted_sum_can_skip_normalization():
    result = apply_operation(
        "weighted-sum",
        [1.0, 2.0],
        [10.0, 20.0],
        OperationConfig(alpha=2.0, beta=-0.5, normalize_output=False),
    )
    assert np.allclose(result, np.array([-3.0, -6.0]))


def test_projection():
    result = apply_operation(
        "project-a-on-b",
        [2.0, 3.0],
        [1.0, 0.0],
        OperationConfig(normalize_output=False),
    )
    assert np.allclose(result, np.array([2.0, 0.0]))


def test_rejection():
    result = apply_operation(
        "reject-a-from-b",
        [2.0, 3.0],
        [1.0, 0.0],
        OperationConfig(normalize_output=False),
    )
    assert np.allclose(result, np.array([0.0, 3.0]))


def test_slerp_returns_unit_vector():
    result = apply_operation("slerp", [1.0, 0.0], [0.0, 1.0], OperationConfig(t=0.5))
    assert np.allclose(np.linalg.norm(result), 1.0)
    assert np.allclose(result, np.array([np.sqrt(0.5), np.sqrt(0.5)]))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        apply_operation("add", [1.0], [1.0, 2.0])
