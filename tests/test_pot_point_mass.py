# pytest -v
# tests/test_pot_point_mass.py
from __future__ import annotations

import numpy as np
import pytest

from gravdyn import pot_point_mass


def test_pot_point_mass_single_point():
    mu = 10.0
    stat = [2.0, 0.0, 0.0]

    p, acc = pot_point_mass(mu=mu, stat=stat)

    expected_p = mu / 2.0
    expected_acc = np.array([-mu / (2.0**2), 0.0, 0.0])

    assert np.isclose(np.asarray(p), expected_p, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(acc), expected_acc, rtol=1e-12, atol=1e-12)


def test_pot_point_mass_multiple_points():
    mu = 10.0
    stat = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 5.0],
    ])

    p, acc = pot_point_mass(mu=mu, stat=stat)

    expected_p = np.array([
        10.0 / 2.0,
        10.0 / 4.0,
        10.0 / 5.0,
    ])

    expected_acc = np.array([
        [-10.0 / (2.0**2), 0.0, 0.0],   # [-2.5, 0, 0]
        [0.0, -10.0 / (4.0**2), 0.0],   # [0, -0.625, 0]
        [0.0, 0.0, -10.0 / (5.0**2)],   # [0, 0, -0.4]
    ])

    np.testing.assert_allclose(np.asarray(p), expected_p, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(acc), expected_acc, rtol=1e-12, atol=1e-12)


def test_pot_point_mass_returns_single_shapes_for_single_point():
    mu = 3.0
    stat = np.array([1.0, 2.0, 2.0])

    p, acc = pot_point_mass(mu=mu, stat=stat)

    # For a single point, the function returns scalar-like p and shape-(3,) acc
    assert np.asarray(p).shape == ()
    assert np.asarray(acc).shape == (3,)


def test_pot_point_mass_returns_batch_shapes_for_multiple_points():
    mu = 3.0
    stat = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ])

    p, acc = pot_point_mass(mu=mu, stat=stat)

    assert np.asarray(p).shape == (2,)
    assert np.asarray(acc).shape == (2, 3)


def test_pot_point_mass_raises_for_invalid_single_point_shape():
    mu = 10.0
    stat = [1.0, 2.0]   # invalid shape

    with pytest.raises(ValueError, match="stat must have shape"):
        pot_point_mass(mu=mu, stat=stat)


def test_pot_point_mass_raises_for_invalid_batch_shape():
    mu = 10.0
    stat = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])  # shape (2, 2), invalid

    with pytest.raises(ValueError, match="stat must have shape"):
        pot_point_mass(mu=mu, stat=stat)


def test_pot_point_mass_zero_mu():
    mu = 0.0
    stat = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ])

    p, acc = pot_point_mass(mu=mu, stat=stat)

    np.testing.assert_array_equal(np.asarray(p), np.array([0.0, 0.0]))
    np.testing.assert_array_equal(np.asarray(acc), np.zeros((2, 3)))


def test_pot_point_mass_at_origin_stays_finite_due_to_epsilon():
    mu = 10.0
    stat = [0.0, 0.0, 0.0]

    p, acc = pot_point_mass(mu=mu, stat=stat)

    # Because the function adds eps to avoid division by zero,
    # the result should be finite, not inf/nan.
    assert np.isfinite(np.asarray(p))
    assert np.all(np.isfinite(np.asarray(acc)))

    # At the exact origin, acceleration should be exactly zero because stat = 0 vector
    np.testing.assert_array_equal(np.asarray(acc), np.array([0.0, 0.0, 0.0]))