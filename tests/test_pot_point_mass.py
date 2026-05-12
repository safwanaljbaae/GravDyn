# tests/test_pot_point_mass.py
from __future__ import annotations

import numpy as np
import pytest

from gravdyn import pot_point_mass


class TestShapeValidation:
    def test_invalid_single_point_shape(self):
        with pytest.raises(ValueError, match="stat must have shape"):
            pot_point_mass(mu=10.0, stat=[1.0, 2.0])

    def test_invalid_batch_shape(self):
        stat = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="stat must have shape"):
            pot_point_mass(mu=10.0, stat=stat)

    def test_3d_array(self):
        stat = np.array([[[1.0, 2.0, 3.0]]])
        with pytest.raises(ValueError, match="stat must have shape"):
            pot_point_mass(mu=10.0, stat=stat)


class TestSinglePoint:
    def test_on_x_axis(self):
        mu = 10.0
        p, acc = pot_point_mass(mu=mu, stat=[2.0, 0.0, 0.0])
        assert p == pytest.approx(-mu / 2.0)
        assert acc == pytest.approx(np.array([-mu / 4.0, 0.0, 0.0]))

    def test_on_y_axis(self):
        mu = 10.0
        p, acc = pot_point_mass(mu=mu, stat=[0.0, 4.0, 0.0])
        assert p == pytest.approx(-mu / 4.0)
        assert acc == pytest.approx(np.array([0.0, -mu / 16.0, 0.0]))

    def test_on_z_axis(self):
        mu = 10.0
        p, acc = pot_point_mass(mu=mu, stat=[0.0, 0.0, 5.0])
        assert p == pytest.approx(-mu / 5.0)
        assert acc == pytest.approx(np.array([0.0, 0.0, -mu / 25.0]))

    def test_list_input(self):
        p, acc = pot_point_mass(mu=3.0, stat=[1.0, 2.0, 2.0])
        r = np.linalg.norm([1.0, 2.0, 2.0])
        assert p == pytest.approx(-3.0 / r)
        assert acc == pytest.approx(-3.0 * np.array([1.0, 2.0, 2.0]) / r**3)

    def test_array_input(self):
        p, acc = pot_point_mass(mu=3.0, stat=np.array([1.0, 2.0, 2.0]))
        r = np.linalg.norm([1.0, 2.0, 2.0])
        assert p == pytest.approx(-3.0 / r)
        assert acc == pytest.approx(-3.0 * np.array([1.0, 2.0, 2.0]) / r**3)

    def test_negative_coordinates(self):
        p, acc = pot_point_mass(mu=5.0, stat=[-3.0, 0.0, 0.0])
        assert p == pytest.approx(-5.0 / 3.0)
        assert acc == pytest.approx(np.array([5.0 / 9.0, 0.0, 0.0]))

    def test_return_shapes(self):
        p, acc = pot_point_mass(mu=3.0, stat=np.array([1.0, 0.0, 0.0]))
        assert np.ndim(p) == 0
        assert acc.shape == (3,)


class TestBatchPoints:
    def test_multiple_points(self):
        mu = 10.0
        stat = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 5.0],
        ])
        p, acc = pot_point_mass(mu=mu, stat=stat)

        expected_p = np.array([-10.0 / 2.0, -10.0 / 4.0, -10.0 / 5.0])
        expected_acc = np.array([
            [-10.0 / 4.0, 0.0, 0.0],
            [0.0, -10.0 / 16.0, 0.0],
            [0.0, 0.0, -10.0 / 25.0],
        ])
        np.testing.assert_allclose(p, expected_p, rtol=1e-12)
        np.testing.assert_allclose(acc, expected_acc, rtol=1e-12)

    def test_single_point_in_batch(self):
        stat = np.array([[2.0, 0.0, 0.0]])
        p, acc = pot_point_mass(mu=10.0, stat=stat)
        assert p.shape == (1,)
        assert acc.shape == (1, 3)

    def test_list_of_lists(self):
        stat = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
        p, acc = pot_point_mass(mu=6.0, stat=stat)
        expected_p = np.array([-6.0 / 2.0, -6.0 / 3.0])
        np.testing.assert_allclose(p, expected_p, rtol=1e-12)

    def test_return_shapes(self):
        stat = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        p, acc = pot_point_mass(mu=3.0, stat=stat)
        assert p.shape == (2,)
        assert acc.shape == (2, 3)

    def test_large_distances(self):
        stat = np.array([[1e6, 0.0, 0.0], [0.0, 1e6, 0.0]])
        mu = 1.0
        p, acc = pot_point_mass(mu=mu, stat=stat)
        np.testing.assert_allclose(p, [-1e-6, -1e-6], rtol=1e-6)
        np.testing.assert_allclose(acc, [[-1e-12, 0, 0], [0, -1e-12, 0]], rtol=1e-6)


class TestEdgeCases:
    def test_zero_mu(self):
        stat = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        p, acc = pot_point_mass(mu=0.0, stat=stat)
        np.testing.assert_array_equal(p, np.array([0.0, 0.0]))
        np.testing.assert_array_equal(acc, np.zeros((2, 3)))

    def test_zero_mu_single_point(self):
        p, acc = pot_point_mass(mu=0.0, stat=[1.0, 0.0, 0.0])
        assert p == pytest.approx(0.0)
        assert np.all(acc == pytest.approx(0.0))

    def test_at_origin_stays_finite(self):
        mu = 10.0
        p, acc = pot_point_mass(mu=mu, stat=[0.0, 0.0, 0.0])
        assert np.isfinite(p)
        assert np.all(np.isfinite(acc))
        np.testing.assert_array_equal(acc, np.array([0.0, 0.0, 0.0]))
