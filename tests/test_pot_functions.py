from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp
from pathlib import Path

from gravdyn.pot_functions import (
    pot_expansion,
    pot_mascon_jax,
    batched_pot_mascon,
    compute_pseudo_potential,
    save_potential_to_file,
    format_time,
)


class TestPotExpansion:
    @pytest.fixture
    def f_pot(self):
        def _pot(x, y, z):
            r = jnp.sqrt(x**2 + y**2 + z**2)
            return -1.0 / r
        return _pot

    @pytest.fixture
    def f_dpot(self):
        def _dpot_x(x, y, z):
            r = jnp.sqrt(x**2 + y**2 + z**2)
            return -x / r**3
        return _dpot_x

    @pytest.fixture
    def pot_funcs(self, f_pot, f_dpot):
        return f_pot, [f_dpot, lambda x, y, z: -y / jnp.sqrt(x**2 + y**2 + z**2)**3,
                       lambda x, y, z: -z / jnp.sqrt(x**2 + y**2 + z**2)**3]

    def test_single_point(self, pot_funcs):
        f_pot, f_dpot = pot_funcs
        p, acc = pot_expansion(stat=[2.0, 0.0, 0.0], f_pot_expansion=f_pot, f_d_pot_expansion=f_dpot)
        assert np.ndim(p) == 0
        assert acc.shape == (3,)
        assert p == pytest.approx(-0.5)
        assert acc[0] == pytest.approx(-2.0 / 8.0)

    def test_batch_points(self, pot_funcs):
        f_pot, f_dpot = pot_funcs
        stat = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        p, acc = pot_expansion(stat=stat, f_pot_expansion=f_pot, f_d_pot_expansion=f_dpot)
        assert p.shape == (2,)
        assert acc.shape == (2, 3)
        assert p[0] == pytest.approx(-0.5)

    def test_list_input(self, pot_funcs):
        f_pot, f_dpot = pot_funcs
        p, acc = pot_expansion(stat=[2.0, 0.0, 0.0], f_pot_expansion=f_pot, f_d_pot_expansion=f_dpot)
        assert np.isfinite(p)
        assert np.all(np.isfinite(acc))

    def test_invalid_shape_raises(self, pot_funcs):
        f_pot, f_dpot = pot_funcs
        with pytest.raises(ValueError, match="stat must have shape"):
            pot_expansion(stat=[1.0, 2.0], f_pot_expansion=f_pot, f_d_pot_expansion=f_dpot)

    def test_3d_array_raises(self, pot_funcs):
        f_pot, f_dpot = pot_funcs
        with pytest.raises(ValueError, match="stat must have shape"):
            pot_expansion(stat=np.array([[[1.0, 2.0, 3.0]]]), f_pot_expansion=f_pot, f_d_pot_expansion=f_dpot)


class TestPotMasconJax:
    @pytest.fixture
    def data_shape(self):
        np.random.seed(42)
        n = 10
        return {
            "x": jnp.array(np.random.randn(n), dtype=jnp.float64),
            "y": jnp.array(np.random.randn(n), dtype=jnp.float64),
            "z": jnp.array(np.random.randn(n), dtype=jnp.float64),
            "mu": jnp.array(np.abs(np.random.randn(n)), dtype=jnp.float64),
        }

    def test_single_point(self, data_shape):
        p, a = pot_mascon_jax(stat=[1.0, 0.0, 0.0], data_shape=data_shape)
        assert np.ndim(p) == 0
        assert a.shape == (3,)
        assert np.isfinite(p)
        assert np.all(np.isfinite(a))

    def test_batch_points(self, data_shape):
        stat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        p, a = pot_mascon_jax(stat=stat, data_shape=data_shape)
        assert p.shape == (3,)
        assert a.shape == (3, 3)
        assert np.all(np.isfinite(p))
        assert np.all(np.isfinite(a))

    def test_invalid_shape_raises(self, data_shape):
        with pytest.raises(ValueError, match="stat must have shape"):
            pot_mascon_jax(stat=[1.0, 2.0], data_shape=data_shape)

    def test_shape_consistency(self, data_shape):
        stat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        p, a = pot_mascon_jax(stat=stat, data_shape=data_shape)
        assert p.shape == (2,)
        assert a.shape == (2, 3)

    def test_gm_body_shifts_potential(self, data_shape):
        stat = jnp.array([1e6, 0.0, 0.0])
        p0, a0 = pot_mascon_jax(stat=stat, data_shape=data_shape, gm_body=0.0)
        p1, a1 = pot_mascon_jax(stat=stat, data_shape=data_shape, gm_body=1.0)
        assert p1 > p0
        assert not np.array_equal(a0, a1)


class TestBatchedPotMascon:
    @pytest.fixture
    def data_shape(self):
        np.random.seed(42)
        n = 10
        return {
            "x": np.random.randn(n),
            "y": np.random.randn(n),
            "z": np.random.randn(n),
            "mu": np.abs(np.random.randn(n)),
        }

    def test_returns_shapes(self, data_shape):
        stat = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        p, a = batched_pot_mascon(stat=stat, data_shape=data_shape, batch_size=2)
        assert p.shape == (3,)
        assert a.shape == (3, 3)
        assert np.all(np.isfinite(p))
        assert np.all(np.isfinite(a))

    def test_single_point_delegates(self, data_shape):
        p, a = batched_pot_mascon(stat=[1.0, 0.0, 0.0], data_shape=data_shape)
        assert np.ndim(p) == 0
        assert a.shape == (3,)

    def test_large_batch(self, data_shape):
        stat = np.random.randn(50, 3)
        p, a = batched_pot_mascon(stat=stat, data_shape=data_shape, batch_size=20)
        assert p.shape == (50,)
        assert a.shape == (50, 3)
        assert np.all(np.isfinite(p))
        assert np.all(np.isfinite(a))


class TestComputePseudoPotential:
    def test_basic(self):
        stat = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        pot = np.array([1.0, 2.0])
        result = compute_pseudo_potential(stat, pot, rot_period_hours=24.0)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_zero_rotation(self):
        stat = np.array([[1.0, 0.0, 0.0]])
        pot = np.array([5.0])
        with pytest.raises(ZeroDivisionError):
            compute_pseudo_potential(stat, pot, rot_period_hours=0.0)

    def test_no_centrifugal_on_z_axis(self):
        stat = np.array([[0.0, 0.0, 1.0]])
        pot = np.array([5.0])
        result = compute_pseudo_potential(stat, pot, rot_period_hours=24.0)
        assert result == pytest.approx(-5.0)

    def test_values(self):
        stat = np.array([[1.0, 0.0, 0.0]])
        pot = np.array([10.0])
        omega = 2.0 * np.pi / (24 * 3600)
        fat = omega**2 * 1.0 / 2.0
        expected = -fat - 10.0
        result = compute_pseudo_potential(stat, pot, rot_period_hours=24.0)
        assert result == pytest.approx(expected)


class TestSavePotentialToFile:
    def test_saves_npz(self, tmp_path):
        stat = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        pot = np.array([1.0, 2.0])
        path = str(tmp_path / "pot.npz")
        save_potential_to_file(stat, pot, pseudo_pot=None, output_path=path)
        data = np.load(path)
        assert "x" in data and "y" in data and "z" in data
        assert "potential" in data
        assert "pseudo_potential" not in data

    def test_with_pseudo_potential(self, tmp_path):
        stat = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        pot = np.array([1.0, 2.0])
        pseudo = np.array([0.5, 1.0])
        path = str(tmp_path / "pot.npz")
        save_potential_to_file(stat, pot, pseudo_pot=pseudo, output_path=path)
        data = np.load(path)
        assert "pseudo_potential" in data
        np.testing.assert_allclose(data["pseudo_potential"], pseudo)

    def test_values_match(self, tmp_path):
        stat = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        pot = np.array([1.0, 2.0])
        path = str(tmp_path / "pot.npz")
        save_potential_to_file(stat, pot, pseudo_pot=None, output_path=path)
        data = np.load(path)
        np.testing.assert_allclose(data["x"], stat[:, 0])
        np.testing.assert_allclose(data["y"], stat[:, 1])
        np.testing.assert_allclose(data["z"], stat[:, 2])
        np.testing.assert_allclose(data["potential"], pot)


class TestFormatTime:
    def test_zero(self):
        assert format_time(0) == "00:00:00.000"

    def test_seconds_only(self):
        assert format_time(45.5) == "00:00:45.500"

    def test_minutes(self):
        assert format_time(125.0) == "00:02:05.000"

    def test_hours(self):
        assert format_time(3723.0) == "01:02:03.000"

    def test_milliseconds(self):
        result = format_time(1.123456)
        assert result.startswith("00:00:01.123")

    def test_large_value(self):
        result = format_time(100000.0)
        assert ":" in result
