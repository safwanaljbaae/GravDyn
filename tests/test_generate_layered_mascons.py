from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import importlib

import gravdyn.constants
from gravdyn.constants import GRAVITATIONAL_CONSTANT as G

glm = importlib.import_module("gravdyn.generate_layered_mascons")


@pytest.fixture
def tetra_dir(tmp_path):
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    faces = np.array([
        [1, 2, 3],
        [1, 2, 4],
        [1, 3, 4],
        [2, 3, 4],
    ])
    asteroid_dir = tmp_path / "TestBody"
    asteroid_dir.mkdir()
    np.savetxt(asteroid_dir / "modified_v.dat", vertices, fmt="%.8e")
    np.savetxt(asteroid_dir / "modified_f.dat", faces, fmt="%d")
    return tmp_path


@pytest.fixture
def no_plots(monkeypatch):
    monkeypatch.setattr(glm, "plot_layers_by_density", lambda *a, **k: None)
    monkeypatch.setattr(glm, "plot_layer_intersections", lambda *a, **k: None)


class TestCoreFunctionality:
    def test_dataframe_output(self, tetra_dir, no_plots):
        df = glm.generate_layered_mascons(
            base_dir=str(tetra_dir),
            asteroid="TestBody",
            total_mass=100.0,
            densities=[1.0, 2.0, 3.0],
        )
        assert isinstance(df, pd.DataFrame)
        assert {"x", "y", "z", "mass", "layer_id", "mu"}.issubset(df.columns)

    def test_mass_conservation(self, tetra_dir, no_plots):
        total_mass = 5.31e10
        df = glm.generate_layered_mascons(
            base_dir=str(tetra_dir),
            asteroid="TestBody",
            total_mass=total_mass,
            densities=[1.8, 2.0, 2.3, 2.7],
        )
        assert df["mass"].sum() == pytest.approx(total_mass, rel=1e-10)

    def test_number_of_points(self, tetra_dir, no_plots):
        densities = [1, 2, 3, 4]
        df = glm.generate_layered_mascons(
            base_dir=str(tetra_dir),
            asteroid="TestBody",
            total_mass=100,
            densities=densities,
        )
        expected_max = 4 * len(densities)
        assert len(df) == expected_max

    def test_mu_relation(self, tetra_dir, no_plots):
        df = glm.generate_layered_mascons(
            base_dir=str(tetra_dir),
            asteroid="TestBody",
            total_mass=100,
            densities=[1, 1],
        )
        assert np.allclose(df["mu"], df["mass"] * G)


class TestEdgeCases:
    def test_negative_density(self, tetra_dir, no_plots):
        with pytest.raises(ValueError):
            glm.generate_layered_mascons(
                base_dir=str(tetra_dir),
                asteroid="TestBody",
                total_mass=100,
                densities=[1, -2],
            )

    def test_zero_total_mass(self, tetra_dir, no_plots):
        with pytest.raises(ValueError):
            glm.generate_layered_mascons(
                base_dir=str(tetra_dir),
                asteroid="TestBody",
                total_mass=0,
                densities=[1, 2],
            )

    def test_all_zero_densities(self, tetra_dir, no_plots):
        with pytest.raises(ValueError):
            glm.generate_layered_mascons(
                base_dir=str(tetra_dir),
                asteroid="TestBody",
                total_mass=100,
                densities=[0, 0, 0],
            )
