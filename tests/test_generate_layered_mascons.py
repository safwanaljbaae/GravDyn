from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import module correctly
glm = importlib.import_module("gravdyn.generate_layered_mascons")


# ---------------------------
# CONFIG: point to your real data
# ---------------------------
BASE_DIR = Path("../Data")   # adjust if needed
ASTEROID = "Apophis"


@pytest.fixture
def no_plots(monkeypatch):
    """
    Disable plotting during tests.
    """
    monkeypatch.setattr(glm, "plot_layers_by_density", lambda *a, **k: None)
    monkeypatch.setattr(glm, "plot_layer_intersections", lambda *a, **k: None)


@pytest.fixture
def check_files_exist():
    """
    Ensure test does not run if files are missing.
    """
    v_file = BASE_DIR / ASTEROID / "modified_v.dat"
    f_file = BASE_DIR / ASTEROID / "modified_f.dat"

    if not v_file.exists() or not f_file.exists():
        pytest.skip("Shape files not available for test")

    return True


# ---------------------------
# Core functionality
# ---------------------------
def test_dataframe_output(no_plots, check_files_exist):
    df = glm.generate_layered_mascons(
        base_dir=str(BASE_DIR),
        asteroid=ASTEROID,
        total_mass=100.0,
        densities=[1.0, 2.0, 3.0],
    )

    assert isinstance(df, pd.DataFrame)
    assert {"x", "y", "z", "mass", "layer_id", "mu"}.issubset(df.columns)


def test_mass_conservation(no_plots, check_files_exist):
    total_mass = 5.31e10

    df = glm.generate_layered_mascons(
        base_dir=str(BASE_DIR),
        asteroid=ASTEROID,
        total_mass=total_mass,
        densities=[1.8, 2.0, 2.3, 2.7],
    )

    assert np.isclose(df["mass"].sum(), total_mass, atol=1e-6)


def test_number_of_points(no_plots, check_files_exist):
    densities = [1, 2, 3, 4]

    df = glm.generate_layered_mascons(
        base_dir=str(BASE_DIR),
        asteroid=ASTEROID,
        total_mass=100,
        densities=densities,
    )

    # number of points = faces × layers (minus zero-density layers)
    n_faces = len(np.loadtxt(BASE_DIR / ASTEROID / "modified_f.dat", dtype=int))
    assert len(df) <= n_faces * len(densities)


def test_mu_relation(no_plots, check_files_exist):
    G = 6.674101262875753845e-20

    df = glm.generate_layered_mascons(
        base_dir=str(BASE_DIR),
        asteroid=ASTEROID,
        total_mass=100,
        densities=[1, 1],
    )

    assert np.allclose(df["mu"], df["mass"] * G)


# ---------------------------
# Edge cases
# ---------------------------
def test_negative_density(no_plots, check_files_exist):
    with pytest.raises(ValueError):
        glm.generate_layered_mascons(
            base_dir=str(BASE_DIR),
            asteroid=ASTEROID,
            total_mass=100,
            densities=[1, -2],
        )


def test_zero_total_mass(no_plots, check_files_exist):
    with pytest.raises(ValueError):
        glm.generate_layered_mascons(
            base_dir=str(BASE_DIR),
            asteroid=ASTEROID,
            total_mass=0,
            densities=[1, 2],
        )


def test_all_zero_densities(no_plots, check_files_exist):
    with pytest.raises(ValueError):
        glm.generate_layered_mascons(
            base_dir=str(BASE_DIR),
            asteroid=ASTEROID,
            total_mass=100,
            densities=[0, 0, 0],
        )