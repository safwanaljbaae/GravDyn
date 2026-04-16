from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import jax.numpy as jnp

mod = importlib.import_module("gravdyn.generate_layered_mascons")


def test_load_real_file():
    result = mod.load_tetrahedron_data(
        base_dir="Data",
        asteroid="Apophis",
        tetrahedron_data_file="layered_mascons.csv",
    )

    assert isinstance(result, dict)
    assert set(result.keys()) == {"x", "y", "z", "mu"}

    assert len(result["x"]) > 0
    assert len(result["x"]) == len(result["mu"])


def test_values_match_csv():
    data_file = Path("Data/Apophis/layered_mascons.csv")
    if not data_file.exists():
        pytest.skip(f"File not found: {data_file}")

    df = pd.read_csv(data_file)
    result = mod.load_tetrahedron_data(
        base_dir="Data",
        asteroid="Apophis",
        tetrahedron_data_file="layered_mascons.csv",
    )

    assert np.allclose(result["x"], df.iloc[:, 0].to_numpy())
    assert np.allclose(result["y"], df.iloc[:, 1].to_numpy())
    assert np.allclose(result["z"], df.iloc[:, 2].to_numpy())
    assert np.allclose(result["mu"], df.iloc[:, 7].to_numpy())


def test_output_is_jax_array():
    data_file = Path("Data/Apophis/layered_mascons.csv")
    if not data_file.exists():
        pytest.skip(f"File not found: {data_file}")

    result = mod.load_tetrahedron_data(
        base_dir="Data",
        asteroid="Apophis",
        tetrahedron_data_file="layered_mascons.csv",
    )

    for key in ["x", "y", "z", "mu"]:
        assert isinstance(result[key], jnp.ndarray)


def test_shapes_consistent():
    data_file = Path("Data/Apophis/layered_mascons.csv")
    if not data_file.exists():
        pytest.skip(f"File not found: {data_file}")

    result = mod.load_tetrahedron_data(
        base_dir="Data",
        asteroid="Apophis",
        tetrahedron_data_file="layered_mascons.csv",
    )

    n = len(result["x"])

    assert result["y"].shape[0] == n
    assert result["z"].shape[0] == n
    assert result["mu"].shape[0] == n


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        mod.load_tetrahedron_data(
            base_dir="Data",
            asteroid="Apophis",
            tetrahedron_data_file="non_existing_file.csv",
        )
