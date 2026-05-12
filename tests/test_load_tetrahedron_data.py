from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import jax.numpy as jnp

from gravdyn import load_tetrahedron_data


@pytest.fixture
def csv_file(tmp_path):
    df = pd.DataFrame({
        "x": [0.1, 0.2, 0.3],
        "y": [0.0, 0.1, 0.2],
        "z": [0.0, 0.0, 0.1],
        "mass": [1.0, 2.0, 3.0],
        "face_id": [1, 1, 2],
        "layer_id": [1, 2, 1],
        "density_input": [1.5, 1.8, 1.5],
        "mu": [6.674e-11, 1.335e-10, 2.002e-10],
    })
    asteroid_dir = tmp_path / "TestBody"
    asteroid_dir.mkdir()
    path = asteroid_dir / "mascons.csv"
    df.to_csv(path, index=False)
    return tmp_path, path.name


class TestLoadTetrahedronData:
    def test_returns_dict_with_expected_keys(self, csv_file):
        base_dir, filename = csv_file[0], csv_file[1]
        result = load_tetrahedron_data(
            base_dir=str(base_dir),
            asteroid="TestBody",
            tetrahedron_data_file=filename,
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {"x", "y", "z", "mu"}

    def test_values_match_csv(self, csv_file):
        base_dir, filename = csv_file
        full_path = base_dir / "TestBody" / filename
        df = pd.read_csv(full_path)
        result = load_tetrahedron_data(
            base_dir=str(base_dir),
            asteroid="TestBody",
            tetrahedron_data_file=filename,
        )
        np.testing.assert_allclose(result["x"], df["x"].to_numpy())
        np.testing.assert_allclose(result["y"], df["y"].to_numpy())
        np.testing.assert_allclose(result["z"], df["z"].to_numpy())
        np.testing.assert_allclose(result["mu"], df["mu"].to_numpy())

    def test_output_is_jax_array(self, csv_file):
        base_dir, filename = csv_file
        result = load_tetrahedron_data(
            base_dir=str(base_dir),
            asteroid="TestBody",
            tetrahedron_data_file=filename,
        )
        for key in ["x", "y", "z", "mu"]:
            assert isinstance(result[key], jnp.ndarray)

    def test_shapes_consistent(self, csv_file):
        base_dir, filename = csv_file
        result = load_tetrahedron_data(
            base_dir=str(base_dir),
            asteroid="TestBody",
            tetrahedron_data_file=filename,
        )
        n = len(result["x"])
        assert result["y"].shape[0] == n
        assert result["z"].shape[0] == n
        assert result["mu"].shape[0] == n

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_tetrahedron_data(
                base_dir=str(tmp_path),
                asteroid="TestBody",
                tetrahedron_data_file="non_existing_file.csv",
            )
