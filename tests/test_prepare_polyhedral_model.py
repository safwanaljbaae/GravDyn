# pytest -v
# tests/test_prepare_polyhedral_model.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib
import numpy as np
import pytest

from gravdyn import prepare_werner_model

MODULE = importlib.import_module("gravdyn.prepare_polyhedral_model")


@dataclass
class FakePolyFiles:
    base_dir: str
    asteroid: str

    def __post_init__(self):
        root = f"{self.base_dir}/{self.asteroid}"
        self.root = root
        self.file_vertices = f"{root}/shape_v.dat"
        self.file_faces = f"{root}/shape_f.dat"
        self.file_edges = f"{root}/edges.dat"
        self.file_centroid_edges = f"{root}/centroid_edges.dat"
        self.file_centroid_faces = f"{root}/centroid_faces.dat"
        self.file_e_e = f"{root}/e_e.dat"
        self.file_n_f = f"{root}/n_f.dat"
        self.file_n_f_e = f"{root}/n_f_e.dat"
        self.file_n_fp_e = f"{root}/n_fp_e.dat"
        self.file_r_e_1 = f"{root}/r_e_1.dat"
        self.file_r_e_2 = f"{root}/r_e_2.dat"
        self.file_r_f_1 = f"{root}/r_f_1.dat"
        self.file_r_f_2 = f"{root}/r_f_2.dat"
        self.file_r_f_3 = f"{root}/r_f_3.dat"


@pytest.fixture
def sample_geometry():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    faces = np.array([[0, 1, 2]], dtype=np.int64)

    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 0],
        ],
        dtype=np.int64,
    )

    c_faces = np.array([[1.0 / 3.0, 1.0 / 3.0, 0.0]], dtype=float)

    c_edges = np.array(
        [
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=float,
    )

    poly_vectors = {
        "e_e": np.ones((3, 3), dtype=float),
        "n_f": np.ones((1, 3), dtype=float) * 2.0,
        "n_f_e": np.ones((3, 3), dtype=float) * 3.0,
        "n_fp_e": np.ones((3, 3), dtype=float) * 4.0,
        "r_e_1": np.ones((3, 3), dtype=float) * 5.0,
        "r_e_2": np.ones((3, 3), dtype=float) * 6.0,
        "r_f_1": np.ones((1, 3), dtype=float) * 7.0,
        "r_f_2": np.ones((1, 3), dtype=float) * 8.0,
        "r_f_3": np.ones((1, 3), dtype=float) * 9.0,
    }

    return vertices, faces, edges, c_faces, c_edges, poly_vectors


def test_prepare_polyhedral_model_raises_if_vertices_or_faces_missing(monkeypatch):
    monkeypatch.setattr(MODULE, "PolyFiles", FakePolyFiles)
    monkeypatch.setattr(MODULE.os.path, "exists", lambda path: False)

    with pytest.raises(FileNotFoundError, match="Shape files"):
        prepare_werner_model(
            asteroid="Apophis",
            base_dir="Data",
            verbose=False,
        )


def test_prepare_polyhedral_model_computes_missing_products(monkeypatch, sample_geometry):
    vertices, faces, edges, c_faces, c_edges, poly_vectors = sample_geometry

    monkeypatch.setattr(MODULE, "PolyFiles", FakePolyFiles)

    # Only essential files exist; derived products are missing
    def fake_exists(path: str) -> bool:
        return path.endswith("shape_v.dat") or path.endswith("shape_f.dat")

    monkeypatch.setattr(MODULE.os.path, "exists", fake_exists)
    monkeypatch.setattr(MODULE, "load_vertices_faces", lambda vfile, ffile: (vertices, faces))

    calls = {"edges": False, "centroids": False, "vectors": False}

    def fake_create_edges_from_facets(faces_in, files, verbose):
        calls["edges"] = True
        np.testing.assert_array_equal(faces_in, faces)
        return edges

    def fake_compute_polyhedron_centroids(vertices_in, faces_in, edges_in, files, verbose):
        calls["centroids"] = True
        np.testing.assert_array_equal(vertices_in, vertices)
        np.testing.assert_array_equal(faces_in, faces)
        np.testing.assert_array_equal(edges_in, edges)
        return c_faces, c_edges

    def fake_compute_polyhedron_vectors(vertices_in, faces_in, edges_in, files, verbose):
        calls["vectors"] = True
        np.testing.assert_array_equal(vertices_in, vertices)
        np.testing.assert_array_equal(faces_in, faces)
        np.testing.assert_array_equal(edges_in, edges)
        return poly_vectors

    monkeypatch.setattr(MODULE, "create_edges_from_facets", fake_create_edges_from_facets)
    monkeypatch.setattr(MODULE, "compute_polyhedron_centroids", fake_compute_polyhedron_centroids)
    monkeypatch.setattr(MODULE, "compute_polyhedron_vectors", fake_compute_polyhedron_vectors)

    result = prepare_werner_model(
        asteroid="Apophis",
        base_dir="Data",
        verbose=False,
    )

    assert calls["edges"] is True
    assert calls["centroids"] is True
    assert calls["vectors"] is True

    np.testing.assert_array_equal(result["vertices"], vertices)
    np.testing.assert_array_equal(result["faces"], faces)
    np.testing.assert_array_equal(result["edges"], edges)
    np.testing.assert_array_equal(result["centroid_faces"], c_faces)
    np.testing.assert_array_equal(result["centroid_edges"], c_edges)

    for key, value in poly_vectors.items():
        np.testing.assert_array_equal(result[key], value)

    assert isinstance(result["files"], dict)
    assert result["files"]["asteroid"] == "Apophis"
    assert result["files"]["base_dir"] == "Data"


def test_prepare_polyhedral_model_loads_existing_products(monkeypatch, sample_geometry):
    vertices, faces, edges, c_faces, c_edges, poly_vectors = sample_geometry

    monkeypatch.setattr(MODULE, "PolyFiles", FakePolyFiles)
    monkeypatch.setattr(MODULE.os.path, "exists", lambda path: True)
    monkeypatch.setattr(MODULE, "load_vertices_faces", lambda vfile, ffile: (vertices, faces))

    def should_not_be_called(*args, **kwargs):
        raise AssertionError("Compute function should not be called when files already exist")

    monkeypatch.setattr(MODULE, "create_edges_from_facets", should_not_be_called)
    monkeypatch.setattr(MODULE, "compute_polyhedron_centroids", should_not_be_called)
    monkeypatch.setattr(MODULE, "compute_polyhedron_vectors", should_not_be_called)

    def fake_load_dat(path: str):
        name = Path(path).name

        load_map = {
            "centroid_faces.dat": c_faces,
            "centroid_edges.dat": c_edges,
            "edges.dat": edges,
            "e_e.dat": poly_vectors["e_e"],
            "n_f.dat": poly_vectors["n_f"],
            "n_f_e.dat": poly_vectors["n_f_e"],
            "n_fp_e.dat": poly_vectors["n_fp_e"],
            "r_e_1.dat": poly_vectors["r_e_1"],
            "r_e_2.dat": poly_vectors["r_e_2"],
            "r_f_1.dat": poly_vectors["r_f_1"],
            "r_f_2.dat": poly_vectors["r_f_2"],
            "r_f_3.dat": poly_vectors["r_f_3"],
        }

        if name in load_map:
            return load_map[name]

        raise AssertionError(f"Unexpected file requested: {path}")

    monkeypatch.setattr(MODULE, "_load_dat", fake_load_dat)

    result = prepare_werner_model(
        asteroid="Apophis",
        base_dir="Data",
        verbose=False,
    )

    np.testing.assert_array_equal(result["vertices"], vertices)
    np.testing.assert_array_equal(result["faces"], faces)
    np.testing.assert_array_equal(result["edges"], edges)
    np.testing.assert_array_equal(result["centroid_faces"], c_faces)
    np.testing.assert_array_equal(result["centroid_edges"], c_edges)

    for key, value in poly_vectors.items():
        np.testing.assert_array_equal(result[key], value)

    assert isinstance(result["files"], dict)
    assert result["files"]["asteroid"] == "Apophis"
    assert result["files"]["base_dir"] == "Data"