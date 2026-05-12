from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import importlib
import numpy as np
import pytest

from gravdyn import prepare_werner_model

MODULE = importlib.import_module("gravdyn.prepare_polyhedral_model")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakePolyFiles:
    """Mimics ``gravdyn.polyhedral_model.poly_files.PolyFiles`` but keeps
    paths as plain attributes (not ``@property``) for monkeypatch compatibility."""

    base_dir: str
    asteroid: str

    def __post_init__(self):
        import os
        self.root = os.path.join(self.base_dir, self.asteroid)
        self.file_vertices = os.path.join(self.root, "modified_v.dat")
        self.file_faces = os.path.join(self.root, "modified_f.dat")
        self.file_edges = os.path.join(self.root, "edges.dat")
        self.file_centroid_edges = os.path.join(self.root, "centroid_edges.dat")
        self.file_centroid_faces = os.path.join(self.root, "centroid_faces.dat")
        self.file_e_e = os.path.join(self.root, "e_e.dat")
        self.file_n_f = os.path.join(self.root, "n_f.dat")
        self.file_n_f_e = os.path.join(self.root, "n_f_e.dat")
        self.file_n_fp_e = os.path.join(self.root, "n_fp_e.dat")
        self.file_r_e_1 = os.path.join(self.root, "r_e_1.dat")
        self.file_r_e_2 = os.path.join(self.root, "r_e_2.dat")
        self.file_r_f_1 = os.path.join(self.root, "r_f_1.dat")
        self.file_r_f_2 = os.path.join(self.root, "r_f_2.dat")
        self.file_r_f_3 = os.path.join(self.root, "r_f_3.dat")


@pytest.fixture
def tetrahedron_geometry():
    """Return data for a non-degenerate tetrahedron with 4 vertices, 4 faces, 6 edges.

    The tetrahedron vertices form a valid 3D volume:
      v0 = (0,0,0),  v1 = (1,0,0),  v2 = (0,1,0),  v3 = (0,0,1)

    Faces are stored 0-based (as ``load_vertices_faces`` returns them after
    1→0 conversion).  Edges are stored in the 4-column format ``[v1, v2, f1, f2]``
    produced by ``create_edges_from_facets`` (1-based vertices *and* 1-based face IDs).
    """
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    # 0-based (after load_vertices_faces converts from 1-based)
    faces = np.array([
        [1, 3, 2],
        [0, 2, 3],
        [0, 3, 1],
        [0, 1, 2],
    ], dtype=np.int64)

    # 1-based vertices + 1-based face IDs (as create_edges_from_facets produces)
    edges = np.array([
        [1, 3, 1, 2],
        [2, 4, 1, 4],
        [1, 2, 1, 3],
        [1, 4, 2, 3],
        [3, 4, 2, 4],
        [2, 3, 3, 4],
    ], dtype=np.int64)

    c_faces = np.array([
        [1.0/3.0, 1.0/3.0, 1.0/3.0],
        [0.0,      1.0/3.0, 1.0/3.0],
        [0.0,      1.0/3.0, 1.0/3.0],
        [1.0/3.0,  1.0/3.0, 0.0],
    ], dtype=float)

    c_edges = np.array([
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0],
    ], dtype=float)

    poly_vectors = {
        "e_e":    np.ones(6, dtype=float),
        "n_f":    np.ones((4, 3), dtype=float) * 2.0,
        "n_f_e":  np.ones((6, 3), dtype=float) * 3.0,
        "n_fp_e": np.ones((6, 3), dtype=float) * 4.0,
        "r_e_1":  np.ones((6, 3), dtype=float) * 5.0,
        "r_e_2":  np.ones((6, 3), dtype=float) * 6.0,
        "r_f_1":  np.ones((4, 3), dtype=float) * 7.0,
        "r_f_2":  np.ones((4, 3), dtype=float) * 8.0,
        "r_f_3":  np.ones((4, 3), dtype=float) * 9.0,
    }

    return vertices, faces, edges, c_faces, c_edges, poly_vectors


def _patch_polyfiles(monkeypatch):
    """Replace ``PolyFiles`` inside ``prepare_werner_model`` with ``FakePolyFiles``."""
    monkeypatch.setattr(MODULE, "PolyFiles", FakePolyFiles)


# ---------------------------------------------------------------------------
# Orchestration tests  (all compute functions are monkeypatched)
# ---------------------------------------------------------------------------

class TestFileExistence:
    """Tests that ``prepare_werner_model`` correctly detects missing files."""

    def test_raises_when_vertices_and_faces_missing(self, monkeypatch):
        _patch_polyfiles(monkeypatch)
        monkeypatch.setattr(MODULE.os.path, "exists", lambda p: False)

        with pytest.raises(FileNotFoundError, match="Shape files"):
            prepare_werner_model(asteroid="Apophis", base_dir="Data", verbose=False)

    def test_raises_when_only_vertices_missing(self, monkeypatch):
        _patch_polyfiles(monkeypatch)

        def fake_exists(path: str) -> bool:
            return "modified_f.dat" in path

        monkeypatch.setattr(MODULE.os.path, "exists", fake_exists)

        with pytest.raises(FileNotFoundError, match="Shape files"):
            prepare_werner_model(asteroid="Apophis", base_dir="Data", verbose=False)

    def test_raises_when_only_faces_missing(self, monkeypatch):
        _patch_polyfiles(monkeypatch)

        def fake_exists(path: str) -> bool:
            return "modified_v.dat" in path

        monkeypatch.setattr(MODULE.os.path, "exists", fake_exists)

        with pytest.raises(FileNotFoundError, match="Shape files"):
            prepare_werner_model(asteroid="Apophis", base_dir="Data", verbose=False)


class TestComputationPath:
    """When derived files are absent the function computes them."""

    def test_computes_all_derived_products(self, monkeypatch, tetrahedron_geometry):
        vertices, faces, edges, c_faces, c_edges, poly_vectors = tetrahedron_geometry
        _patch_polyfiles(monkeypatch)

        def fake_exists(path: str) -> bool:
            return "modified_v.dat" in path or "modified_f.dat" in path

        monkeypatch.setattr(MODULE.os.path, "exists", fake_exists)
        monkeypatch.setattr(MODULE, "load_vertices_faces",
                            lambda v, f: (vertices, faces))

        calls = {"edges": 0, "centroids": 0, "vectors": 0}

        def fake_create_edges(faces_in, files):
            calls["edges"] += 1
            np.testing.assert_array_equal(faces_in, faces)
            return edges

        def fake_centroids(v_in, f_in, e_in, files, verbose):
            calls["centroids"] += 1
            np.testing.assert_array_equal(v_in, vertices)
            np.testing.assert_array_equal(f_in, faces)
            np.testing.assert_array_equal(e_in, edges)
            return c_faces, c_edges

        def fake_vectors(v_in, f_in, e_in, files, verbose):
            calls["vectors"] += 1
            np.testing.assert_array_equal(v_in, vertices)
            np.testing.assert_array_equal(f_in, faces)
            np.testing.assert_array_equal(e_in, edges)
            return poly_vectors

        monkeypatch.setattr(MODULE, "create_edges_from_facets", fake_create_edges)
        monkeypatch.setattr(MODULE, "compute_polyhedron_centroids", fake_centroids)
        monkeypatch.setattr(MODULE, "compute_polyhedron_vectors", fake_vectors)

        result = prepare_werner_model(asteroid="Apophis", base_dir="Data", verbose=False)

        assert calls["edges"] == 1
        assert calls["centroids"] == 1
        assert calls["vectors"] == 1

        np.testing.assert_array_equal(result["vertices"], vertices)
        np.testing.assert_array_equal(result["faces"], faces)
        np.testing.assert_array_equal(result["edges"], edges)
        np.testing.assert_array_equal(result["centroid_faces"], c_faces)
        np.testing.assert_array_equal(result["centroid_edges"], c_edges)
        for k, v in poly_vectors.items():
            np.testing.assert_array_equal(result[k], v)

        assert isinstance(result["files"], dict)
        assert result["files"]["asteroid"] == "Apophis"
        assert result["files"]["base_dir"] == "Data"


class TestLoadCachePath:
    """When derived files already exist the function loads them from disk."""

    def test_loads_existing_products(self, monkeypatch, tetrahedron_geometry):
        vertices, faces, edges, c_faces, c_edges, poly_vectors = tetrahedron_geometry
        _patch_polyfiles(monkeypatch)

        monkeypatch.setattr(MODULE.os.path, "exists", lambda p: True)
        monkeypatch.setattr(MODULE, "load_vertices_faces",
                            lambda v, f: (vertices, faces))

        def should_not_be_called(*args, **kwargs):
            raise AssertionError("Compute function should not be called when files exist")

        monkeypatch.setattr(MODULE, "create_edges_from_facets", should_not_be_called)
        monkeypatch.setattr(MODULE, "compute_polyhedron_centroids", should_not_be_called)
        monkeypatch.setattr(MODULE, "compute_polyhedron_vectors", should_not_be_called)

        load_map = {
            "edges.dat": edges,
            "centroid_faces.dat": c_faces,
            "centroid_edges.dat": c_edges,
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

        def fake_load_dat(path: str):
            name = Path(path).name
            if name in load_map:
                return load_map[name]
            raise AssertionError(f"Unexpected file requested: {path}")

        monkeypatch.setattr(MODULE, "_load_dat", fake_load_dat)

        result = prepare_werner_model(asteroid="Apophis", base_dir="Data", verbose=False)

        np.testing.assert_array_equal(result["vertices"], vertices)
        np.testing.assert_array_equal(result["faces"], faces)
        np.testing.assert_array_equal(result["edges"], edges)
        np.testing.assert_array_equal(result["centroid_faces"], c_faces)
        np.testing.assert_array_equal(result["centroid_edges"], c_edges)
        for k, v in poly_vectors.items():
            np.testing.assert_array_equal(result[k], v)

        assert isinstance(result["files"], dict)
        assert result["files"]["asteroid"] == "Apophis"
        assert result["files"]["base_dir"] == "Data"


# ---------------------------------------------------------------------------
# Integration test (runs real compute functions on real files)
# ---------------------------------------------------------------------------

class TestIntegration:
    """Exercises the full ``prepare_werner_model`` with real file I/O and real
    compute functions on a tiny tetrahedron."""

    def _write_shape_files(self, tmp_path, asteroid, vertices, faces_1based):
        """Write ``modified_v.dat`` and ``modified_f.dat`` with 1-based faces
        so that ``load_vertices_faces`` correctly converts to 0-based."""
        d = Path(tmp_path) / asteroid
        d.mkdir(parents=True, exist_ok=True)
        np.savetxt(d / "modified_v.dat", vertices, fmt="%.6f")
        np.savetxt(d / "modified_f.dat", faces_1based, fmt="%d")

    def test_real_pipeline_on_tetrahedron(self, tmp_path):
        asteroid = "TetraTest"
        base_dir = str(tmp_path)

        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        faces_1based = np.array([
            [2, 4, 3],
            [1, 3, 4],
            [1, 4, 2],
            [1, 2, 3],
        ], dtype=np.int64)

        self._write_shape_files(tmp_path, asteroid, vertices, faces_1based)
        result = prepare_werner_model(asteroid=asteroid, base_dir=base_dir, verbose=False)

        expected_keys = [
            "vertices", "faces", "edges",
            "centroid_faces", "centroid_edges",
            "e_e", "n_f", "n_f_e", "n_fp_e",
            "r_e_1", "r_e_2", "r_f_1", "r_f_2", "r_f_3",
            "files",
        ]
        for k in expected_keys:
            assert k in result, f"Missing key: {k}"

        assert result["vertices"].shape == (4, 3)
        assert result["faces"].shape == (4, 3)
        assert result["edges"].shape[1] == 4
        assert len(result["centroid_faces"]) == 4
        assert all(len(c) == 3 for c in result["centroid_faces"])
        assert len(result["centroid_edges"]) == result["edges"].shape[0]
        assert all(len(c) == 3 for c in result["centroid_edges"])
        assert result["e_e"].shape[0] == result["edges"].shape[0]
        assert result["n_f"].shape == (4, 3)
        assert result["n_f_e"].shape[0] == result["edges"].shape[0]
        assert result["n_fp_e"].shape[0] == result["edges"].shape[0]

        assert np.all(np.isfinite(result["vertices"]))
        assert np.all(np.isfinite(result["edges"]))
        assert np.all(np.isfinite(result["e_e"]))
        assert np.all(np.isfinite(result["n_f"]))
        assert np.all(np.isfinite(result["n_f_e"]))
        assert np.all(np.isfinite(result["n_fp_e"]))

    def test_verbose_mode_does_not_crash(self, tmp_path):
        asteroid = "VerboseTest"
        base_dir = str(tmp_path)

        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        faces_1based = np.array([
            [2, 4, 3],
            [1, 3, 4],
            [1, 4, 2],
            [1, 2, 3],
        ], dtype=np.int64)

        self._write_shape_files(tmp_path, asteroid, vertices, faces_1based)
        result = prepare_werner_model(asteroid=asteroid, base_dir=base_dir, verbose=True)

        assert result["vertices"].shape == (4, 3)
        assert len(result["centroid_faces"]) == 4
        assert np.all(np.isfinite(result["n_f"]))