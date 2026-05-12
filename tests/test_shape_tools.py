import numpy as np
import pytest
from pathlib import Path

from gravdyn.shape_tools import load_vertices, load_faces


class TestLoadVertices:
    def test_loads_3d_points(self, tmp_path):
        f = tmp_path / "v.dat"
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.savetxt(f, data)
        result = load_vertices(str(f))
        np.testing.assert_allclose(result, data)
        assert result.shape == (2, 3)

    def test_single_vertex_reshaped(self, tmp_path):
        f = tmp_path / "v_single.dat"
        np.savetxt(f, [[1.0, 2.0, 3.0]])
        result = load_vertices(str(f))
        assert result.shape == (1, 3)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_vertices("/nonexistent/path.dat")

    def test_directory_raises_value_error(self, tmp_path):
        d = tmp_path / "adir"
        d.mkdir()
        with pytest.raises(ValueError, match="not a file"):
            load_vertices(str(d))

    def test_wrong_columns_raises(self, tmp_path):
        f = tmp_path / "bad.dat"
        np.savetxt(f, [[1.0, 2.0]])
        with pytest.raises(ValueError, match="exactly 3 columns"):
            load_vertices(str(f))

    def test_returns_float64(self, tmp_path):
        f = tmp_path / "v.dat"
        np.savetxt(f, [[1, 2, 3]], fmt="%d")
        result = load_vertices(str(f))
        assert result.dtype == np.float64


class TestLoadFaces:
    def test_loads_0based_faces(self, tmp_path):
        f = tmp_path / "f.dat"
        data = np.array([[0, 1, 2], [1, 2, 3]])
        np.savetxt(f, data, fmt="%d")
        result = load_faces(str(f))
        np.testing.assert_array_equal(result, data)

    def test_converts_1based_to_0based(self, tmp_path):
        f = tmp_path / "f.dat"
        data = np.array([[1, 2, 3], [2, 3, 4]])
        np.savetxt(f, data, fmt="%d")
        result = load_faces(str(f))
        np.testing.assert_array_equal(result, data - 1)

    def test_single_face_reshaped(self, tmp_path):
        f = tmp_path / "f_single.dat"
        np.savetxt(f, [[0, 1, 2]], fmt="%d")
        result = load_faces(str(f))
        assert result.shape == (1, 3)

    def test_mixed_indexing_no_conversion(self, tmp_path):
        f = tmp_path / "f.dat"
        data = np.array([[0, 1, 2], [0, 2, 3]])
        np.savetxt(f, data, fmt="%d")
        result = load_faces(str(f))
        np.testing.assert_array_equal(result, data)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_faces("/nonexistent/path.dat")

    def test_wrong_columns_raises(self, tmp_path):
        f = tmp_path / "bad.dat"
        np.savetxt(f, [[0, 1]], fmt="%d")
        with pytest.raises(ValueError, match="exactly 3 columns"):
            load_faces(str(f))

    def test_returns_int64(self, tmp_path):
        f = tmp_path / "f.dat"
        np.savetxt(f, [[0, 1, 2]], fmt="%d")
        result = load_faces(str(f))
        assert result.dtype == np.int64
