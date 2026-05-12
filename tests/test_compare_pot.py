import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from gravdyn.plot_tools import compare_pot


class TestComparePot:
    @pytest.fixture
    def mock_data_dir(self, tmp_path):
        asteroid = "TestAsteroid"
        d = tmp_path / asteroid
        d.mkdir(parents=True, exist_ok=True)

        xyz = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        r = np.linalg.norm(xyz, axis=1)

        df_w = pd.DataFrame({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
                              "potential": [1.0, 0.5, 0.25], "r": r})
        df_w.to_csv(d / "pot_Werner.csv", index=False)

        df_m = pd.DataFrame({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
                              "potential": [1.01, 0.51, 0.26]})
        df_m.to_csv(d / "pot_Mascon.csv", index=False)

        df_e = pd.DataFrame({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
                              "potential": [1.02, 0.52, 0.27]})
        df_e.to_csv(d / "pot_Expansion.csv", index=False)

        verts = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                          [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
        np.savetxt(d / "modified_v.dat", verts, fmt="%.6f")
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                          [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                          [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
        np.savetxt(d / "modified_f.dat", faces, fmt="%d")

        return str(tmp_path)

    def test_runs_with_mascon_and_expansion(self, mock_data_dir):
        compare_pot("TestAsteroid", mock_data_dir, compare_mascon=True,
                     compare_expansion=True)
        out = Path(mock_data_dir) / "TestAsteroid" / "d_pot.png"
        assert out.exists()
        assert out.stat().st_size > 0

    def test_runs_with_mascon_only(self, mock_data_dir):
        compare_pot("TestAsteroid", mock_data_dir, compare_mascon=True,
                     compare_expansion=False)
        out = Path(mock_data_dir) / "TestAsteroid" / "d_pot.png"
        assert out.exists()

    def test_runs_with_expansion_only(self, mock_data_dir):
        compare_pot("TestAsteroid", mock_data_dir, compare_mascon=False,
                     compare_expansion=True)
        out = Path(mock_data_dir) / "TestAsteroid" / "d_pot.png"
        assert out.exists()

    def test_mismatched_coordinates_raises(self, mock_data_dir):
        asteroid = "TestAsteroid"
        df = pd.DataFrame({"x": [99.0], "y": [0.0], "z": [0.0], "potential": [1.0]})
        df.to_csv(Path(mock_data_dir) / asteroid / "pot_Mascon.csv", index=False)
        with pytest.raises(ValueError, match="different number of points"):
            compare_pot(asteroid, mock_data_dir, compare_mascon=True,
                         compare_expansion=False)
