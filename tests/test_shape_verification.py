# pytest -v
# tests/test_shape_verification_real_data.py
from pathlib import Path
import numpy as np
import trimesh
import pytest
from gravdyn.shape_verification import shape_verification, load_vertices, load_faces


def test_shape_verification_with_real_files(tmp_path, monkeypatch):
    """
    Integration test using real Apophis data files.
    """

    asteroid_name = "Apophis"
    mass = 5.31e10
    density = 1.75

    vertices_file = "/home/aljbaae/Script_Safwan/GravDyn/Data/Apophis/shape_v.dat"
    faces_file = "/home/aljbaae/Script_Safwan/GravDyn/Data/Apophis/shape_f.dat"

    # Ensure input files exist (otherwise skip test)
    if not Path(vertices_file).exists() or not Path(faces_file).exists():
        pytest.skip("Apophis data files not available")

    # Run inside temp directory to avoid overwriting real data
    monkeypatch.chdir(tmp_path)

    shape_verification(
        asteroid_name=asteroid_name,
        mass=mass,
        density=density,
        vertices_file=str(vertices_file),
        faces_file=str(faces_file),
    )

    # ---- Assertions ----

    output_file = tmp_path / "Data" / asteroid_name / "modified_v.dat"
    assert output_file.exists()

    vertices = load_vertices(vertices_file=vertices_file)
    faces = load_faces(faces_file=faces_file)

    # Basic sanity checks
    assert vertices.ndim == 2
    assert vertices.shape[1] == 3
    assert np.isfinite(vertices).all()

    # Optional stronger check: center ~ 0 after recentering
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert np.allclose(mesh.center_mass, [0, 0, 0], atol=1e-8)