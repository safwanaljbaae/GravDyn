from __future__ import annotations

import numpy as np
import pytest
import trimesh
from pathlib import Path

from gravdyn.shape_verification import shape_verification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_spherical_mesh():
    """Return (vertices, faces) of a valid closed mesh centered at origin."""
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    return mesh.vertices, mesh.faces


def _write_mesh_files(directory, asteroid, vertices, faces,
                      v_name="shape_v.dat", f_name="shape_f.dat"):
    """Write vertex / face files to ``directory / asteroid /``."""
    out = Path(directory) / asteroid
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / v_name, vertices, fmt="%.6f")
    np.savetxt(out / f_name, faces, fmt="%d")
    return str(directory), v_name, f_name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestShapeVerificationPipeline:
    """Integration tests for the full shape_verification pipeline."""

    def test_basic_pipeline(self, tmp_path):
        """Complete pipeline with a synthetic mesh (no rescaling)."""
        vertices, faces = _valid_spherical_mesh()
        base_dir, v_file, f_file = _write_mesh_files(tmp_path, "TestBody", vertices, faces)

        density = 1000.0
        mesh_in = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        vol_in = mesh_in.volume
        mass = vol_in * density / 1e-12   # cancels 1e-12 conversion_factor

        shape_verification(
            asteroid_name="TestBody",
            mass=mass,
            density=density,
            base_dir=base_dir,
            vertices_file=v_file,
            faces_file=f_file,
        )

        out_dir = Path(base_dir) / "TestBody"

        # --- output files exist ---
        assert (out_dir / "modified_v.dat").exists()
        assert (out_dir / "modified_f.dat").exists()
        assert (out_dir / "shape_projection.png").exists()
        assert (out_dir / "shape_verification.log").exists()

        # --- output vertices are valid ---
        out_v = np.loadtxt(out_dir / "modified_v.dat")
        out_f = np.loadtxt(out_dir / "modified_f.dat", dtype=int)
        assert out_v.ndim == 2
        assert out_v.shape[1] == 3
        assert np.isfinite(out_v).all()
        assert out_f.shape[1] == 3

        # --- volume matches reference after scaling ---
        mesh_out = trimesh.Trimesh(vertices=out_v, faces=out_f, process=False)
        ref_volume = (mass / density) * 1e-12
        rel_error = abs(mesh_out.volume - ref_volume) / ref_volume
        assert rel_error < 5e-6, f"Volume mismatch: {mesh_out.volume} vs {ref_volume}"

        # --- center of mass is near origin ---
        assert np.allclose(mesh_out.center_mass, [0, 0, 0], atol=1e-8)

    def test_pipeline_with_scaling(self, tmp_path):
        """Pipeline where the mesh is rescaled to a different volume."""
        vertices, faces = _valid_spherical_mesh()
        base_dir, v_file, f_file = _write_mesh_files(tmp_path, "ScaledBody", vertices, faces)

        mass = 1.0e10
        density = 500.0
        ref_volume = (mass / density) * 1e-12

        shape_verification(
            asteroid_name="ScaledBody",
            mass=mass,
            density=density,
            base_dir=base_dir,
            vertices_file=v_file,
            faces_file=f_file,
        )

        out_v = np.loadtxt(Path(base_dir) / "ScaledBody" / "modified_v.dat")
        out_f = np.loadtxt(Path(base_dir) / "ScaledBody" / "modified_f.dat", dtype=int)
        mesh_out = trimesh.Trimesh(vertices=out_v, faces=out_f, process=False)

        rel_error = abs(mesh_out.volume - ref_volume) / ref_volume
        assert rel_error < 5e-6, f"Volume mismatch after scaling: {mesh_out.volume} vs {ref_volume}"

    def test_invalid_mesh_raises_error(self, tmp_path):
        """Mesh with degenerate faces should raise RuntimeError."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        faces = np.array([
            [0, 0, 0],   # degenerate – repeated vertex
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
        ])
        base_dir, v_file, f_file = _write_mesh_files(tmp_path, "BadMesh", vertices, faces)

        with pytest.raises(RuntimeError, match="Mesh has topological problems"):
            shape_verification(
                asteroid_name="BadMesh",
                mass=1.0,
                density=1.0,
                base_dir=base_dir,
                vertices_file=v_file,
                faces_file=f_file,
            )


class TestShapeVerificationErrors:
    """Unit-style tests for error handling."""

    def test_missing_vertices_file(self, tmp_path):
        """Non-existent vertex file raises FileNotFoundError."""
        base_dir = str(tmp_path)
        asteroid = "Missing"
        (Path(base_dir) / asteroid).mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileNotFoundError, match="does not exist"):
            shape_verification(
                asteroid_name=asteroid,
                mass=1.0,
                density=1.0,
                base_dir=base_dir,
                vertices_file="nonexistent.dat",
                faces_file="shape_f.dat",
            )

    def test_missing_faces_file(self, tmp_path):
        """Non-existent face file raises FileNotFoundError."""
        vertices, _ = _valid_spherical_mesh()
        base_dir, v_file, _ = _write_mesh_files(tmp_path, "MissingFace", vertices,
                                                 np.array([[0, 1, 2]]))

        with pytest.raises(FileNotFoundError, match="does not exist"):
            shape_verification(
                asteroid_name="MissingFace",
                mass=1.0,
                density=1.0,
                base_dir=base_dir,
                vertices_file=v_file,
                faces_file="nonexistent.dat",
            )

    def test_wrong_vertex_columns_raises_error(self, tmp_path):
        """File with wrong number of columns should raise ValueError."""
        asteroid = "BadColumns"
        out = Path(tmp_path) / asteroid
        out.mkdir(parents=True, exist_ok=True)

        np.savetxt(out / "bad_v.dat", np.array([[1.0, 2.0]]))         # only 2 columns
        np.savetxt(out / "bad_f.dat", np.array([[0, 1, 2]]), fmt="%d")

        with pytest.raises(ValueError, match="exactly 3 columns"):
            shape_verification(
                asteroid_name=asteroid,
                mass=1.0,
                density=1.0,
                base_dir=str(tmp_path),
                vertices_file="bad_v.dat",
                faces_file="bad_f.dat",
            )