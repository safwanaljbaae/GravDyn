import pytest
import trimesh
from pathlib import Path

from gravdyn.plot_tools import save_all_mesh_problem_html


class TestSaveAllMeshProblemHtml:
    def test_returns_dict_with_all_problem_types(self, tmp_path):
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        output_dir = str(tmp_path / "diags")
        result = save_all_mesh_problem_html(mesh, output_dir=output_dir)
        expected_types = [
            "boundary_edges", "non_manifold_edges", "degenerate_faces",
            "duplicate_faces", "unused_vertices", "broken_faces", "components",
        ]
        for ptype in expected_types:
            assert ptype in result
            assert Path(result[ptype]).exists()
            assert Path(result[ptype]).stat().st_size > 0

    def test_creates_output_directory(self, tmp_path):
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        output_dir = str(tmp_path / "nested" / "diags")
        save_all_mesh_problem_html(mesh, output_dir=output_dir)
        assert Path(output_dir).is_dir()
        assert (Path(output_dir) / "boundary_edges.html").exists()

    def test_default_output_dir(self, tmp_path):
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp_path)
            result = save_all_mesh_problem_html(mesh)
            assert Path("mesh_diagnostics_html").is_dir()
            for ptype in result:
                assert Path(result[ptype]).exists()
        finally:
            os.chdir(cwd)
