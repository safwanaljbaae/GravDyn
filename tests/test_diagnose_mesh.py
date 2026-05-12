import numpy as np
import pytest
import trimesh

from gravdyn.shape_verification import diagnose_polyhedral_mesh


class TestDiagnosePolyhedralMesh:
    def test_valid_mesh_has_no_problems(self):
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        report = diagnose_polyhedral_mesh(mesh, verbose=False, return_elements=True)
        assert report["is_watertight"]
        assert report["is_volume"]
        assert report["num_boundary_edges"] == 0
        assert report["num_degenerate_faces"] == 0
        assert report["num_duplicate_faces"] == 0
        assert report["num_unused_vertices"] == 0
        assert len(report["problems"]) == 0

    def test_non_trimesh_raises(self):
        with pytest.raises(TypeError, match="trimesh.Trimesh"):
            diagnose_polyhedral_mesh("not_a_mesh")

    def test_detects_boundary_edges(self):
        mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            faces=[[0, 1, 2], [0, 1, 3]],
            process=False,
        )
        report = diagnose_polyhedral_mesh(mesh, verbose=False)
        assert report["num_boundary_edges"] > 0
        assert any("boundary" in p.lower() for p in report["problems"])

    def test_detects_degenerate_faces(self):
        mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[[0, 0, 1], [0, 1, 2]],
            process=False,
        )
        report = diagnose_polyhedral_mesh(mesh, verbose=False)
        assert report["num_degenerate_faces"] > 0

    def test_detects_duplicate_faces(self):
        mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            faces=[[0, 1, 2], [0, 1, 2], [0, 1, 3]],
            process=False,
        )
        report = diagnose_polyhedral_mesh(mesh, verbose=False)
        assert report["num_duplicate_faces"] > 0

    def test_detects_unused_vertices(self):
        mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [99, 99, 99]],
            faces=[[0, 1, 2]],
            process=False,
        )
        report = diagnose_polyhedral_mesh(mesh, verbose=False)
        assert report["num_unused_vertices"] > 0

    def test_return_elements_structure(self):
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        report = diagnose_polyhedral_mesh(mesh, verbose=False, return_elements=True)
        assert "problem_elements" in report
        for key in ("boundary_edges", "degenerate_faces", "duplicate_face_ids",
                     "unused_vertices", "broken_faces"):
            assert key in report["problem_elements"]

    def test_return_elements_false_omits_problem_elements(self):
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        report = diagnose_polyhedral_mesh(mesh, verbose=False, return_elements=False)
        assert "problem_elements" not in report

    def test_basic_info_present(self):
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        report = diagnose_polyhedral_mesh(mesh, verbose=False)
        for key in ("num_vertices", "num_faces", "is_watertight", "is_volume",
                     "volume", "num_boundary_edges", "num_degenerate_faces"):
            assert key in report

    def test_verbose_does_not_crash(self, capsys):
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        diagnose_polyhedral_mesh(mesh, verbose=True)
        captured = capsys.readouterr()
        assert "Polyhedral mesh diagnostic" in captured.out
