import numpy as np
import pytest
import trimesh

from gravdyn.shape_verification import principal_axes


class TestPrincipalAxes:
    def test_box_aligned_to_identity(self):
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        eigenvectors, M_4x4, angles = principal_axes(mesh)
        assert eigenvectors.shape == (3, 3)
        assert M_4x4.shape == (4, 4)
        assert len(angles) == 3
        assert np.allclose(M_4x4[:3, :3], np.eye(3), atol=1e-10)

    def test_transformed_mesh_identity_after_apply(self):
        mesh = trimesh.creation.box(extents=[1.0, 2.0, 3.0])
        _, M_4x4, _ = principal_axes(mesh)
        mesh.apply_transform(M_4x4)
        new_eig, _, _ = principal_axes(mesh)
        assert np.allclose(new_eig, np.eye(3), atol=1e-10)

    def test_returns_orthonormal_columns(self):
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        eigenvectors, _, _ = principal_axes(mesh)
        for i in range(3):
            assert np.isclose(np.linalg.norm(eigenvectors[:, i]), 1.0)
            for j in range(i + 1, 3):
                assert np.isclose(np.dot(eigenvectors[:, i], eigenvectors[:, j]), 0, atol=1e-10)

    def test_cube_angles_are_0_or_180(self):
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        _, _, angles = principal_axes(mesh)
        for a in angles:
            assert np.isclose(abs(a), 0, atol=1e-10) or np.isclose(abs(a), 180, atol=1e-10)

    def test_ellipsoid_principal_axes_are_distinct(self):
        mesh = trimesh.creation.box(extents=[1.0, 2.0, 4.0])
        eigenvectors, _, _ = principal_axes(mesh)
        for i in range(3):
            assert np.any(np.abs(eigenvectors[:, i]) > 0.5)

    def test_homogeneous_matrix_rotation_part(self):
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 2.0])
        _, M_4x4, _ = principal_axes(mesh)
        R = M_4x4[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(abs(np.linalg.det(R)), 1.0, atol=1e-10)

    def test_single_tetrahedron(self):
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        eigenvectors, M_4x4, angles = principal_axes(mesh)
        assert eigenvectors.shape == (3, 3)
        assert np.all(np.isfinite(eigenvectors))
        assert all(np.isfinite(angles))
