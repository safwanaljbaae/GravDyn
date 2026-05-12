# tests/test_pot_polyhedral_model.py
from __future__ import annotations
import numpy as np
import pytest
from gravdyn import pot_werner_model, batched_werner_potential


@pytest.fixture
def polyhedral_data():
    """Build a valid polyhedral dataset for a regular tetrahedron."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ], dtype=np.int64)

    # edges: [v1, v2, f1, f2] using 1-based indexing
    edges = np.array([
        [1, 2, 1, 2],
        [1, 3, 1, 3],
        [1, 4, 2, 3],
        [2, 3, 1, 4],
        [2, 4, 2, 4],
        [3, 4, 3, 4],
    ], dtype=np.int64)

    r_f_1 = vertices[faces[:, 0]]
    r_f_2 = vertices[faces[:, 1]]
    r_f_3 = vertices[faces[:, 2]]

    edge_v1 = edges[:, 0] - 1
    edge_v2 = edges[:, 1] - 1
    r_e_1 = vertices[edge_v1]
    r_e_2 = vertices[edge_v2]

    centroid_faces = (r_f_1 + r_f_2 + r_f_3) / 3.0
    centroid_edges = (r_e_1 + r_e_2) / 2.0

    edge_vectors = r_e_2 - r_e_1
    e_e = np.linalg.norm(edge_vectors, axis=1)

    vec1 = r_f_2 - r_f_1
    vec2 = r_f_3 - r_f_2
    nn = np.cross(vec1, vec2)
    n_f = nn / np.linalg.norm(nn, axis=1)[:, None]

    face1_idx = edges[:, 2] - 1
    face2_idx = edges[:, 3] - 1

    vec_f1 = centroid_edges - centroid_faces[face1_idx]
    u_f1 = vec_f1 / np.linalg.norm(vec_f1, axis=1)[:, None]
    vp1 = np.cross(u_f1, edge_vectors)
    vip1 = np.cross(vp1, edge_vectors)
    n_f_e = vip1 / np.linalg.norm(vip1, axis=1)[:, None]
    n_f_e[np.sum(u_f1 * n_f_e, axis=1) < 0] *= -1

    vec_f2 = centroid_edges - centroid_faces[face2_idx]
    u_f2 = vec_f2 / np.linalg.norm(vec_f2, axis=1)[:, None]
    vp2 = np.cross(u_f2, edge_vectors)
    vip2 = np.cross(vp2, edge_vectors)
    n_fp_e = vip2 / np.linalg.norm(vip2, axis=1)[:, None]
    n_fp_e[np.sum(u_f2 * n_fp_e, axis=1) < 0] *= -1

    return {
        "vertices": vertices,
        "faces": faces,
        "edges": edges,
        "centroid_faces": centroid_faces,
        "centroid_edges": centroid_edges,
        "e_e": e_e,
        "n_f": n_f,
        "n_f_e": n_f_e,
        "n_fp_e": n_fp_e,
        "r_e_1": r_e_1,
        "r_e_2": r_e_2,
        "r_f_1": r_f_1,
        "r_f_2": r_f_2,
        "r_f_3": r_f_3,
    }


class TestShapeValidation:
    def test_invalid_single_point_shape(self, polyhedral_data):
        with pytest.raises(ValueError, match="stat must have shape"):
            pot_werner_model(gm_body=1.0, stat=[1.0, 2.0], polyhedral_data=polyhedral_data)

    def test_invalid_batch_shape(self, polyhedral_data):
        stat = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="stat must have shape"):
            pot_werner_model(gm_body=1.0, stat=stat, polyhedral_data=polyhedral_data)

    def test_3d_array(self, polyhedral_data):
        stat = np.array([[[1.0, 2.0, 3.0]]])
        with pytest.raises(ValueError, match="stat must have shape"):
            pot_werner_model(gm_body=1.0, stat=stat, polyhedral_data=polyhedral_data)


class TestComputation:
    def test_single_point(self, polyhedral_data):
        U, A = pot_werner_model(gm_body=1.0, stat=[2.0, 0.0, 0.0], polyhedral_data=polyhedral_data)
        assert np.ndim(U) == 0
        assert A.shape == (3,)
        assert np.isfinite(U)
        assert np.all(np.isfinite(A))

    def test_batch_points(self, polyhedral_data):
        stat = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        U, A = pot_werner_model(gm_body=1.0, stat=stat, polyhedral_data=polyhedral_data)
        assert U.shape == (3,)
        assert A.shape == (3, 3)
        assert np.all(np.isfinite(U))
        assert np.all(np.isfinite(A))

    def test_zero_gm_body(self, polyhedral_data):
        U, A = pot_werner_model(gm_body=0.0, stat=[2.0, 0.0, 0.0], polyhedral_data=polyhedral_data)
        assert U == pytest.approx(0.0)
        assert np.all(A == pytest.approx(0.0))

    def test_single_point_returns_scalar_and_3vec(self, polyhedral_data):
        U, A = pot_werner_model(gm_body=1.0, stat=[2.0, 0.0, 0.0], polyhedral_data=polyhedral_data)
        assert isinstance(U, (float, np.floating))
        assert A.shape == (3,)

    def test_list_input(self, polyhedral_data):
        U, A = pot_werner_model(gm_body=1.0, stat=[2.0, 0.0, 0.0], polyhedral_data=polyhedral_data)
        assert np.isfinite(U)
        assert A.shape == (3,)

    def test_multiple_points_from_list(self, polyhedral_data):
        stat = [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        U, A = pot_werner_model(gm_body=1.0, stat=stat, polyhedral_data=polyhedral_data)
        assert U.shape == (2,)
        assert A.shape == (2, 3)
        assert np.all(np.isfinite(U))
        assert np.all(np.isfinite(A))

    def test_batched_werner_potential(self, polyhedral_data):
        stat = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        U, A = batched_werner_potential(stat=stat, gm_body=1.0, polyhedral_data=polyhedral_data, batch_size=2)
        assert U.shape == (3,)
        assert A.shape == (3, 3)
        assert np.all(np.isfinite(U))
        assert np.all(np.isfinite(A))
