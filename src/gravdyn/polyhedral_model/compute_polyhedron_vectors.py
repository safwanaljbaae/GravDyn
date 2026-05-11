# -*- coding: utf-8 -*-
"""
# !===============================================================
# !==   Dr. Safwan Aljbaae                                      ==
# !==   October 2025                                            ==
# !===============================================================
# python3 -m pip install -r requirements.txt                    ==
# !===============================================================
"""
import numpy as np


def compute_polyhedron_vectors(vertices, faces, edges, files, verbose: bool = True):
    """
    Compute and store all geometric vectors required for the polyhedral gravity model of an asteroid.

    This function computes key quantities based on a polyhedral shape model defined by its vertices,
    triangular faces, and edges. The resulting vectors are essential for evaluating gravitational
    potential and acceleration in polyhedral formulations (e.g., Werner & Scheeres model).

    The following components are computed:
      - Edge lengths.
      - Face unit normal vectors.
      - Edge-normal vectors projected within adjacent face planes (for both sides of each edge).
      - Vertex coordinates per edge and per face (stored separately for future acceleration kernels).

    All outputs are saved in .dat files for efficient reuse. The function also returns these arrays
    in a dictionary for immediate in-memory access.

    NOTE:
    - Input face and edge arrays must use 1-based indexing (as commonly found in shape model files).
    - Output files are written in tab-separated ASCII format using 20-digit scientific notation.
    - The edge-face plane normals ('n_f_e', 'n_fp_e') are adjusted to point outward from the face center
        to ensure correct orientation in gravity computations.

    :param vertices: (np.ndarray) Array of shape (N_vertices, 3) containing vertex coordinates.
    :param faces: (np.ndarray) Array of shape (N_faces, 3) with vertex indices (1-based) for each triangle.
    :param edges: (np.ndarray) Array of shape (N_edges, 4) where each row is [v1, v2, f1, f2]
                  with vertex and adjacent face indices (1-based).
    :param files: An object containing paths to output files where each computed array will be saved.
                  Must include file paths for e_e, n_f, n_f_e, n_fp_e, r_e_1, r_e_2, r_f_1, r_f_2, r_f_3.
    :param verbose: If True, prints progress updates and save locations for each computed array.

    :return: dict[str, np.ndarray]
        A dictionary containing:
            - 'e_e'     : edge lengths.
            - 'n_f'     : face unit normal vectors.
            - 'n_f_e'   : face-plane edge normals (face1 side).
            - 'n_fp_e'  : face-plane edge normals (face2 side).
            - 'r_e_1'   : coordinates of vertex 1 for each edge.
            - 'r_e_2'   : coordinates of vertex 2 for each edge.
            - 'r_f_1'   : coordinates of vertex 1 for each face.
            - 'r_f_2'   : coordinates of vertex 2 for each face.
            - 'r_f_3'   : coordinates of vertex 3 for each face.

    """

    faces_idx = faces
    edge_v1_idx = edges[:, 0]-1
    edge_v2_idx = edges[:, 1]-1
    face1_idx = edges[:, 2]-1
    face2_idx = edges[:, 3]-1

    # 2. Extract coordinates for faces and edges using the indices
    r_f_1 = vertices[faces_idx[:, 0]]
    r_f_2 = vertices[faces_idx[:, 1]]
    r_f_3 = vertices[faces_idx[:, 2]]
    r_e_1 = vertices[edge_v1_idx]
    r_e_2 = vertices[edge_v2_idx]

    # 3. Compute edge lengths (Euclidean norm of edge vectors)
    edge_vectors = r_e_2 - r_e_1
    e_e = np.linalg.norm(edge_vectors, axis=1)  # length of each edge

    # 4. Compute face unit normals (n_f) via cross product of two edge vectors for each face
    vec1 = r_f_2 - r_f_1
    vec2 = r_f_3 - r_f_2
    non_unit_normals = np.cross(vec1, vec2)                          # cross product for each face:contentReference[oaicite:16]{index=16}
    n_f = non_unit_normals / np.linalg.norm(non_unit_normals, axis=1)[:, None]  # normalize to unit length

    # 5. Compute edge-plane normals for each adjacent face of each edge (n_f_e for face1, n_fp_e for face2)
    # Compute face centroids and edge midpoints
    centroid_faces = (r_f_1 + r_f_2 + r_f_3) / 3.0
    centroid_edges = (r_e_1 + r_e_2) / 2.0

    # For face1 of each edge:
    vec_face1 = centroid_edges - centroid_faces[face1_idx]                    # vector from face1 centroid to edge midpoint
    unit_face1_dir = vec_face1 / np.linalg.norm(vec_face1, axis=1)[:, None]   # normalize within face1 plane
    v_perp1 = np.cross(unit_face1_dir, edge_vectors)                          # perpendicular to face1 plane (temp vector)
    v_inplane1 = np.cross(v_perp1, edge_vectors)                              # in-plane vector perpendicular to edge:contentReference[oaicite:17]{index=17}
    n_f_e = v_inplane1 / np.linalg.norm(v_inplane1, axis=1)[:, None]          # normalize to unit
    # Flip to ensure it points outward from face1
    dot1 = np.sum(unit_face1_dir * n_f_e, axis=1)
    n_f_e[dot1 < 0] *= -1                                                     # flip if pointing inward:contentReference[oaicite:18]{index=18}

    # For face2 of each edge:
    vec_face2 = centroid_edges - centroid_faces[face2_idx]
    unit_face2_dir = vec_face2 / np.linalg.norm(vec_face2, axis=1)[:, None]
    v_perp2 = np.cross(unit_face2_dir, edge_vectors)
    v_inplane2 = np.cross(v_perp2, edge_vectors)
    n_fp_e = v_inplane2 / np.linalg.norm(v_inplane2, axis=1)[:, None]
    dot2 = np.sum(unit_face2_dir * n_fp_e, axis=1)
    n_fp_e[dot2 < 0] *= -1                                                    # flip if pointing inward:contentReference[oaicite:19]{index=19}
    
    # 6. Write all output arrays to .dat files (tab-separated values, ASCII format)
    np.savetxt(files.file_e_e, e_e, delimiter='\t', fmt='%.20e')
    if verbose:
        print("            Saved edge lengths to", files.file_e_e)
    np.savetxt(files.file_n_f, n_f, delimiter='\t', fmt='%.20e')
    if verbose:
        print("            Saved face normals to", files.file_n_f)
    np.savetxt(files.file_n_f_e, n_f_e, delimiter='\t', fmt='%.20e')
    if verbose:
        print("            Saved edge-face1 normals to", files.file_n_f_e)
    np.savetxt(files.file_n_fp_e, n_fp_e, delimiter='\t', fmt='%.20e')
    if verbose:
        print("            Saved edge-face2 normals to", files.file_n_fp_e)
    np.savetxt(files.file_r_e_1, r_e_1, delimiter='\t', fmt='%.20e')
    if verbose:
        print("            Saved edge vertex 1 coordinates to", files.file_r_e_1)
    np.savetxt(files.file_r_e_2, r_e_2, delimiter='\t', fmt='%.20e')
    if verbose:
        print("            Saved edge vertex 2 coordinates to", files.file_r_e_2)
    np.savetxt(files.file_r_f_1, r_f_1, delimiter='\t', fmt='%.20e')
    if verbose:
        print("            Saved face vertex 1 coordinates to", files.file_r_f_1)
    np.savetxt(files.file_r_f_2, r_f_2, delimiter='\t', fmt='%.20e')
    if verbose:
        print("            Saved face vertex 2 coordinates to", files.file_r_f_2)
    np.savetxt(files.file_r_f_3, r_f_3, delimiter='\t', fmt='%.20e')
    if verbose:
        print("            Saved face vertex 3 coordinates to", files.file_r_f_3)
    
    polyhedron_vectors = {
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
    return polyhedron_vectors

