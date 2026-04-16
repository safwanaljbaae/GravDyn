
import numpy as np

from gravdyn.polyhedral_model.load_vertices_faces import load_vertices_faces, _load_dat

def compute_polyhedron_centroids(vertices, faces, edges, files: classmethod, verbose: bool = True):
    """
    Compute and save the geometric centroids of faces and edges for a polyhedral shape model.

    The function takes vertex coordinates, face connectivity, and edge definitions
    (including associated face indices), computes:
      - the centroid of each triangular face (mean of its three vertices), and
      - the centroid of each edge (midpoint of its two vertices).
    Results are written to text files `centroid_faces.dat` and `centroid_edges.dat`
    in scientific notation, preserving the order of the input lists.

    NOTE:
      - Vertex, face, and edge indexing follows the 1-based convention used in the
        input data files (`shape_v.dat`, `shape_f.dat`, and `edges.dat`).
      - Each output line corresponds directly to the same line index of its source
        entity (e.g., line N in `centroid_faces.dat` is the centroid of face N).
      - Centroid coordinates are written with 20-digit scientific precision,
        separated by tab characters.

    :param vertices: numpy.ndarray or list of shape (N, 3)
        Array containing 3D coordinates (X, Y, Z) of all vertices in the polyhedron.
        The order must match the vertex indices referenced in `faces` and `edges`.
    :param faces: numpy.ndarray or list of shape (F, 3)
        Array defining the triangular faces of the polyhedron, with 1-based vertex
        indices (i, j, k) per row.
    :param edges: numpy.ndarray or list of shape (E, 4)
        Array defining edges with 1-based vertex indices and adjacent faces:
        [v1, v2, f1, f2], where v1 and v2 are the two end vertices, and f1, f2
        are the faces that share that edge.
    :param files: classmethod PolyFiles
        Dataclass instance containing file paths for input/output data.
    :param verbose: If True, print progress messages.

    :return: tuple (face_centroids, edge_centroids)
        - face_centroids: list of tuples (cx, cy, cz) for each face.
        - edge_centroids: list of tuples (mx, my, mz) for each edge.

    """
    face_centroids = []
    for (i, j, k) in faces:
        # Convert 1-based indices to 0-based for list access
        x1, y1, z1 = vertices[i-1]
        x2, y2, z2 = vertices[j-1]
        x3, y3, z3 = vertices[k-1]
        centroid_x = (x1 + x2 + x3) / 3.0
        centroid_y = (y1 + y2 + y3) / 3.0
        centroid_z = (z1 + z2 + z3) / 3.0
        face_centroids.append((centroid_x, centroid_y, centroid_z))

    # Compute centroids for each edge (midpoint of 2 vertices)
    edge_centroids = []
    for (v1, v2, f1, f2) in edges:
        # Only v1 and v2 are used for midpoint calculation
        x1, y1, z1 = vertices[v1-1]
        x2, y2, z2 = vertices[v2-1]
        midpoint_x = (x1 + x2) / 2.0
        midpoint_y = (y1 + y2) / 2.0
        midpoint_z = (z1 + z2) / 2.0
        edge_centroids.append((midpoint_x, midpoint_y, midpoint_z))

    # Write centroids to output files, preserving the input order
    with open(files.file_centroid_faces, 'w') as cf:
        for (cx, cy, cz) in face_centroids:
            cf.write(f"{cx:.20e}\t{cy:.20e}\t{cz:.20e}\n")
    
    if verbose:
        print(f"    [✓] Sorted centroide faces saved to {files.file_centroid_faces}")

    with open(files.file_centroid_edges, 'w') as ce:
        for (mx, my, mz) in edge_centroids:
            ce.write(f"{mx:.20e}\t{my:.20e}\t{mz:.20e}\n")
    if verbose:
        print(f"    [✓] Sorted centroide edges saved to {files.file_centroid_edges}")
    
    return face_centroids, edge_centroids

if __name__ == "__main__":

    from src.potential.polyhedral_model.poly_files import PolyFiles

    file_vertices, file_faces, file_edges = "Data/Apophis/shape_v.dat", "Data/Apophis/shape_f.dat", "Data/Apophis/edges.dat"
    # Example usage
    vertices, faces = load_vertices_faces(file_vertices, file_faces)
    print(f"    Loaded {vertices.shape[0]} vertices and {faces.shape[0]} faces.")
    edges = _load_dat(file_edges).astype(np.int64)
    print(f"    Loaded {edges.shape[0]} edges.")


    base_dir: str = "Data"
    asteroid: str = "Apophis"
    files = PolyFiles(base_dir=base_dir, asteroid=asteroid)

    # Compute edges
    c_faces, c_edges = compute_polyhedron_centroids(vertices, faces, edges, files)