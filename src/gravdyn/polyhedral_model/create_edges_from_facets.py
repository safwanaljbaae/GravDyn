import numpy as np
from collections import defaultdict

from gravdyn.polyhedral_model.load_vertices_faces import load_vertices_faces


def create_edges_from_facets(faces: list, files: classmethod, verbose: bool) -> np.ndarray:

    """
    Generate an `edges.dat`-style table from triangular face connectivity, recording
    each unique undirected edge and the two faces that meet at it.

    The function scans the triangular mesh (`faces`), enumerates the three edges of
    each face, normalizes each edge as an unordered pair `(min(v), max(v))` to avoid
    duplicates, and maps every unique edge → list of incident face IDs. It then
    formats a four-column table `[v1, v2, f1, f2]` and writes it to `file_edges`
    as tab-separated integers. The output preserves 1-based face indices
    (`f1`, `f2` = row numbers of `faces`), and keeps vertex indices exactly as
    supplied in `faces` (commonly 1-based in MATLAB/Julia workflows).

    NOTE:
      - Closed manifold assumption: in a closed triangular mesh, each undirected edge
        is shared by exactly two faces. If an edge appears in only one face, the mesh
        has a boundary; if it appears in more than two, it is non-manifold.
      - Expected edge count: for a consistent closed triangular mesh with `F` faces,
        the number of unique edges should be `E = 3F/2`.
      - Sorting: the returned array is lexicographically sorted by `(f1, f2, v1, v2)`
        to mimic common MATLAB pipelines and facilitate reproducibility.
      - Indexing convention: face IDs in the output are 1-based (`i + 1`); vertex
        indices are written as provided. If your `faces` array uses 0-based indices,
        either convert to 1-based beforehand or keep in mind the mixed conventions
        when consuming `edges.dat`.

    :param faces: Triangular face connectivity, shape (M, 3), dtype=int.
                  Each row lists three vertex indices forming one face. Values are
                  carried through to the output as `v1`, `v2` (after per-edge sorting).
    :param files: Path to save the edge table.
    :param verbose: If True, print progress messages.

    :return: numpy.ndarray of shape (E, 4), dtype=int, containing the sorted edge table:
             - Column 0: v1 (smaller of the two vertex indices for the edge)
             - Column 1: v2 (larger vertex index)
             - Column 2: f1 (1-based index of one incident face)
             - Column 3: f2 (1-based index of the other incident face)
    """
    edge_to_faces = defaultdict(list)

    # Step 1: Build edge -> [faces]
    for i, face in enumerate(faces):
        v1, v2, v3 = face
        for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
            edge = tuple(sorted((a, b)))
            edge_to_faces[edge].append(i + 1)  # 1-based like MATLAB

    # Step 2: Format output
    edge_list = []
    for (v1, v2), fs in edge_to_faces.items():
        f1 = fs[0]
        f2 = fs[1] if len(fs) > 1 else fs[0]  # Duplicate if only one face
        edge_list.append([v1, v2, f1, f2])

    # Step 3: Sort as MATLAB: by f1, f2, v1, v2
    edge_array = np.array(edge_list, dtype=int)
    sorted_edges = edge_array[np.lexsort((edge_array[:,1], edge_array[:,0], edge_array[:,3], edge_array[:,2]))]

    # Step 4: Save
    np.savetxt(files.file_edges, sorted_edges, fmt="%d", delimiter="\t")
    print(f"    [✓] Sorted edges saved to {files.file_edges}")

    return sorted_edges



if __name__ == "__main__":
    
    from src.potential.polyhedral_model.poly_files import PolyFiles

    base_dir: str = "Data"
    asteroid: str = "Apophis"
    files = PolyFiles(base_dir=base_dir, asteroid=asteroid)

    vertices, faces = load_vertices_faces(files.file_vertices, files.file_faces)
    print(f"    Loaded {vertices.shape[0]} vertices and {faces.shape[0]} faces.")

    # Compute edges
    edges = create_edges_from_facets(faces, files)

