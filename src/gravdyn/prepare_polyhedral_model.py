
from dataclasses import asdict
from typing import Dict, Any
import numpy as np
import argparse
import os

from gravdyn.polyhedral_model.poly_files import PolyFiles
from gravdyn.polyhedral_model.create_edges_from_facets import create_edges_from_facets
from gravdyn.polyhedral_model.load_vertices_faces import load_vertices_faces, _load_dat
from gravdyn.polyhedral_model.compute_polyhedron_vectors import compute_polyhedron_vectors
from gravdyn.polyhedral_model.compute_polyhedron_centroids import compute_polyhedron_centroids


def prepare_polyhedral_model(
        asteroid: str,
        base_dir: str = "Data",
        verbose: bool = True,
) -> Dict[str, Any]:
    """
    Prepare or load all necessary data structures for computing the gravitational potential of an asteroid
    using a polyhedral shape model. This includes loading vertex and face data, generating edges,
    computing centroids, and constructing polyhedron-specific vectors.

    This function serves as the main entry point for setting up a polyhedral gravity model. It loads
    precomputed data if available, and otherwise generates and saves missing components. The output
    dictionary contains all relevant arrays (vertices, faces, edges, vectors, centroids) and paths to
    the input/output files used in the model construction.

    The function is mainly used before applying a polyhedral gravity field to orbital or dynamical simulations.

    NOTE:
        - If any data file is missing, the function automatically computes and stores the necessary
          components (edges, centroids, vectors) unless explicitly skipped.
        - If critical shape model files (vertices or faces) are missing, the function exits early.
        - This routine is typically called once before any polyhedral potential evaluation begins.

    Parameters
    ----------
    asteroid :
        Name of the asteroid model (used to locate its folder in `base_dir`).
    base_dir :
        Path to the directory containing asteroid shape model files (default: 'Data').
    verbose :
        If True, prints progress and file status messages.

    Returns
    ----------
        - A dictionary with the following keys:
        - 'vertices': (np.ndarray) vertex coordinates of the polyhedral model.
        - 'faces': (np.ndarray) triangular face indices.
        - 'edges': (np.ndarray) edge definitions generated from the faces.
        - 'centroid_faces': (np.ndarray) centroids of all triangular faces.
        - 'centroid_edges': (np.ndarray) centroids of all edges.
        - 'e_e', 'n_f', 'n_f_e', 'n_fp_e': (np.ndarray) edge and face vectors used in potential calculation.
        - 'r_e_1', 'r_e_2', 'r_f_1', 'r_f_2', 'r_f_3': (np.ndarray) vectors from vertices to centroids and faces.
        - 'files': (dict) dictionary of all file paths used or created, from the PolyFiles structure.

    """

    files = PolyFiles(base_dir=base_dir, asteroid=asteroid)

    if verbose:
        print(f"    Preparing the polyhedral model of {asteroid}...")
        print(f"        Base directory: {base_dir}")
        print(f"        Root directory: {files.root}")

    # Graceful file existence check
    missing_files = []
    if not os.path.exists(files.file_vertices):
        missing_files.append(files.file_vertices)
    if not os.path.exists(files.file_faces):
        missing_files.append(files.file_faces)

    if missing_files:
        raise FileNotFoundError(
            f"Shape files do not exist: {missing_files}. Please ensure that you run the 'shape_verification' function beforehand."
        )

    vertices, faces = load_vertices_faces(files.file_vertices, files.file_faces)
    if verbose:
        print(f"        Loaded {vertices.shape[0]} vertices and {faces.shape[0]} faces.")

    if not os.path.exists(files.file_edges):
        if verbose:
            print(f"        Computing edges from faces...")
        edges = create_edges_from_facets(faces, files, verbose)
    else:
        if verbose:
            print(f"        Edges file already exists at {files.file_edges}")
        edges = _load_dat(files.file_edges).astype(np.int64)

    missing_files = []
    if not os.path.exists(files.file_centroid_edges):
        missing_files.append(files.file_centroid_edges)
    if not os.path.exists(files.file_centroid_faces):
        missing_files.append(files.file_centroid_faces)

    if missing_files:
        if verbose:
            print(f"        Computing centroids for faces and edges...")
        c_faces, c_edges = compute_polyhedron_centroids(vertices, faces, edges, files, verbose)
    else:
        if verbose:
            print(
                f"        Centroid files already exist at {files.file_centroid_edges} and {files.file_centroid_faces}")
        c_faces, c_edges = _load_dat(files.file_centroid_faces).astype(np.float64), _load_dat(
            files.file_centroid_edges).astype(np.float64)

    missing_files = []
    if not os.path.exists(files.file_e_e):
        missing_files.append(files.file_e_e)
    if not os.path.exists(files.file_n_f):
        missing_files.append(files.file_n_f)
    if not os.path.exists(files.file_n_f_e):
        missing_files.append(files.file_n_f_e)
    if not os.path.exists(files.file_n_fp_e):
        missing_files.append(files.file_n_fp_e)
    if not os.path.exists(files.file_r_e_1):
        missing_files.append(files.file_r_e_1)
    if not os.path.exists(files.file_r_e_2):
        missing_files.append(files.file_r_e_2)
    if not os.path.exists(files.file_r_f_1):
        missing_files.append(files.file_r_f_1)
    if not os.path.exists(files.file_r_f_2):
        missing_files.append(files.file_r_f_2)
    if not os.path.exists(files.file_r_f_3):
        missing_files.append(files.file_r_f_3)

    if missing_files:
        if verbose:
            print(f"        Computing polyhedron vectors...")
        polyhedron_vectors = compute_polyhedron_vectors(vertices, faces, edges, files, verbose)
    else:
        if verbose:
            print(f"        Polyhedron vector files already exist. Loading...")
        e_e = _load_dat(files.file_e_e).astype(np.float64)
        n_f = _load_dat(files.file_n_f).astype(np.float64)
        n_f_e = _load_dat(files.file_n_f_e).astype(np.float64)
        n_fp_e = _load_dat(files.file_n_fp_e).astype(np.float64)
        r_e_1 = _load_dat(files.file_r_e_1).astype(np.float64)
        r_e_2 = _load_dat(files.file_r_e_2).astype(np.float64)
        r_f_1 = _load_dat(files.file_r_f_1).astype(np.float64)
        r_f_2 = _load_dat(files.file_r_f_2).astype(np.float64)
        r_f_3 = _load_dat(files.file_r_f_3).astype(np.float64)
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

    return {
        "vertices": vertices,
        "faces": faces,
        "edges": edges,
        "centroid_faces": c_faces,
        "centroid_edges": c_edges,
        "e_e": polyhedron_vectors['e_e'],
        "n_f": polyhedron_vectors['n_f'],
        "n_f_e": polyhedron_vectors['n_f_e'],
        "n_fp_e": polyhedron_vectors['n_fp_e'],
        "r_e_1": polyhedron_vectors['r_e_1'],
        "r_e_2": polyhedron_vectors['r_e_2'],
        "r_f_1": polyhedron_vectors['r_f_1'],
        "r_f_2": polyhedron_vectors['r_f_2'],
        "r_f_3": polyhedron_vectors['r_f_3'],
        "files": asdict(files),
    }


if __name__ == "__main__":
    # current working directory
    current_path = os.getcwd()

    p = argparse.ArgumentParser(description="Polyhedral preparation pipeline")
    p.add_argument("--asteroid", type=str, default="Apophis")
    p.add_argument("--base_dir", type=str, default="../../Data")
    p.add_argument("--verbose", type=bool, default=True, help="If True, prints progress information.")
    args = p.parse_args()

    data = prepare_polyhedral_model(
        asteroid=args.asteroid,
        base_dir=args.base_dir,
        verbose=args.verbose,
    )

