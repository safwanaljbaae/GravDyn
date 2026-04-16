from pathlib import Path
import numpy as np

def load_vertices(vertices_file: str) -> np.ndarray:
    """
    Load mesh vertex coordinates from a text file.

    Parameters
    ----------
    vertices_file : str
        Path to a plain-text file containing mesh vertices. The file is
        expected to store one vertex per row with exactly three floating-point
        columns corresponding to Cartesian coordinates ``x``, ``y``, and ``z``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, 3)`` containing the mesh vertex coordinates as
        floating-point values, where ``N`` is the number of vertices. If the
        file contains a single vertex, the returned array is reshaped to
        ``(1, 3)``.

    Raises
    ------
    FileNotFoundError
        If ``vertices_file`` does not exist.
    ValueError
        If ``vertices_file`` is not a regular file or if the loaded data does
        not have exactly three columns.

    Notes
    -----
    This function is intended for triangular surface meshes used in scientific
    computing and astrodynamics applications, where vertices define the
    Cartesian geometry of a body-fixed shape model.

    """
    path = Path(vertices_file)

    if not path.exists():
        raise FileNotFoundError(f"Vertices file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Vertices path is not a file: {path}")

    vertices = np.loadtxt(path, dtype=float)

    # If file has only one vertex, force shape (1, 3)
    vertices = np.atleast_2d(vertices)

    if vertices.shape[1] != 3:
        raise ValueError(
            f"Vertices file must have exactly 3 columns (x y z). Found shape: {vertices.shape}"
        )

    return vertices


def load_faces(faces_file: str) -> np.ndarray:
    """
    Load triangular mesh face connectivity from a text file.

    Parameters
    ----------
    faces_file : str
        Path to a plain-text file containing mesh face definitions. The file is
        expected to store one triangular face per row with exactly three
        integer columns corresponding to vertex indices ``(i, j, k)``.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(M, 3)`` containing triangular face
        connectivity, where ``M`` is the number of faces. If the file contains
        a single face, the returned array is reshaped to ``(1, 3)``. If the
        input indexing is one-based, it is converted in place to zero-based
        indexing before returning.

    Raises
    ------
    FileNotFoundError
        If ``faces_file`` does not exist.
    ValueError
        If ``faces_file`` is not a regular file or if the loaded data does
        not have exactly three columns.

    Notes
    -----
    The returned connectivity is suitable for mesh structures in which rows of
    ``faces`` reference rows of a corresponding vertex array of shape
    ``(N, 3)``. The function assumes triangular faces only.

    """
    path = Path(faces_file)

    if not path.exists():
        raise FileNotFoundError(f"Faces file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Faces path is not a file: {path}")

    faces = np.loadtxt(path, dtype=int)

    # If file has only one face, force shape (1, 3)
    faces = np.atleast_2d(faces)

    if faces.shape[1] != 3:
        raise ValueError(
            f"Faces file must have exactly 3 columns (i j k). Found shape: {faces.shape}"
        )

    if min([f[0] for f in faces]) == 1:
        faces -= 1
    return faces
