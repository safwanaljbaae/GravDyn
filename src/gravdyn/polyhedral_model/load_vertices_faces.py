import numpy as np

def _load_dat(path: str) -> np.ndarray:
    """
    Load a DAT file as a numpy array.
    """
    return np.loadtxt(path)


def load_vertices_faces(file_vertices, file_faces):
    """
    Load vertices (X Y Z per line) and faces (i j k per line, 1-based indexing) from DAT files.
    Returns:
        vertices: (N,3) float64
        faces   : (M,3) int64 (0-based indices)
    """
    V = _load_dat(file_vertices).astype(np.float64)
    F = _load_dat(file_faces).astype(np.int64)
    # Convert 1-based -> 0-based if needed
    if F.min() == 0:
        F = F + 1
    return V, F
