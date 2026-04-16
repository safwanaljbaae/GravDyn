from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Dict
from pathlib import Path
import jax.numpy as jax_np
from typing import Sequence
from dataclasses import dataclass

from gravdyn.plot_tools import plot_layers_by_density, plot_layer_intersections
from gravdyn.constants import GRAVITATIONAL_CONSTANT

@dataclass
class PolyFiles:
    base_dir: str = "DATA"
    asteroid: str = "BENNU"

    @property
    def root(self) -> str:
        return os.path.join(self.base_dir, self.asteroid)

    @property
    def file_vertices(self) -> str:
        return os.path.join(self.root, "modified_v.dat")

    @property
    def file_faces(self) -> str:
        return os.path.join(self.root, "modified_f.dat")


def generate_layered_mascons(
    base_dir: str,
    asteroid: str,
    total_mass: float,
    densities: Sequence[float],
    output_csv: str | Path = "layered_mascons.csv",
) -> pd.DataFrame:
    """
    Generate a layered mascon model from a triangulated polyhedron.

    The polyhedron is assumed to be represented by triangular faces. Each face,
    together with the origin, defines one tetrahedron. Each tetrahedron is
    subdivided radially into n layers, where n = len(densities). One point mass
    is assigned to each layer using the exact center of mass of the layer
    obtained as the difference between two similar tetrahedra.

    The masses are first computed from the relative/absolute density values in
    `densities`, then rescaled so that the final sum of all point masses is
    exactly equal to `total_mass`.

    Parameters
    ----------
    base_dir : str
        Path to the directory containing asteroid shape model files.
    asteroid : str
        Name of the asteroid model (used to locate its folder in `base_dir`).
    total_mass : float
        Total mass of the asteroid [kg].
    densities : sequence of float
        Density values for the layers. Its length defines the number of layers.
        These values control the relative mass distribution among layers.
    output_csv : str or Path, optional
        Path to the CSV file where the mascon coordinates and masses will be saved.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        ['x', 'y', 'z', 'mass', 'face_id', 'layer_id', 'density_input']

    Notes
    -----
    1. This method assumes the origin lies inside the body and that the face
       triangulation consistently represents the polyhedron.
    2. The input densities do not need to integrate to the provided total mass.
       The function rescales all mascon masses at the end so that their sum is
       exactly `total_mass`.
    3. If the density list is uniform, this produces a layered discretization
       of a homogeneous body.
    """
    files = PolyFiles(base_dir=base_dir, asteroid=asteroid)

    missing_files = []
    if not os.path.exists(files.file_vertices):
        missing_files.append(files.file_vertices)
    if not os.path.exists(files.file_faces):
        missing_files.append(files.file_faces)

    if missing_files:
        raise FileNotFoundError(
            f"Shape files do not exist: {missing_files}.\n"
            f"Please ensure that you run the 'shape_verification' function beforehand."
        )

    output_csv = Path(output_csv)

    vertices = np.loadtxt(files.file_vertices, dtype=float)
    faces = np.loadtxt(files.file_faces, dtype=int)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices_file must contain an array of shape (N, 3).")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces_file must contain an array of shape (M, 3).")

    densities = np.asarray(densities, dtype=float)
    if densities.ndim != 1 or len(densities) == 0:
        raise ValueError("densities must be a non-empty 1D sequence.")
    if np.any(densities < 0):
        raise ValueError("densities must be non-negative.")
    if not np.any(densities > 0):
        raise ValueError("at least one density must be strictly positive.")
    if total_mass <= 0:
        raise ValueError("total_mass must be positive.")

    # Convert 1-based faces to 0-based if needed
    if faces.min() == 1:
        faces = faces - 1

    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("faces contain invalid vertex indices.")

    n_layers = len(densities)

    def tetra_volume(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Signed volume of tetrahedron (0, a, b, c)."""
        return abs(np.dot(a, np.cross(b, c))) / 6.0

    def layer_center_of_mass(a: np.ndarray, b: np.ndarray, c: np.ndarray, s0: float, s1: float) -> np.ndarray:
        """
        Center of mass of the layer between two similar tetrahedra scaled by s0 and s1.
        """
        face_centroid_tetra = (a + b + c) / 4.0

        # Exact factor from difference of two similar tetrahedra
        denom = s1**3 - s0**3
        if np.isclose(denom, 0.0):
            raise ValueError("Degenerate layer encountered: s1^3 - s0^3 is zero.")

        factor = (s1**4 - s0**4) / denom
        return factor * face_centroid_tetra

    rows = []
    raw_total_mass = 0.0

    for face_id, (i, j, k) in enumerate(faces):
        a = vertices[i]
        b = vertices[j]
        c = vertices[k]

        vol_full = tetra_volume(a, b, c)
        if np.isclose(vol_full, 0.0):
            continue

        for layer_id in range(n_layers):
            s0 = layer_id / n_layers
            s1 = (layer_id + 1) / n_layers

            # Volume of this radial layer from similarity scaling
            layer_volume = vol_full * (s1**3 - s0**3)

            # Raw mass from given density
            m_raw = densities[layer_id] * layer_volume

            if np.isclose(m_raw, 0.0):
                continue

            r_cm = layer_center_of_mass(a, b, c, s0, s1)

            rows.append(
                {
                    "x": r_cm[0],
                    "y": r_cm[1],
                    "z": r_cm[2],
                    "mass_raw": m_raw,
                    "face_id": face_id,
                    "layer_id": layer_id + 1,
                    "density_input": densities[layer_id],
                }
            )
            raw_total_mass += m_raw

    if len(rows) == 0:
        raise ValueError("No valid mascon points were generated. Check the input geometry.")

    if np.isclose(raw_total_mass, 0.0):
        raise ValueError("Computed raw total mass is zero. Check densities and geometry.")

    df = pd.DataFrame(rows)

    # Rescale masses so that the sum matches the requested total mass exactly
    scale_factor = total_mass / raw_total_mass
    df["mass"] = df["mass_raw"] * scale_factor
    df = df[["x", "y", "z", "mass", "face_id", "layer_id", "density_input"]]

    # Force exact total mass by correcting the last point for roundoff
    mass_error = total_mass - df["mass"].sum()
    df.loc[df.index[-1], "mass"] += mass_error

    df['mu'] = [x * GRAVITATIONAL_CONSTANT for x in df["mass"]]
    output_csv = os.path.join(base_dir, asteroid, output_csv)
    df.to_csv(output_csv, index=False)

    output_file = os.path.join(base_dir, asteroid, 'layered_mascons.png')
    plot_layers_by_density(df, output_file=output_file)
    output_file = os.path.join(base_dir, asteroid, 'layered_mascons_intersections.png')
    plot_layer_intersections(df, output_file=output_file, point_size=8)

    # print(f"Mascon file saved to: {output_csv}")
    # print(f"Number of faces: {len(faces)}")
    # print(f"Number of mascon points: {len(df)}")
    # print(f"Requested total mass : {total_mass:.16e} kg")
    # print(f"Computed total mass  : {df['mass'].sum():.16e} kg")
    # print(f"Mass difference      : {df['mass'].sum() - total_mass:.16e} kg")

    return df


def load_tetrahedron_data(
        base_dir: str,
        asteroid : str,
        tetrahedron_data_file: str,
) -> Dict[str, jax_np.ndarray]:
    """
    Load tetrahedron center data from a .dat file and convert it to JAX arrays.

    Parameters
    ----------
    base_dir : str
        Path to the directory containing asteroid shape model files.
    asteroid : str
        Name of the asteroid model (used to locate its folder in `base_dir`).
    tetrahedron_data_file : str, optional
        file containing the tetrahedron data.
        The file is expected to have at least 4 columns:
        [x, y, z, m] in this order.

    Returns
    -------
    dict
        Dictionary with keys 'm', 'x', 'y', 'z' mapped to JAX arrays.
    """
    filename = os.path.join(base_dir, asteroid, tetrahedron_data_file)

    # Check if the file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Mascon files do not exist: {filename}.\n"
            f"Please ensure that you run the 'generate_layered_mascons' function beforehand."
        )
    try:
        data_df = pd.read_csv(filename)

        data_shape = {
            'mu': jax_np.array(data_df['mu'].values, dtype=jax_np.float64),
            'x': jax_np.array(data_df['x'].values, dtype=jax_np.float64),
            'y': jax_np.array(data_df['y'].values, dtype=jax_np.float64),
            'z': jax_np.array(data_df['z'].values, dtype=jax_np.float64),
        }

        return data_shape

    except Exception as e:
        raise RuntimeError(f"Failed to load or process file '{filename}': {e}")


