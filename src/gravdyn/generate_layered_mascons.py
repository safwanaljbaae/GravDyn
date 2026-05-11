from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pathlib import Path
import jax.numpy as jax_np
from typing import Sequence, Dict

from gravdyn.polyhedral_model.poly_files import PolyFiles
from gravdyn.plot_tools import plot_layers_by_density, plot_layer_intersections
from gravdyn.constants import GRAVITATIONAL_CONSTANT


def generate_layered_mascons(
    base_dir: str,
    asteroid: str,
    total_mass: float,
    densities: Sequence[float],
    output_csv: str | Path = "layered_mascons.csv",
) -> pd.DataFrame:
    """
    Generate a layered mascon model from a triangulated polyhedron.

    This function replicates the behavior of fit_polyhedron_nlayer.f90.
    Each face with origin defines a tetrahedron, subdivided into n layers
    (n = len(densities)). Point masses use approximate centers of mass.

    Parameters
    ----------
    base_dir : str
        Path to directory containing asteroid shape model files.
    asteroid : str
        Name of the asteroid model.
    total_mass : float
        Total mass of the asteroid [kg].
    densities : sequence of float
        Density values for layers [g/cm³]. Length defines number of layers.
    output_csv : str or Path, optional
        Output CSV file path.

    Returns
    -------
    pandas.DataFrame
        Columns: ['x', 'y', 'z', 'mass', 'face_id', 'layer_id', 'density_input', 'mu']
    """
    files = PolyFiles(base_dir=base_dir, asteroid=asteroid)

    missing_files = []
    if not os.path.exists(files.file_vertices):
        missing_files.append(files.file_vertices)
    if not os.path.exists(files.file_faces):
        missing_files.append(files.file_faces)

    if missing_files:
        raise FileNotFoundError(f"Shape files do not exist: {missing_files}.")

    vertices = np.loadtxt(files.file_vertices, dtype=float)
    faces = np.loadtxt(files.file_faces, dtype=int)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be shape (N, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be shape (M, 3).")

    densities = np.asarray(densities, dtype=float)
    if densities.ndim != 1 or len(densities) == 0:
        raise ValueError("densities must be non-empty 1D.")
    if np.any(densities < 0):
        raise ValueError("densities must be non-negative.")
    if not np.any(densities > 0):
        raise ValueError("at least one density must be positive.")
    if total_mass <= 0:
        raise ValueError("total_mass must be positive.")

    if faces.min() == 1:
        faces = faces - 1

    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("faces contain invalid vertex indices.")

    n_layers = len(densities)
    densities_scaled = densities * 1.0e12  # g/cm³ to kg/km³

    def tetra_volume(a, b, c):
        """Volume of tetrahedron (0, a, b, c)."""
        return abs(np.dot(a, np.cross(b, c))) / 6.0

    rows = []

    for face_id, (i, j, k) in enumerate(faces):
        a = vertices[i]
        b = vertices[j]
        c = vertices[k]

        # Layered decomposition (match Fortran EXACTLY)
        part_x = np.zeros((n_layers, 3))
        part_y = np.zeros((n_layers, 3))
        part_z = np.zeros((n_layers, 3))

        n_part = n_layers
        for ii in range(n_layers):
            scale = n_part / n_layers  # Fortran: REAL(n_part, 8) / REAL(nlayer, 8)
            part_x[n_part-1] = [a[0] * scale, b[0] * scale, c[0] * scale]
            part_y[n_part-1] = [a[1] * scale, b[1] * scale, c[1] * scale]
            part_z[n_part-1] = [a[2] * scale, b[2] * scale, c[2] * scale]
            n_part -= 1

        # Compute volumes for each layer (match Fortran vol subroutine)
        vol_tetr_all = np.zeros(n_layers)
        for ii in range(n_layers):
            # Fortran indices are 1-based, Python are 0-based
            v1 = np.array([part_x[ii][0], part_y[ii][0], part_z[ii][0]])
            v2 = np.array([part_x[ii][1], part_y[ii][1], part_z[ii][1]])
            v3 = np.array([part_x[ii][2], part_y[ii][2], part_z[ii][2]])
            vol_tetr_all[ii] = np.dot(v1, np.cross(v2, v3)) / 6.0

        # Compute mu for all layers first (match Fortran)
        mu_layers = np.zeros(n_layers)
        for ii in range(n_layers):
            if ii == 0:
                # Layer 1 (innermost)
                mu_layers[ii] = GRAVITATIONAL_CONSTANT * vol_tetr_all[ii] * densities_scaled[ii]
            else:
                # Layers 2..nlayer
                vol_tetr = vol_tetr_all[ii] - vol_tetr_all[ii-1]
                mu_layers[ii] = GRAVITATIONAL_CONSTANT * vol_tetr * densities_scaled[ii]

        # Write output in Fortran order: layers 2..nlayer first, then layer 1
        for ii in range(1, n_layers):
            vol_tetr = vol_tetr_all[ii] - vol_tetr_all[ii-1]
            xc = (part_x[ii][0] + part_x[ii][1] + part_x[ii][2] +
                  part_x[ii-1][0] + part_x[ii-1][1] + part_x[ii-1][2]) / 6.0
            yc = (part_y[ii][0] + part_y[ii][1] + part_y[ii][2] +
                  part_y[ii-1][0] + part_y[ii-1][1] + part_y[ii-1][2]) / 6.0
            zc = (part_z[ii][0] + part_z[ii][1] + part_z[ii][2] +
                  part_z[ii-1][0] + part_z[ii-1][1] + part_z[ii-1][2]) / 6.0

            rows.append({
                "x": xc, "y": yc, "z": zc,
                "mass": mu_layers[ii] / GRAVITATIONAL_CONSTANT,
                "face_id": face_id + 1,
                "layer_id": ii + 1,
                "density_input": densities[ii],
                "mu": mu_layers[ii],
            })

        # Layer 1 (innermost) - after layers 2..nlayer (match Fortran order)
        xc = (part_x[0][0] + part_x[0][1] + part_x[0][2]) / 4.0
        yc = (part_y[0][0] + part_y[0][1] + part_y[0][2]) / 4.0
        zc = (part_z[0][0] + part_z[0][1] + part_z[0][2]) / 4.0

        rows.append({
            "x": xc, "y": yc, "z": zc,
            "mass": mu_layers[0] / GRAVITATIONAL_CONSTANT,
            "face_id": face_id + 1,
            "layer_id": 1,
            "density_input": densities[0],
            "mu": mu_layers[0],
        })

    df = pd.DataFrame(rows)

    # --- Adjust masses to match total_mass as close as possible ---
    current_mass = df['mass'].sum()
    mass_diff = current_mass - total_mass

    # If mass difference is significant, scale all masses proportionally
    # This preserves the density distribution while matching total mass
    tolerance = 1e-12 * total_mass  # Relative tolerance
    if abs(mass_diff) > tolerance:
        scaling_factor = total_mass / current_mass
        df['mass'] = df['mass'] * scaling_factor
        df['mu'] = df['mass'] * GRAVITATIONAL_CONSTANT

    output_csv_full = Path(base_dir) / asteroid / output_csv
    output_csv_full.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_full, float_format="%.8e", index=False)

    output_file = os.path.join(base_dir, asteroid, 'layered_mascons.png')
    plot_layers_by_density(df, output_file=output_file)
    output_file = os.path.join(base_dir, asteroid, 'layered_mascons_intersections.png')
    plot_layer_intersections(df, output_file=output_file, point_size=8)

    print(f"    Mascon file saved to: {output_csv_full}")
    print(f"    Number of faces: {len(faces)}")
    print(f"    Number of mascon points: {len(df)}")
    print(f"    Requested total mass : {total_mass:.16e} kg")
    print(f"    Computed total mass  : {df['mass'].sum():.16e} kg")
    print(f"    Mass difference      : {df['mass'].sum() - total_mass:.16e} kg")

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

