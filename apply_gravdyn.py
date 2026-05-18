from __future__ import annotations
import os
import time
import trimesh
import gravdyn
import numpy as np
import pandas as pd
import jax.numpy as jnp
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def save_potential_acceleration_csv(
    points: np.ndarray,
    potential: np.ndarray,
    acceleration: np.ndarray,
    output_csv: str | Path,
) -> None:
    """Save positions, radial distances, gravitational potential, and acceleration
    components to a CSV file.

    Parameters
    ----------
    points : np.ndarray
        Array with shape (N, 3) containing the Cartesian coordinates of the
        evaluation points. The columns must correspond to x, y, and z.
    potential : np.ndarray
        Array with shape (N,) containing the potential value computed at each
        point.
    acceleration : np.ndarray
        Array with shape (N, 3) containing the acceleration components at each
        point. The columns must correspond to ax, ay, and az.
    output_csv : str or pathlib.Path
        Path where the CSV file will be saved.

    Returns
    -------
    None
        The function saves the data directly to ``output_csv``.

    Raises
    ------
    ValueError
        If the input arrays do not have compatible shapes.
    """

    points = np.asarray(points, dtype=float)
    potential = np.asarray(potential, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")

    if acceleration.ndim != 2 or acceleration.shape[1] != 3:
        raise ValueError("acceleration must have shape (N, 3).")

    if potential.ndim != 1:
        raise ValueError("potential must have shape (N,).")

    if not (len(points) == len(potential) == len(acceleration)):
        raise ValueError(
            "points, potential, and acceleration must have the same number of rows."
        )

    r = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)

    df = pd.DataFrame({
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
        "r": r,
        "potential": potential,
        "ax": acceleration[:, 0],
        "ay": acceleration[:, 1],
        "az": acceleration[:, 2],
    })

    df.to_csv(output_csv, index=False)


def format_time(seconds: float) -> str:
    """Format a time duration in seconds to HH:MM:SS.fff.

    Parameters
    ----------
    seconds : float
        Time duration in seconds.

    Returns
    -------
    str
        Formatted string in ``HH:MM:SS.fff`` format.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def compare_pot(
    asteroid: str,
    directory_path: str,
    compare_mascon: bool = True,
    compare_expansion: bool = True,
) -> None:
    """Compare mascon and/or spherical-harmonic expansion potential models
    against the Werner polyhedral reference, and save a relative-error scatter
    plot.

    Reads the pre-computed Werner, Mascon, and Expansion CSV files, computes
    the relative percent difference ``(pot_model - pot_Werner) / pot_Werner *
    100`` for each requested model, and plots the results against radial
    distance. The Brillouin radius is computed from the shape mesh and marked
    on the plot.  The figure is saved as ``d_pot.png`` in the asteroid data
    directory.

    Parameters
    ----------
    asteroid : str
        Name of the asteroid (used to locate data files).
    directory_path : str
        Base directory that contains the asteroid subdirectory with CSV and
        shape files.
    compare_mascon : bool, optional
        If ``True``, compare the mascon model against Werner (default
        ``True``).
    compare_expansion : bool, optional
        If ``True``, compare the spherical-harmonic expansion model against
        Werner (default ``True``).

    Raises
    ------
    ValueError
        If the point arrays in the CSV files differ in shape or coordinate
        values.
    """

    # Build file paths using directory_path
    werner_file = f"{directory_path}/{asteroid}/pot_Werner.csv"
    mascon_file = f"{directory_path}/{asteroid}/pot_Mascon.csv"
    expansion_file = f"{directory_path}/{asteroid}/pot_Expansion.csv"

    faces_file = f"{directory_path}/{asteroid}/modified_f.dat"
    vertices_file = f"{directory_path}/{asteroid}/modified_v.dat"

    # --- Read Werner (reference)
    df_werner = pd.read_csv(werner_file)
    xyz_werner = jnp.asarray(df_werner[["x", "y", "z"]].values)
    pot_werner = jnp.asarray(df_werner["potential"].values)
    r = jnp.asarray(df_werner["r"].values)

    plt.figure()

    # --- Compare with Mascon if requested
    if compare_mascon:
        df_mascon = pd.read_csv(mascon_file)
        xyz_mascon = jnp.asarray(df_mascon[["x", "y", "z"]].values)
        pot_mascon = jnp.asarray(df_mascon["potential"].values)

        # Check shape
        if xyz_werner.shape != xyz_mascon.shape:
            raise ValueError("Werner and Mascon files have different number of points")

        # Check equality (with tolerance)
        same = jnp.all(jnp.isclose(xyz_werner, xyz_mascon, atol=1e-12))
        if not bool(same):
            raise ValueError("x, y, z columns are not identical between Werner and Mascon")

        # Compute difference relative to Werner
        d_pot_mascon = (pot_mascon - pot_werner) * 100 / pot_werner

        plt.scatter(r, d_pot_mascon, marker='.', color='blue', s=2)

    # --- Compare with Expansion if requested
    if compare_expansion:
        df_expansion = pd.read_csv(expansion_file)
        xyz_expansion = jnp.asarray(df_expansion[["x", "y", "z"]].values)
        pot_expansion = jnp.asarray(df_expansion["potential"].values)

        # Check shape
        if xyz_werner.shape != xyz_expansion.shape:
            raise ValueError("Werner and Expansion files have different number of points")

        # Check equality (with tolerance)
        same = jnp.all(jnp.isclose(xyz_werner, xyz_expansion, atol=1e-12))
        if not bool(same):
            raise ValueError("x, y, z columns are not identical between Werner and Expansion")

        # Compute difference relative to Werner
        d_pot_expansion = (pot_expansion - pot_werner) * 100 / pot_werner

        plt.scatter(r, d_pot_expansion, marker='.', color='green', s=2)

    # --- Read mesh for Brillouin radius
    faces = pd.read_table(
        faces_file,
        skiprows=0,
        header=None,
        sep=r"\s+",
        index_col=None,
        names=["f1", "f2", "f3"],
        low_memory=False,
        encoding="unicode_escape")

    vertices = pd.read_table(
        vertices_file,
        skiprows=0,
        header=None,
        sep=r"\s+",
        index_col=None,
        names=["x", "y", "z"],
        low_memory=False,
        encoding="unicode_escape",
    )

    plt.scatter(0, 100, marker='.', color='blue', s=20, label="Mascon vs Werner")
    plt.scatter(0, 100, marker='.', color='green', s=20, label="Expansion vs Werner")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    distances = cdist(mesh.vertices, np.array([mesh.center_mass]))
    R_brillouin = max(distances)[0]

    # --- Plot setup
    plt.axvline(x=R_brillouin, color='red', label='Brillouin radius')

    plt.xlabel(r"$r(\text{km})$")
    plt.ylabel("Relative Error (%)")

    plt.xlim(min(r), max(r))
    y_max = 2e-1
    plt.ylim(-y_max, y_max)

    plt.legend(loc='upper center', bbox_to_anchor=(0.45, 1.15), fancybox=True, shadow=True, ncol=6)

    plt.savefig(f"{directory_path}/{asteroid}/d_pot.png", dpi=300)
    plt.close()


def main() -> None:
    """Run the full gravitational potential and acceleration analysis for
    asteroid 99942 Apophis using four models.

    The workflow performs the following steps in order:

    1. Shape-model verification via ``gravdyn.shape_verification``.
    2. Point-mass potential and acceleration.
    3. Werner polyhedral potential and acceleration (batched).
    4. Spherical-harmonic expansion potential and acceleration.
    5. Layered-mascon potential and acceleration (batched).
    6. Comparison plot of all models relative to the Werner reference.

    Results are saved as CSV files and a ``d_pot.png`` figure in
    ``Data/<asteroid>/``.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # current working directory
    current_path = os.getcwd()

    gravitation = gravdyn.constants.GRAVITATIONAL_CONSTANT

    print("Current path:", current_path)

    asteroid = "Apophis"
    mass = 5.31e10
    density = 1.75e0

    asteroid="Bennu"
    mass=7.793e10
    density=1.25e0

    asteroid="Sylvia"
    mass=1.4692e19
    density=1.373e0

    asteroid="Lutetia"
    mass=1.68e18
    density=3.4e0

    asteroid="1996_HW1"
    mass=2.27e13
    density=1.727e0

    asteroid="1998_QE2"
    mass=1.12e13
    density=0.7e0

    asteroid="Arrokoth"
    mass=1.0e8
    density=0.50

    asteroid="Eros"
    mass=6.689e15
    density=2.675e0

    # asteroid="Itokawa"
    # mass=3.524e10
    # density=1.98e0

    # asteroid="Phaeyhon"
    # mass=2.27e13
    # density=1.727e0

    densities = [density for _ in range(20)]


    base_dir = 'Data/'
    vertices_file = "shape_v.dat"
    faces_file = "shape_f.dat"

    # gravdyn.shape_verification(asteroid, mass, density, base_dir,
    #                    vertices_file, faces_file)


    file_path = Path(base_dir) / asteroid / "modified_v.dat"
    vertices = np.loadtxt(file_path)
    distances = np.linalg.norm(vertices[:, :3], axis=1)
    max_dist = distances.max()

    grid_radius = 5*max_dist
    n_points = 200
    z_value = 0.0

    x_vals = np.linspace(-grid_radius, grid_radius, n_points)
    y_vals = np.linspace(-grid_radius, grid_radius, n_points)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.full_like(X, z_value)

    points = np.column_stack([
        X.ravel(),
        Y.ravel(),
        Z.ravel(),
    ])


    # print('Point mass')
    # start = time.perf_counter()
    # p, acc = gravdyn.pot_point_mass(mu=mass*gravitation, stat=points)
    # end = time.perf_counter()
    # elapsed = end - start
    # print("    Execution time:", format_time(elapsed))
    # save_potential_acceleration_csv(
    #     points=points,
    #     potential=p,
    #     acceleration=acc,
    #     output_csv=f"{base_dir}/{asteroid}/pot_point_mass.csv",
    # )

    # print('Werner')
    # batch_size = 2000
    # polyhedral_data = gravdyn.prepare_werner_model(
    #     asteroid=asteroid,
    #     base_dir=base_dir,
    #     verbose=True,
    # )
    start = time.perf_counter()
    # p, acc = gravdyn.batched_werner_potential(gm_body=mass*gravitation,
    #                                   stat=points,
    #                                   polyhedral_data=polyhedral_data,
    #                                   batch_size=batch_size)
    # end = time.perf_counter()
    # elapsed = end - start
    # print("    Execution time:", format_time(elapsed))
    # save_potential_acceleration_csv(
    #     points=points,
    #     potential=p,
    #     acceleration=acc,
    #     output_csv=f"{base_dir}/{asteroid}/pot_Werner.csv",
    # )

    # print('Mascon')
    # df_mascons = gravdyn.generate_layered_mascons(
    #     base_dir=base_dir,
    #     asteroid=asteroid,
    #     total_mass=mass,
    #     densities=densities,
    #     output_csv="layered_mascons.csv",
    # )
    # data_shape = gravdyn.load_tetrahedron_data(
    #     asteroid=asteroid,
    #     base_dir=base_dir,
    #     tetrahedron_data_file="layered_mascons.csv",
    # )
    # start = time.perf_counter()
    # p, acc = gravdyn.batched_pot_mascon(points, data_shape, batch_size=batch_size)
    # save_potential_acceleration_csv(
    #     points=points,
    #     potential=p,
    #     acceleration=acc,
    #     output_csv=f"{base_dir}/{asteroid}/pot_Mascon.csv",
    # )
    # end = time.perf_counter()
    # elapsed = end - start
    # print("    Execution time:", format_time(elapsed))


    print('Expansion')
    f_pot_expansion, f_d_pot_expansion = gravdyn.build_potential_derivatives(
        name_central_body=asteroid,
        pattern="pot_*.dat",
        n_files=700,
        gm0=mass*gravitation,
        lambdify_backend="jax",
        base_dir=base_dir,
        verbose=True,
    )
    start = time.perf_counter()
    p, acc = gravdyn.pot_expansion(
        stat=points,
        f_pot_expansion=f_pot_expansion,
        f_d_pot_expansion=f_d_pot_expansion,
    )
    end = time.perf_counter()
    elapsed = end - start
    print("    Execution time:", format_time(elapsed))

    save_potential_acceleration_csv(
        points=points,
        potential=p,
        acceleration=acc,
        output_csv=f"{base_dir}/{asteroid}/pot_Expansion.csv",
    )

    compare_pot(asteroid, directory_path=base_dir, compare_mascon=True, compare_expansion=True)


if __name__ == "__main__":
    main()
