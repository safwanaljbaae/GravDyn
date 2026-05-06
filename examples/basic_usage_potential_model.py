from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from gravdyn import load_tetrahedron_data
from gravdyn import prepare_werner_model
from gravdyn import build_potential_derivatives
from gravdyn import pot_point_mass, batched_wrener_potential, pot_expansion, batched_pot_mascon
from gravdyn.plot_tools import compare_pot
from gravdyn import generate_layered_mascons


def save_potential_acceleration_csv(
    points: np.ndarray,
    potential: np.ndarray,
    acceleration: np.ndarray,
    output_csv: str | Path,
) -> None:
    """
    Save positions, radial distances, gravitational potential, and acceleration
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
        The function saves the data directly to `output_csv`.

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


def main() -> None:

    gravitation = 6.674101262875753845e-20

    asteroid = "1996_HW1"
    mass = 2.27e13
    densities = [1.727 for _ in range(40)]

    mu = mass*gravitation

    file_path = Path("../Data") / asteroid / "modified_v.dat"
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

    print('Point mass')
    p, acc = pot_point_mass(mu=mu, stat=points)
    save_potential_acceleration_csv(
        points=points,
        potential=p,
        acceleration=acc,
        output_csv=f"../Data/{asteroid}/pot_point_mass.csv",
    )

    print('Werner')
    batch_size = 20000
    polyhedral_data = prepare_werner_model(
        asteroid=asteroid,
        base_dir="../Data",
        verbose=False,
    )
    p, acc = batched_wrener_potential(gm_body=mu,
                                      stat=points,
                                      polyhedral_data=polyhedral_data,
                                      batch_size=batch_size)
    save_potential_acceleration_csv(
        points=points,
        potential=p,
        acceleration=acc,
        output_csv=f"../Data/{asteroid}/pot_Werner.csv",
    )

    print('Expansion')
    f_pot_expansion, f_d_pot_expansion = build_potential_derivatives(
        name_central_body=asteroid,
        pattern="pot_*.dat",
        n_files=700,
        gm0=mu,
        lambdify_backend="jax",
        base_dir="../Data",
        verbose=True,
    )
    p, acc = pot_expansion(
        stat=points,
        f_pot_expansion=f_pot_expansion,
        f_d_pot_expansion=f_d_pot_expansion,
    )
    save_potential_acceleration_csv(
        points=points,
        potential=p,
        acceleration=acc,
        output_csv=f"../Data/{asteroid}/pot_Expansion.csv",
    )

    print('Mascon')
    df_mascons = generate_layered_mascons(
        base_dir='../Data/',
        asteroid=asteroid,
        total_mass=mass,
        densities=densities,
        output_csv="layered_mascons.csv",
    )
    data_shape = load_tetrahedron_data(
        asteroid=asteroid,
        base_dir="../Data",
        tetrahedron_data_file="layered_mascons.csv",
    )
    p, acc = batched_pot_mascon(points, data_shape, batch_size=2000)
    save_potential_acceleration_csv(
        points=points,
        potential=p,
        acceleration=acc,
        output_csv=f"../Data/{asteroid}/pot_Mascon.csv",
    )

    compare_pot(asteroid, directory_path="../Data", compare_mascon=True, compare_expansion=True)

if __name__ == "__main__":
    main()