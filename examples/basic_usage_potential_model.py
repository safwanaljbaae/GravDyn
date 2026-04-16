from __future__ import annotations
import numpy as np
from gravdyn import prepare_polyhedral_model
from gravdyn import pot_point_mass, batched_polyhedral_potential, pot_expansion, batched_pot_mascon
from gravdyn import build_potential_derivatives
from gravdyn import load_tetrahedron_data

def main() -> None:

    gravitation = 6.674101262875753845e-20
    mass = 5.31e10
    mu = mass*gravitation
    asteroid = "Apophis"

    grid_radius = 3.0
    n_points = 100
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
    print(p[0])
    print(acc[0])

    print('Polyhedral')
    batch_size = 20000
    polyhedral_data = prepare_polyhedral_model(
        asteroid=asteroid,
        base_dir="../Data",
        verbose=False,
    )
    p, acc = batched_polyhedral_potential(gm_body=mu,
                                          stat=points,
                                          polyhedral_data=polyhedral_data,
                                          batch_size=batch_size)
    print(p[0])
    print(acc[0])

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
    print(p[0])
    print(acc[0])

    print('Mascon')
    data_shape = load_tetrahedron_data(
        asteroid=asteroid,
        base_dir="../Data",
        tetrahedron_data_file="layered_mascons.csv",
    )
    p, acc = batched_pot_mascon(points, data_shape, batch_size=20000)
    print(p[0])
    print(acc[0])




if __name__ == "__main__":
    main()