
from __future__ import annotations
import os
from gravdyn import generate_layered_mascons


def main() -> None:

    # current working directory
    current_path = os.getcwd()

    print("Current path:", current_path)
    asteroid = "Apophis"
    mass = 5.31e10
    densities = [1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75]
    base_dir = '../Data/'

    df_mascons = generate_layered_mascons(
        base_dir=base_dir,
        asteroid=asteroid,
        total_mass=mass,
        densities=densities,
        output_csv="layered_mascons.csv",
    )
    print(df_mascons)

if __name__ == "__main__":
    main()