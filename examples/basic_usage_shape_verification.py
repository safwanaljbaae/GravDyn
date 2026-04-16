
from __future__ import annotations
import os
from gravdyn import shape_verification


def main() -> None:

    # current working directory
    current_path = os.getcwd()

    print("Current path:", current_path)

    asteroid_name = "Apophis"
    mass = 5.31e10
    density = 1.75e0
    base_dir = '../Data/'
    vertices_file = "shape_v.dat"
    faces_file = "shape_f.dat"

    shape_verification(asteroid_name, mass, density, base_dir,
                       vertices_file, faces_file)


if __name__ == "__main__":
    main()