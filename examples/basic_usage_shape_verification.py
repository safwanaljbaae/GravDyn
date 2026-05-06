
from __future__ import annotations
import os
from gravdyn import shape_verification


def main() -> None:

    # current working directory
    current_path = os.getcwd()

    print("Current path:", current_path)

    asteroid_data = {
        "Apophis":
            {
                "mass": 5.31e10,
                "density": 1.75e0
            },
        "Bennu":
            {
                "mass": 7.793e10,
                "density": 1.25e0
             },
        "Sylvia":
            {
                "mass": 1.4692e19,
                "density": 1.373e0
            },
        "Lutetia":
            {
                "mass": 1.68e18,
                "density": 3.4e0
             },
        "Betulia":
            {
                "mass": 1.64e14,
                "density": 2.00e0
            }
    }

    for asteroid_name in asteroid_data.keys():
        print(asteroid_name)
        mass = asteroid_data[asteroid_name]["mass"]
        density = asteroid_data[asteroid_name]["density"]

        base_dir = '../Data/'
        vertices_file = "shape_v.dat"
        faces_file = "shape_f.dat"
        shape_verification(asteroid_name, mass, density, base_dir,
                           vertices_file, faces_file)


if __name__ == "__main__":
    main()