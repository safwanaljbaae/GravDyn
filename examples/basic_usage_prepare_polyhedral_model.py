
from __future__ import annotations
import argparse
import os

from gravdyn import prepare_polyhedral_model

def main() -> None:
    # current working directory
    current_path = os.getcwd()
    print(current_path)

    p = argparse.ArgumentParser(description="Polyhedral preparation pipeline")
    p.add_argument("--asteroid", type=str, default="Apophis")
    p.add_argument("--base_dir", type=str, default="../Data")
    p.add_argument("--verbose", type=bool, default=True, help="If True, prints progress information.")
    args = p.parse_args()

    data = prepare_polyhedral_model(
        asteroid=args.asteroid,
        base_dir=args.base_dir,
        verbose=args.verbose,
    )
    print(data)
    exit()
    prepare_polyhedral_model(asteroid_name, base_dir=current_path)


if __name__ == "__main__":
    main()