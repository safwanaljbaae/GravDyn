
from __future__ import annotations
import argparse
import os

from gravdyn import build_potential_derivatives

def main() -> None:
    # current working directory
    current_path = os.getcwd()
    print(current_path)

    p = argparse.ArgumentParser(description="Polyhedral preparation pipeline")
    p.add_argument("--asteroid", type=str, default="Apophis")
    p.add_argument("--base_dir", type=str, default="../Data")
    p.add_argument("--verbose", type=bool, default=True, help="If True, prints progress information.")
    args = p.parse_args()

    gravitation = 6.674101262875753845e-20
    mass = 5.3099986439921903e10
    mu = mass*gravitation

    d_exprs, d_funcs = build_potential_derivatives(
        name_central_body=args.asteroid,
        gm0=mu,
        base_dir=args.base_dir,
        verbose=True,
    )
    print(d_exprs)
    exit()
    prepare_polyhedral_model(asteroid_name, base_dir=current_path)


if __name__ == "__main__":
    main()