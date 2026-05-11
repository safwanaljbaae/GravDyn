# -*- coding: utf-8 -*-
"""
# !===============================================================
# !==   Dr. Safwan Aljbaae                                      ==
# !==   October 2025                                            ==
# !===============================================================
# python3 -m pip install -r requirements.txt                    ==
# !===============================================================
"""

import glob
import pickle
import pandas as pd
import sympy as sym
from pathlib import Path
from typing import Callable, List, Tuple, Any
from gravdyn.github_drive_service import list_github_folder_files, download_github_folder
from gravdyn.constants import GRAVITATIONAL_CONSTANT


def _ensure_dir(p: Path) -> None:
    """
    Ensure that a directory exists, creating it (and all missing parents) if necessary.

    :param p: Filesystem path of the directory to create/verify.
    :return: None
    """

    p.mkdir(parents=True, exist_ok=True)


def _read_derivative_files(base_dir: Path, name_central_body: str) -> tuple[Any, list[Any]] | None:
    """
    Read previously saved symbolic partial derivatives dU/dx, dU/dy, and dU/dz from text files.

    The function looks for the files 'd_x.txt', 'd_y.txt', and 'd_z.txt' under
    base_dir/<name_central_body>/. If all are present, it parses their textual
    expressions into SymPy objects using the symbols x, y, z.

    :param base_dir: Root data directory (e.g., Path("Data")).
    :param name_central_body: Name of the central body (e.g., "Apophis").
    :return: list of SymPy expressions [dU/dx, dU/dy, dU/dz] if found; otherwise None.
    """

    dpaths = [
        base_dir / name_central_body / "pot.txt",
        base_dir / name_central_body / "d_x.txt",
        base_dir / name_central_body / "d_y.txt",
        base_dir / name_central_body / "d_z.txt",
    ]
    if not all(p.exists() for p in dpaths):
        return None
    x, y, z = sym.symbols("x y z")
    exprs = []
    for p in dpaths:
        txt = p.read_text(encoding="utf-8").strip()
        exprs.append(sym.sympify(txt, locals={"x": x, "y": y, "z": z}))

    return exprs[0], [exprs[i] for i in range(1,4)]


def _write_derivative_files(base_dir: Path, name_central_body: str, pot_expansion: sym.Expr, derivs: List[sym.Expr]) -> None:
    """
    Write symbolic partial derivatives dU/dx, dU/dy, and dU/dz to text files.

    The function writes one file per axis — 'd_x.txt', 'd_y.txt', 'd_z.txt' —
    in the folder base_dir/<name_central_body>/. Each file contains the string
    representation of the corresponding SymPy expression.

    :param base_dir: Root data directory (e.g., Path("Data")).
    :param name_central_body: Name of the central body (e.g., "Apophis").
    :param derivs: SymPy expressions of the potential.
    :param derivs: List of three SymPy expressions in the order [dU/dx, dU/dy, dU/dz].
    :return: None
    """
    
    bx = base_dir / name_central_body
    _ensure_dir(bx)
    for axis, expr in zip(["x", "y", "z"], derivs):
        (bx / f"d_{axis}.txt").write_text(str(expr), encoding="utf-8")

    (bx / f"pot.txt").write_text(str(pot_expansion), encoding="utf-8")

def generate_fortran_code(expression, function_name):
    """
    Emit a single-precision-compatible (REAL*8) Fortran 77 function for a given SymPy expression.

    The function generates fixed-form Fortran code for a function with signature
    `REAL*8 function <function_name>(x, y, z)`. The expression is split into
    50-character chunks and written with continuation lines to maintain Fortran 77 style.

    Files created:
      - <function_name>.for  (written in the current working directory)

    :param expression: SymPy expression to be compiled into Fortran source.
    :param function_name: Base name of the Fortran function (and output file, with '.for' extension).
    :return: None
    """

    fortran_code = sym.fcode(expression, standard=77).replace('@', '').replace(' ', '').replace('\n', '')
    chunk_size = 50
    fortran_code_lines = [fortran_code[i:i + chunk_size] for i in range(0, len(fortran_code), chunk_size)]

    # Function to add continuation character with proper spacing
    def add_continuation_lines(code_lines):
        result = []
        for i, line in enumerate(code_lines):
            result.append('     & ' + line.strip())
        return result

    # Apply continuation formatting to the split lines
    fortran_code_lines = add_continuation_lines(fortran_code_lines)

    # Write the formatted code to a Fortran file
    with open(f'{function_name}.for', 'w') as file:
        file.write(f"      REAL*8 function {function_name.split('/')[-1]}(x, y, z)\n")
        file.write('      REAL*8 x, y, z\n')
        file.write('      REAL*8 result\n')
        file.write('      result = \n')
        for line in fortran_code_lines:
            file.write(f'{line}\n')
        file.write(f"      {function_name.split('/')[-1]} = result\n")
        file.write('      return\n')
        file.write('      end\n')


def build_potential_derivatives(
    name_central_body: str = "Apophis",
    pattern: str = "pot_*.dat",
    n_files: int = 100,
    gm0: float = 1.0,
    lambdify_backend: str = "jax",
    *,
    base_dir: str = "Data",
    verbose: bool = True,
) -> Tuple[List[sym.Expr], List[Callable]]:
    """
    Build or load the gravitational potential derivatives dU/dx, dU/dy, dU/dz for a central body
    and return both their symbolic forms and fast numerical callables.

    This function supports two cases:

    1) If derivative files already exist in Data/<body>/:
         - Reads the symbolic expressions from d_x.txt, d_y.txt, d_z.txt.
         - Uses `sym.lambdify` to convert each expression into a fast numerical function.
           Lambdify takes a SymPy expression (symbolic math) and generates a regular Python
           function that can evaluate the expression numerically using backends such as
           NumPy, JAX, or Numba. This allows efficient evaluation during simulations
           (e.g., computing accelerations many times inside an ODE integrator).

    2) If no derivative files exist:
         - Loads and sums up to `n_files` partial potential pickle files matching `pattern`
           in Data/<body>/POT_EXPANSION/.
         - Substitutes constants (G=1, mass=gm0, r = sqrt(x²+y²+z²)).
         - Symbolically differentiates the total potential with respect to x, y, z.
         - Generates Fortran files (d_x.for, d_y.for, d_z.for) if missing.
         - Saves the symbolic expressions to d_x.txt, d_y.txt, d_z.txt.
         - Lambdifies each derivative expression to create fast numerical functions.

    The returned functions can be called as f(x, y, z) and will evaluate dU/dx, dU/dy, or dU/dz
    numerically using the specified backend (e.g., "jax").

    :param name_central_body: Name of the central body (e.g., "Apophis").
    :param pattern: Glob pattern for partial potential pickle files (e.g., "pot_*.dat").
    :param n_files: Maximum number of partial files to load and sum.
    :param gm0: Value to substitute for 'mass' in the potential (assuming G=1).
    :param lambdify_backend: Backend for `sym.lambdify` ("jax", "numpy", "numba", etc.).
    :param base_dir: Root data directory (expects POT_EXPANSION and derivative files inside).
    :param verbose: If True, prints progress information.

    :return: tuple containing:
        - derivative_exprs (list[sym.Expr]):
            Symbolic expressions [dU/dx, dU/dy, dU/dz].
        - derivative_funcs (list[Callable]):
            Numerical functions [fx(x,y,z), fy(x,y,z), fz(x,y,z)] generated via `lambdify`.
    """

    folder = Path(f'{base_dir}/{name_central_body}/Pot_Expansion')
    if not folder.exists():
        print(
            f"No potential data found locally at '{folder}'.\n"
            "We will check the GitHub repository of the package."
        )
        list_files = list_github_folder_files(
            owner="safwanaljbaae",
            repo="GravDyn",
            path=f"Data/{name_central_body}/Pot_Expansion",
            branch="feature-test"
        )
        files = download_github_folder(
            owner="safwanaljbaae",
            repo="GravDyn",
            path=f"Data/{name_central_body}/Pot_Expansion",
            branch="feature-test",
            output_dir=f'{base_dir}/{name_central_body}/Pot_Expansion'
        )

    data_root_path = Path(base_dir)
    body_dir = data_root_path / name_central_body
    pot_dir = body_dir / "Pot_Expansion"

    x, y, z = sym.symbols("x y z")
    g_sym, mass_sym, r_sym = sym.symbols("g mass r")

    base = Path(data_root_path) / name_central_body
    required = ["pot.for", "d_x.for", "d_y.for", "d_z.for"]
    missing = [f for f in required if not (base / f).is_file()]
    if not missing:
        # if verbose:
        #     print(f"    reading existing derivative files for {name_central_body}")

        pot_expansion, d_pot_expansions = _read_derivative_files(data_root_path, name_central_body)
        
        # Lambdify
        f_pot_expansions = sym.lambdify([x, y, z], pot_expansion, lambdify_backend)
        f_d_pot_expansion = [sym.lambdify([x, y, z], d, lambdify_backend) for d in d_pot_expansions]

        return f_pot_expansions, f_d_pot_expansion



    if verbose:
        print(f"    Body: {name_central_body} | Root: {data_root_path}")

    if not pot_dir.exists():
        if verbose:
            print(f"   Potential dir not found: {pot_dir}")
        return [], []

    files = sorted(glob.glob(str(pot_dir / pattern)))
    if not files:
        if verbose:
            print(f"   No files matching '{pattern}' in {pot_dir}")
        return [], []

    # Build a sorted list by the embedded index (like ..._NNN.dat)
    files_pot = pd.DataFrame({"file": files})
    # Robust index extraction: fall back to plain sort if parsing fails
    def _extract_idx(p: str) -> int:
        base = Path(p).stem  # e.g., "pot_12"
        try:
            # last "_" segment as int
            return int(base.split("_")[-1])
        except:
            return 10**12  # push unparseable to end
    files_pot["pot_idx"] = files_pot["file"].map(_extract_idx)
    files_pot.sort_values("pot_idx", inplace=True)
    files_pot.reset_index(drop=True, inplace=True)

    # Sum selected partial potentials
    pot_expansion = 0
    used = min(n_files, len(files_pot))
    if verbose:
        print(f"    Loading {used} potential parts from {pot_dir}")
    for i in range(used):
        with open(files_pot.loc[i, "file"], "rb") as inf:
            pot_part = pickle.load(inf)
        pot_expansion += pot_part

    # Substitute constants / radius
    pot_expansion = pot_expansion.subs({
        g_sym: 1,  # if you want: use G explicitly, or include in gm0
        mass_sym: gm0,
        r_sym: sym.sqrt(x**2 + y**2 + z**2)
    })

    # potential
    generate_fortran_code(pot_expansion, f'{base}/{required[0].split(".for")[0]}')

    # Derivatives
    d_pot_expansions = [sym.diff(pot_expansion, var) for var in (x, y, z)]

    # emit Fortran
    missing = [f for f in required if not (base / f).is_file()]
    if missing:
        if verbose:
            print("    Missing Fortran derivative files:", missing)
            for i, d_pot_expansion in enumerate(d_pot_expansions):
                generate_fortran_code(d_pot_expansion, f'{base}/{required[i+1].split(".for")[0]}')

    # save d_x/d_y/d_z
    required = ["pot.txt", "d_x.txt", "d_y.txt", "d_z.txt"]
    missing = [f for f in required if not (base / f).is_file()]

    if missing:
        if verbose:
            print("    Missing derivative files:", missing)
            print(f"    Saving derivatives to {pot_dir} (d_x.txt, d_y.txt, d_z.txt)")
        _write_derivative_files(data_root_path, name_central_body, pot_expansion, d_pot_expansions)
   
    # Lambdify
    f_pot_expansions = sym.lambdify([x, y, z], pot_expansion, lambdify_backend)
    f_d_pot_expansion = [sym.lambdify([x, y, z], d, lambdify_backend) for d in d_pot_expansions]

    if verbose:
        print("    Derivatives are ready.")

    return f_pot_expansions, f_d_pot_expansion

