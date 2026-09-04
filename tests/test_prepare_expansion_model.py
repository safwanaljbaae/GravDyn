import importlib
import pickle
from pathlib import Path

import numpy as np
import pytest
import sympy as sym

from gravdyn import build_potential_derivatives

BPD_MODULE = importlib.import_module("gravdyn.build_potential_derivatives")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def symbols():
    return sym.symbols("x y z")


def _write_derivative_text_files(body_dir: Path, pot_expr: str,
                                 dx: str, dy: str, dz: str) -> None:
    """Write ``pot.txt`` and ``d_{x,y,z}.txt`` plus the ``.for`` sentinel files
    that signal the fast path.

    A dummy ``Pot_Expansion/`` directory is also created because the function
    checks for it *before* checking the ``.for`` files.
    """
    body_dir.mkdir(parents=True, exist_ok=True)
    (body_dir / "Pot_Expansion").mkdir(exist_ok=True)

    (body_dir / "pot.txt").write_text(pot_expr, encoding="utf-8")
    (body_dir / "d_x.txt").write_text(dx, encoding="utf-8")
    (body_dir / "d_y.txt").write_text(dy, encoding="utf-8")
    (body_dir / "d_z.txt").write_text(dz, encoding="utf-8")
    for fname in ["pot.for", "d_x.for", "d_y.for", "d_z.for"]:
        (body_dir / fname).write_text("dummy", encoding="utf-8")


def _write_pickle(pot_dir: Path, filename: str, expr: sym.Expr) -> None:
    pot_dir.mkdir(parents=True, exist_ok=True)
    with open(pot_dir / filename, "wb") as f:
        pickle.dump(expr, f)


def _point_mass_expr(gm: float = 1.0) -> sym.Expr:
    """Return ``g * mass * factor / r``, the standard point-mass potential
    expression in the symbolic form the code expects."""
    g, mass, r = sym.symbols("g mass r")
    return gm * g * mass / r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExistingDerivatives:
    """Fast path: ``.for`` + ``.txt`` files already exist."""

    def test_loads_existing_derivatives(self, tmp_path, symbols):
        x, y, z = symbols
        body = "Apophis"
        body_dir = tmp_path / body

        _write_derivative_text_files(
            body_dir=body_dir,
            pot_expr=str(x**2 + y**2 + z**2),
            dx=str(2 * x),
            dy=str(2 * y),
            dz=str(2 * z),
        )

        f_pot, f_derivs = build_potential_derivatives(
            name_central_body=body,
            base_dir=str(tmp_path),
            lambdify_backend="numpy",
            verbose=False,
        )

        assert callable(f_pot)
        assert len(f_derivs) == 3

        assert f_pot(1.0, 2.0, 3.0) == pytest.approx(14.0)
        assert f_derivs[0](1.0, 2.0, 3.0) == pytest.approx(2.0)
        assert f_derivs[1](1.0, 2.0, 3.0) == pytest.approx(4.0)
        assert f_derivs[2](1.0, 2.0, 3.0) == pytest.approx(6.0)

    def test_loads_with_jax_backend(self, tmp_path, symbols):
        """Exercise the 'jax' lambdify backend (must be installed)."""
        x, y, z = symbols
        body = "JaxBody"
        body_dir = tmp_path / body

        _write_derivative_text_files(
            body_dir=body_dir,
            pot_expr=str(x**2 + y**2),
            dx=str(2 * x),
            dy=str(2 * y),
            dz="0",
        )

        f_pot, f_derivs = build_potential_derivatives(
            name_central_body=body,
            base_dir=str(tmp_path),
            lambdify_backend="jax",
            verbose=False,
        )

        assert f_pot(3.0, 4.0, 0.0) == pytest.approx(25.0)
        assert f_derivs[0](3.0, 4.0, 0.0) == pytest.approx(6.0)
        assert f_derivs[1](3.0, 4.0, 0.0) == pytest.approx(8.0)

    def test_missing_derivative_files_returns_empty(self, tmp_path):
        """No ``.for`` / ``.txt`` files and empty ``Pot_Expansion/`` returns []."""
        body = "EmptyBody"
        # The function checks Pot_Expansion/ first; create it empty so
        # it doesn't try GitHub.  Without matching pickle files the
        # result is [].
        (tmp_path / body / "Pot_Expansion").mkdir(parents=True, exist_ok=True)

        result_pot, result_derivs = build_potential_derivatives(
            name_central_body=body,
            base_dir=str(tmp_path),
            lambdify_backend="numpy",
            verbose=False,
        )
        assert result_pot == []
        assert result_derivs == []


class TestLineaFallback:
    """When ``Pot_Expansion/`` is absent the code tries the LInEA server."""

    def test_linea_failure_returns_empty(self, tmp_path, monkeypatch):
        """If the LInEA download fails, the function should return [], [].

        We mock the LInEA service calls to do nothing.
        """
        monkeypatch.setattr(BPD_MODULE, "list_linea_folder_files",
                            lambda *a, **kw: [])
        monkeypatch.setattr(BPD_MODULE, "download_linea_folder_files",
                            lambda *a, **kw: None)

        body = "NoData"
        result_pot, result_derivs = build_potential_derivatives(
            name_central_body=body,
            base_dir=str(tmp_path),
            pattern="pot_*.dat",
            lambdify_backend="numpy",
            verbose=False,
        )
        assert result_pot == []
        assert result_derivs == []


class TestEmptyPotDir:
    """``Pot_Expansion/`` exists but contains no matching files."""

    def test_no_matching_files_returns_empty(self, tmp_path):
        body = "Apophis"
        pot_dir = tmp_path / body / "Pot_Expansion"
        pot_dir.mkdir(parents=True, exist_ok=True)

        result_pot, result_derivs = build_potential_derivatives(
            name_central_body=body,
            base_dir=str(tmp_path),
            pattern="pot_*.dat",
            lambdify_backend="numpy",
            verbose=False,
        )
        assert result_pot == []
        assert result_derivs == []


class TestBuildFromPickledParts:
    """Compute potential and derivatives from scratch using pickle files."""

    def test_single_pickle_file(self, tmp_path):
        """One pickle file produces correct point-mass potential."""
        body = "TestSingle"
        pot_dir = tmp_path / body / "Pot_Expansion"
        _write_pickle(pot_dir, "pot_0.dat", _point_mass_expr())

        gm0 = 10.0
        f_pot, f_derivs = build_potential_derivatives(
            name_central_body=body,
            base_dir=str(tmp_path),
            pattern="pot_*.dat",
            gm0=gm0,
            lambdify_backend="numpy",
            verbose=False,
        )

        assert callable(f_pot)
        assert len(f_derivs) == 3

        # pot = gm0 / sqrt(x**2 + y**2 + z**2)
        #     = 10 / 5 = 2.0  at (3, 4, 0)
        assert f_pot(3.0, 4.0, 0.0) == pytest.approx(2.0)

        # d/dx = -gm0 * x / r**3  = -10 * 3 / 125 = -0.24
        assert f_derivs[0](3.0, 4.0, 0.0) == pytest.approx(-0.24)
        # d/dy = -10 * 4 / 125 = -0.32
        assert f_derivs[1](3.0, 4.0, 0.0) == pytest.approx(-0.32)
        # d/dz = 0
        assert f_derivs[2](3.0, 4.0, 0.0) == pytest.approx(0.0)

    def test_multiple_pickle_files_summed(self, tmp_path):
        """Multiple pickle files are summed before substitution."""
        body = "TestMulti"
        pot_dir = tmp_path / body / "Pot_Expansion"

        g, mass, r = sym.symbols("g mass r")
        _write_pickle(pot_dir, "pot_0.dat", g * mass / r)
        _write_pickle(pot_dir, "pot_1.dat", 0.5 * g * mass / r)
        # total = 1.5 * g * mass / r

        gm0 = 10.0
        f_pot, f_derivs = build_potential_derivatives(
            name_central_body=body,
            base_dir=str(tmp_path),
            pattern="pot_*.dat",
            gm0=gm0,
            lambdify_backend="numpy",
            verbose=False,
        )

        # pot = 1.5 * gm0 / sqrt(x**2 + y**2 + z**2)
        #     = 15 / 5 = 3.0  at (3, 4, 0)
        assert f_pot(3.0, 4.0, 0.0) == pytest.approx(3.0)
        # d/dx = -1.5 * 10 * 3 / 125 = -0.36
        assert f_derivs[0](3.0, 4.0, 0.0) == pytest.approx(-0.36)
        # d/dy = -1.5 * 10 * 4 / 125 = -0.48
        assert f_derivs[1](3.0, 4.0, 0.0) == pytest.approx(-0.48)

    def test_n_files_limit(self, tmp_path):
        """``n_files`` controls how many pickle files are loaded."""
        body = "TestLimit"
        pot_dir = tmp_path / body / "Pot_Expansion"

        g, mass, r = sym.symbols("g mass r")
        for i in range(5):
            _write_pickle(pot_dir, f"pot_{i}.dat", g * mass / r)

        gm0 = 10.0
        # Only load 3 out of 5 files → pot = 3 * gm0 / r
        f_pot, f_derivs = build_potential_derivatives(
            name_central_body=body,
            base_dir=str(tmp_path),
            pattern="pot_*.dat",
            n_files=3,
            gm0=gm0,
            lambdify_backend="numpy",
            verbose=False,
        )

        assert f_pot(3.0, 4.0, 0.0) == pytest.approx(6.0)

    def test_output_files_written(self, tmp_path):
        """After computation, ``pot.txt`` and ``d_{x,y,z}.txt`` are saved."""
        body = "Apophis"
        pot_dir = tmp_path / body / "Pot_Expansion"
        _write_pickle(pot_dir, "pot_0.dat", _point_mass_expr())

        build_potential_derivatives(
            name_central_body=body,
            base_dir=str(tmp_path),
            pattern="pot_*.dat",
            gm0=10.0,
            lambdify_backend="numpy",
            verbose=False,
        )

        body_dir = tmp_path / body
        assert (body_dir / "pot.txt").exists()
        assert (body_dir / "d_x.txt").exists()
        assert (body_dir / "d_y.txt").exists()
        assert (body_dir / "d_z.txt").exists()

    def test_verbose_mode_does_not_crash(self, tmp_path):
        """``verbose=True`` should not raise."""
        body = "Verbose"
        pot_dir = tmp_path / body / "Pot_Expansion"
        _write_pickle(pot_dir, "pot_0.dat", _point_mass_expr())

        f_pot, f_derivs = build_potential_derivatives(
            name_central_body=body,
            base_dir=str(tmp_path),
            pattern="pot_*.dat",
            gm0=10.0,
            lambdify_backend="numpy",
            verbose=True,
        )
        assert callable(f_pot)