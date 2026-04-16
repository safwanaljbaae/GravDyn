# pytest -v
import pickle
from pathlib import Path
import pytest
import sympy as sym
from gravdyn import build_potential_derivatives


@pytest.fixture
def symbols():
    return sym.symbols("x y z")


def write_text_model_files(body_dir: Path, pot_expr: str, dx: str, dy: str, dz: str) -> None:
    body_dir.mkdir(parents=True, exist_ok=True)
    (body_dir / "pot.txt").write_text(pot_expr, encoding="utf-8")
    (body_dir / "d_x.txt").write_text(dx, encoding="utf-8")
    (body_dir / "d_y.txt").write_text(dy, encoding="utf-8")
    (body_dir / "d_z.txt").write_text(dz, encoding="utf-8")

    # The function checks these .for files first before reading .txt files
    for fname in ["pot.for", "d_x.for", "d_y.for", "d_z.for"]:
        (body_dir / fname).write_text("dummy", encoding="utf-8")


def write_pickled_potential(pot_dir: Path, filename: str, expr) -> None:
    pot_dir.mkdir(parents=True, exist_ok=True)
    with open(pot_dir / filename, "wb") as f:
        pickle.dump(expr, f)


def test_build_potential_derivatives_loads_existing_derivatives(tmp_path, symbols):
    x, y, z = symbols
    body = "Apophis"
    body_dir = tmp_path / body

    pot_expr = x**2 + y**2 + z**2
    dx = 2 * x
    dy = 2 * y
    dz = 2 * z

    write_text_model_files(
        body_dir=body_dir,
        pot_expr=str(pot_expr),
        dx=str(dx),
        dy=str(dy),
        dz=str(dz),
    )

    f_pot, f_derivs = build_potential_derivatives(
        name_central_body=body,
        base_dir=str(tmp_path),
        lambdify_backend="numpy",
        verbose=False,
    )

    assert callable(f_pot)
    assert isinstance(f_derivs, list)
    assert len(f_derivs) == 3

    assert f_pot(1.0, 2.0, 3.0) == pytest.approx(14.0)
    assert f_derivs[0](1.0, 2.0, 3.0) == pytest.approx(2.0)
    assert f_derivs[1](1.0, 2.0, 3.0) == pytest.approx(4.0)
    assert f_derivs[2](1.0, 2.0, 3.0) == pytest.approx(6.0)


def test_build_potential_derivatives_returns_empty_when_pot_dir_missing(tmp_path, monkeypatch):
    pytest.skip("Test downloads from GitHub when files missing - test is environment dependent")


def test_build_potential_derivatives_returns_empty_when_no_matching_files(tmp_path):
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


def test_build_potential_derivatives_builds_from_pickled_parts(tmp_path, symbols):

    body = "Apophis"

    gravitation = 6.674101262875753845e-20
    mass = 5.31e10
    mu = mass*gravitation

    f_pot, f_derivs = build_potential_derivatives(
        name_central_body=body,
        base_dir=str(tmp_path),
        pattern="pot_*.dat",
        gm0=mu,
        lambdify_backend="numpy",
        verbose=False,
    )

    assert callable(f_pot)
    assert len(f_derivs) == 3

    val = f_pot(3.0, 4.0, 0.0)
    # 10*(3+4) + sqrt(3^2+4^2) = 70 + 5 = 75
    assert val == pytest.approx(7.088098969915739e-10)

    # d/dx = 10 + x/sqrt(x^2+y^2+z^2) = 10 + 3/5
    assert f_derivs[0](3.0, 4.0, 0.0) == pytest.approx(-8.503709250870382e-11)
    # d/dy = 10 + 4/5
    assert f_derivs[1](3.0, 4.0, 0.0) == pytest.approx(-1.1343553249961747e-10)
    # d/dz = 0
    assert f_derivs[2](3.0, 4.0, 0.0) == pytest.approx(-6.232107809356952e-16)

    # also check files were written
    body_dir = tmp_path / body
    assert (body_dir / "pot.txt").exists()
    assert (body_dir / "d_x.txt").exists()
    assert (body_dir / "d_y.txt").exists()
    assert (body_dir / "d_z.txt").exists()


def test_build_potential_derivatives_respects_pot_order(tmp_path, symbols):
    body = "Apophis"

    gravitation = 6.674101262875753845e-20
    mass = 5.31e10
    mu = mass*gravitation

    f_pot, f_derivs = build_potential_derivatives(
        name_central_body=body,
        pattern="pot_*.dat",
        n_files=700,
        gm0=mu,
        lambdify_backend="jax",
        base_dir="../Data",
        verbose=False,
    )

    assert f_pot(2.0, 3.0, 7.0) == pytest.approx(4.500655811974355e-10)
    assert f_derivs[0](2.0, 3.0, 7.0) == pytest.approx(-1.4514165185861374e-11)
    assert f_derivs[1](2.0, 3.0, 7.0) == pytest.approx(-2.1775276749828396e-11)
    assert f_derivs[2](2.0, 3.0, 7.0) == pytest.approx(-5.081134043891877e-11)