import sympy as sym
from pathlib import Path

from gravdyn.build_potential_derivatives import generate_fortran_code


class TestGenerateFortranCode:
    def test_writes_file_with_given_name(self, tmp_path):
        x, y, z = sym.symbols("x y z")
        expr = x**2 + y**2 + z**2
        cwd = tmp_path
        generate_fortran_code(expr, str(cwd / "test_func"))
        assert (cwd / "test_func.for").exists()

    def test_file_contains_function_header(self, tmp_path):
        x, y = sym.symbols("x y")
        expr = x * sym.sin(y)
        cwd = tmp_path
        generate_fortran_code(expr, str(cwd / "myfunc"))
        content = (cwd / "myfunc.for").read_text()
        assert "REAL*8 function" in content
        assert "myfunc" in content

    def test_contains_variable_declarations(self, tmp_path):
        x, y, z = sym.symbols("x y z")
        expr = x + y + z
        cwd = tmp_path
        generate_fortran_code(expr, str(cwd / "sum3"))
        content = (cwd / "sum3.for").read_text()
        assert "REAL*8 x, y, z" in content
        assert "result" in content

    def test_long_expression_splits_into_lines(self, tmp_path):
        x, y = sym.symbols("x y")
        expr = x**50 + y**50 + x * y
        cwd = tmp_path
        generate_fortran_code(expr, str(cwd / "long_expr"))
        content = (cwd / "long_expr.for").read_text()
        lines = content.strip().split("\n")
        assert len(lines) > 5

    def test_ends_with_end_statement(self, tmp_path):
        x = sym.symbols("x")
        cwd = tmp_path
        generate_fortran_code(x, str(cwd / "identity"))
        content = (cwd / "identity.for").read_text().strip()
        assert content.endswith("end")
