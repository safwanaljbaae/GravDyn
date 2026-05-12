import numpy as np

from gravdyn.shape_verification import report_principal_axes


class TestReportPrincipalAxes:
    def test_perfect_alignment(self, capsys):
        I = np.eye(3)
        report_principal_axes(I, I, [0.0, 0.0, 0.0])
        captured = capsys.readouterr()
        assert "successfully aligned" in captured.out

    def test_poor_alignment_shows_warning(self, capsys):
        I = np.eye(3)
        poor = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        report_principal_axes(I, poor, [45.0, 45.0, 45.0])
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "inaccurate" in captured.out

    def test_prints_angles(self, capsys):
        I = np.eye(3)
        angles = [10.5, 20.3, 30.7]
        report_principal_axes(I, I, angles)
        captured = capsys.readouterr()
        for a in angles:
            assert str(a) in captured.out

    def test_output_format(self, capsys):
        I = np.eye(3)
        report_principal_axes(I, I, [0.0, 0.0, 0.0])
        captured = capsys.readouterr()
        assert "Principal Axes Transformation Report" in captured.out
        assert "Alignment error" in captured.out
