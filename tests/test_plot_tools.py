import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
import pytest
from pathlib import Path

from gravdyn.plot_tools import (
    plot_projection,
    save_mesh_projections,
    save_mesh_3d_html,
    plot_mesh_problem_html,
    plot_layers_by_density,
)


@pytest.fixture
def tetra_mesh():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


@pytest.fixture
def box_mesh():
    return trimesh.creation.box(extents=[2.0, 2.0, 2.0])


class TestPlotProjection:
    @pytest.mark.parametrize(
        "plane,xlabel,ylabel,title",
        [
            ("xy", "X", "Y", "XY projection"),
            ("xz", "X", "Z", "XZ projection"),
            ("yz", "Y", "Z", "YZ projection"),
        ],
    )
    def test_sets_labels_and_title(self, tetra_mesh, plane, xlabel, ylabel, title):
        fig, ax = plt.subplots()
        plot_projection(ax, tetra_mesh, plane=plane)
        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel
        assert ax.get_title() == title
        assert ax.get_aspect() == 1.0
        plt.close(fig)

    def test_sets_symmetric_limits(self, box_mesh):
        fig, ax = plt.subplots()
        plot_projection(ax, box_mesh, plane="xy")
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert np.isclose(abs(xlim[0]), abs(xlim[1]))
        assert np.isclose(abs(ylim[0]), abs(ylim[1]))
        plt.close(fig)

    def test_invalid_plane_raises(self, tetra_mesh):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="Plane must be xy, xz, or yz"):
            plot_projection(ax, tetra_mesh, plane="ab")
        plt.close(fig)

    def test_limit_is_110_percent_of_extent(self, box_mesh):
        fig, ax = plt.subplots()
        plot_projection(ax, box_mesh, plane="xy")
        xlim = ax.get_xlim()
        assert xlim[1] == pytest.approx(1.1 * 1.0)
        plt.close(fig)


class TestSaveMeshProjections:
    def test_creates_output_file(self, tetra_mesh, tmp_path):
        data_folder = str(tmp_path)
        save_mesh_projections(tetra_mesh, asteroid_name="TestBody",
                              data_folder=data_folder, file_nam="projection.png")
        output_file = tmp_path / "TestBody" / "projection.png"
        assert output_file.exists()
        assert output_file.is_file()
        assert output_file.stat().st_size > 0

    def test_uses_given_filename(self, tetra_mesh, tmp_path):
        save_mesh_projections(tetra_mesh, asteroid_name="TestBody",
                              data_folder=str(tmp_path), file_nam="custom_name.png")
        assert (tmp_path / "TestBody" / "custom_name.png").exists()


class TestSaveMesh3DHTML:
    def test_creates_output_file(self, tetra_mesh, tmp_path):
        save_mesh_3d_html(tetra_mesh, asteroid_name="TestBody",
                          data_folder=str(tmp_path), file_nam="mesh_3d.html")
        output_file = tmp_path / "TestBody" / "mesh_3d.html"
        assert output_file.exists()
        assert output_file.is_file()
        assert output_file.stat().st_size > 0

    def test_html_content(self, tetra_mesh, tmp_path):
        save_mesh_3d_html(tetra_mesh, asteroid_name="TestBody",
                          data_folder=str(tmp_path), file_nam="mesh_3d.html")
        content = (tmp_path / "TestBody" / "mesh_3d.html").read_text()
        assert "Plotly" in content
        assert "x" in content and "y" in content and "z" in content


class TestPlotMeshProblemHTML:
    def test_boundary_edges(self, tmp_path):
        mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            faces=[[0, 1, 2], [0, 1, 3]],
            process=False,
        )
        output = str(tmp_path / "boundary.html")
        fig = plot_mesh_problem_html(mesh, problem_type="boundary_edges", output_file=output)
        assert Path(output).exists()
        assert fig is not None

    def test_degenerate_faces(self, tmp_path):
        mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[[0, 0, 1], [0, 1, 2]],
            process=False,
        )
        output = str(tmp_path / "degenerate.html")
        fig = plot_mesh_problem_html(mesh, problem_type="degenerate_faces", output_file=output)
        assert Path(output).exists()

    def test_duplicate_faces(self, tmp_path):
        mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            faces=[[0, 1, 2], [0, 1, 2], [0, 1, 3]],
            process=False,
        )
        output = str(tmp_path / "duplicate.html")
        fig = plot_mesh_problem_html(mesh, problem_type="duplicate_faces", output_file=output)
        assert Path(output).exists()

    def test_unused_vertices(self, tmp_path):
        mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [99, 99, 99]],
            faces=[[0, 1, 2]],
            process=False,
        )
        output = str(tmp_path / "unused.html")
        fig = plot_mesh_problem_html(mesh, problem_type="unused_vertices", output_file=output)
        assert Path(output).exists()

    def test_invalid_problem_type_raises(self, tmp_path):
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]], process=False)
        output = str(tmp_path / "invalid.html")
        with pytest.raises(ValueError, match="problem_type must be one of"):
            plot_mesh_problem_html(mesh, problem_type="invalid_type", output_file=output)

    def test_non_trimesh_input_raises(self, tmp_path):
        output = str(tmp_path / "bad.html")
        with pytest.raises(TypeError, match="Input must be a trimesh.Trimesh"):
            plot_mesh_problem_html(mesh="not_a_mesh", problem_type="boundary_edges", output_file=output)


class TestPlotLayersByDensity:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "x": [0.1, 0.2, 0.3, 0.4],
            "y": [0.0, 0.1, 0.0, 0.1],
            "z": [0.0, 0.0, 0.1, 0.1],
            "layer_id": [0, 0, 1, 1],
            "density_input": [1.5, 1.5, 2.0, 2.0],
        })

    def test_returns_none_without_output_file(self, sample_df):
        result = plot_layers_by_density(sample_df)
        assert result is None

    def test_saves_to_file(self, sample_df, tmp_path):
        output = str(tmp_path / "layers.png")
        plot_layers_by_density(sample_df, output_file=output)
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0
