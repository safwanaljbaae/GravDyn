# pytest -v
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import trimesh
import pytest

from pathlib import Path
from gravdyn.plot_tools import plot_projection, save_mesh_projections, save_mesh_3d_html
from gravdyn.shape_tools import load_vertices, load_faces

matplotlib.use("Agg")


@pytest.fixture
def simple_mesh():
    vertices_file = "/home/aljbaae/Script_Safwan/GravDyn/Package_GravDyn/Data/Apophis/shape_v.dat"
    faces_file = "/home/aljbaae/Script_Safwan/GravDyn/Package_GravDyn/Data/Apophis/shape_f.dat"

    vertices = load_vertices(vertices_file=vertices_file)
    faces = load_faces(faces_file=faces_file)

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


@pytest.mark.parametrize(
    "plane,xlabel,ylabel,title",
    [
        ("xy", "X", "Y", "XY projection"),
        ("xz", "X", "Z", "XZ projection"),
        ("yz", "Y", "Z", "YZ projection"),
    ],
)
def test_plot_projection_sets_labels_and_title(simple_mesh, plane, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    try:
        plot_projection(ax, simple_mesh, plane=plane)

        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel
        assert ax.get_title() == title
        assert ax.get_aspect() == 1.0
    finally:
        plt.close(fig)


def test_plot_projection_sets_symmetric_limits(simple_mesh):
    fig, ax = plt.subplots()
    try:
        plot_projection(ax, simple_mesh, plane="xy")
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        assert np.isclose(abs(xlim[0]), abs(xlim[1]))
        assert np.isclose(abs(ylim[0]), abs(ylim[1]))
    finally:
        plt.close(fig)


def test_plot_projection_invalid_plane_raises(simple_mesh):
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="Plane must be xy, xz, or yz"):
            plot_projection(ax, simple_mesh, plane="ab")
    finally:
        plt.close(fig)


def test_save_mesh_projections_creates_output_file(tmp_path, simple_mesh, monkeypatch):
    monkeypatch.chdir(tmp_path)

    save_mesh_projections(simple_mesh, asteroid_name="Apophis", data_folder="../Data", file_nam="shape_projection.png")

    output_file = Path("../Data/Apophis/shape_projection.png")
    assert output_file.exists()
    assert output_file.is_file()
    assert output_file.stat().st_size > 0


def test_save_mesh_3d_html_creates_output_file(tmp_path, simple_mesh, monkeypatch):
    monkeypatch.chdir(tmp_path)

    save_mesh_3d_html(simple_mesh, asteroid_name="Apophis", data_folder="../Data", file_nam="shape_3d.html")

    output_file = Path("../Data/Apophis/shape_3d.html")
    assert output_file.exists()
    assert output_file.is_file()
    assert output_file.stat().st_size > 0