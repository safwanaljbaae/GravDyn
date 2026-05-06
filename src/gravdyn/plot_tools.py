import os
import trimesh
import numpy as np
import pandas as pd
import jax.numpy as jnp
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import cm, colors
from scipy.spatial.distance import cdist

def plot_projection(ax, mesh, plane="xy"):
    """
    Plot a 2D orthogonal projection of a triangular surface mesh.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes object on which the projection is drawn.
    mesh : object
        Mesh-like object with attributes ``vertices`` and ``faces``.
        ``vertices`` must be an array of shape ``(N, 3)`` containing the
        Cartesian coordinates of the mesh vertices. ``faces`` must be an
        integer array of shape ``(M, 3)`` containing indices of triangular
        faces referencing rows of ``vertices``.
    plane : {'xy', 'xz', 'yz'}, optional
        Projection plane. ``'xy'`` projects onto the X-Y plane, ``'xz'``
        onto the X-Z plane, and ``'yz'`` onto the Y-Z plane.

    Returns
    -------
    None
        The function modifies ``ax`` in place.

    Raises
    ------
    ValueError
        If ``plane`` is not one of ``'xy'``, ``'xz'``, or ``'yz'``.

    Notes
    -----
    Each triangular face is plotted as a closed polygonal outline. Axis
    limits are set symmetrically about zero using 110% of the maximum
    absolute projected coordinate extent, and the aspect ratio is forced
    to be equal.

    """
    verts = mesh.vertices
    faces = mesh.faces

    if plane == "xy":
        i, j = 0, 1
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        title = "XY projection"

    elif plane == "xz":
        i, j = 0, 2
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        title = "XZ projection"

    elif plane == "yz":
        i, j = 1, 2
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
        title = "YZ projection"

    else:
        raise ValueError("Plane must be xy, xz, or yz")

    for face in faces:
        poly = verts[face][:, [i, j]]
        poly = np.vstack([poly, poly[0]])
        ax.plot(poly[:, 0], poly[:, 1], "k-", linewidth=0.5)

    # Compute symmetric limits with +10%
    coords = verts[:, [i, j]]
    max_extent = np.max(np.abs(coords))
    limit = 1.1 * max_extent

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    ax.set_title(title)
    ax.set_aspect("equal")


def save_mesh_projections(mesh, asteroid_name, data_folder, file_nam):
    """
    Save three orthogonal 2D projections of a triangular mesh as a single image.

    Parameters
    ----------
    mesh : object
        Mesh-like object with attributes ``vertices`` and ``faces``.
        ``vertices`` must be an array of shape ``(N, 3)`` containing the
        Cartesian coordinates of the mesh vertices. ``faces`` must be an
        integer array of shape ``(M, 3)`` containing indices of triangular
        faces referencing rows of ``vertices``.
    asteroid_name : str
        Name of the target body. A subdirectory with this name is created
        inside the ``Data/`` directory.
    data_folder : str
        Path to the directory containing the mesh data files (vertices and faces).
    file_nam : str or path-like
        Output filename, including the desired image extension
        (for example, ``"projections.png"`` or ``"shape_views.pdf"``).

    Returns
    -------
    None

    Notes
    -----
    The function generates a 1x3 Matplotlib figure containing the ``xy``,
    ``xz``, and ``yz`` projections of the mesh.

    """

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    plot_projection(axs[0], mesh, "xy")
    plot_projection(axs[1], mesh, "xz")
    plot_projection(axs[2], mesh, "yz")

    plt.tight_layout()

    # Build full output directory
    output_dir = Path(data_folder) / asteroid_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define full file path (with extension!)
    output_filename = output_dir / file_nam

    # Save figure
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_filename}")


def save_mesh_3d_html(mesh, asteroid_name, data_folder, file_nam):
    """
    Save an interactive 3D Plotly visualization of a triangular mesh as an HTML file.

    Parameters
    ----------
    mesh : object
        Mesh-like object with attributes ``vertices`` and ``faces``.
        ``vertices`` must be an array of shape ``(N, 3)`` containing the
        Cartesian coordinates of the mesh vertices. ``faces`` must be an
        integer array of shape ``(M, 3)`` containing indices of triangular
        faces referencing rows of ``vertices``.
    asteroid_name : str
        Name of the target body. A subdirectory with this name is created
        inside the ``Data/`` directory.
    data_folder : str
        Path to the directory containing the mesh data files (vertices and faces).
    file_nam : str or path-like
        Output filename, including the ``.html`` extension.

    Returns
    -------
    None

    Notes
    -----
    The mesh is rendered using ``plotly.graph_objects.Mesh3d`` with flat
    shading. The x, y, and z axes are given symmetric limits about zero
    using 110% of the maximum absolute coordinate value, and the scene
    aspect ratio is fixed to a cube.
    """
    verts = mesh.vertices
    faces = mesh.faces

    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]

    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]

    limit = 1.1 * np.max(np.abs(verts))

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                opacity=1.0,
                flatshading=True
            )
        ]
    )

    fig.update_layout(
        title="Asteroid 3D Mesh",
        scene=dict(
            xaxis=dict(title="X", range=[-limit, limit]),
            yaxis=dict(title="Y", range=[-limit, limit]),
            zaxis=dict(title="Z", range=[-limit, limit]),
            aspectmode="cube"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Build full output directory
    output_dir = Path(data_folder) / asteroid_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define full file path (with extension!)
    output_filename = output_dir / file_nam

    fig.write_html(output_filename)

    print(f"3D HTML plot saved to: {output_filename}")


def plot_mesh_problem_html(
        mesh: trimesh.Trimesh,
        problem_type: str,
        output_file: str,
        tol: float = 1e-12,
        max_items: int | None = None,
        include_plotlyjs: bool | str = True,
):
    """
    Save an interactive HTML figure showing a specific mesh problem.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input triangular mesh.

    problem_type : str
        One of:
        - "boundary_edges"
        - "non_manifold_edges"
        - "degenerate_faces"
        - "duplicate_faces"
        - "unused_vertices"
        - "broken_faces"
        - "components"

    output_file : str
        Path to the output HTML file.

    tol : float
        Tolerance for detecting near-zero-area faces.

    max_items : int or None
        Maximum number of problematic elements to plot.

    include_plotlyjs : bool or str
        If True, the HTML file is fully standalone.
        If "cdn", the file is smaller but needs internet access.
    """

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Input must be a trimesh.Trimesh object.")

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    fig = go.Figure()

    # ------------------------------------------------------------
    # Base mesh
    # ------------------------------------------------------------
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=0.18,
            color="lightgray",
            name="Mesh",
            hoverinfo="skip",
        )
    )

    title = ""

    # ------------------------------------------------------------
    # Boundary or non-manifold edges
    # ------------------------------------------------------------
    if problem_type in ["boundary_edges", "non_manifold_edges"]:
        edges = mesh.edges_sorted

        unique_edges, edge_counts = np.unique(
            edges,
            axis=0,
            return_counts=True,
        )

        if problem_type == "boundary_edges":
            problem_edges = unique_edges[edge_counts == 1]
            title = f"Boundary / open edges: {len(problem_edges)}"
            trace_name = "Boundary edges"

        else:
            problem_edges = unique_edges[edge_counts > 2]
            title = f"Non-manifold edges: {len(problem_edges)}"
            trace_name = "Non-manifold edges"

        if max_items is not None:
            problem_edges = problem_edges[:max_items]

        x_lines, y_lines, z_lines = [], [], []

        for edge in problem_edges:
            p1 = vertices[edge[0]]
            p2 = vertices[edge[1]]

            x_lines += [p1[0], p2[0], None]
            y_lines += [p1[1], p2[1], None]
            z_lines += [p1[2], p2[2], None]

        fig.add_trace(
            go.Scatter3d(
                x=x_lines,
                y=y_lines,
                z=z_lines,
                mode="lines",
                line=dict(width=8, color="red"),
                name=trace_name,
            )
        )

    # ------------------------------------------------------------
    # Degenerate faces
    # ------------------------------------------------------------
    elif problem_type == "degenerate_faces":
        repeated_vertex_faces = np.where(
            (faces[:, 0] == faces[:, 1])
            | (faces[:, 1] == faces[:, 2])
            | (faces[:, 0] == faces[:, 2])
        )[0]

        zero_area_faces = np.where(mesh.area_faces <= tol)[0]

        problem_faces = np.unique(
            np.concatenate([repeated_vertex_faces, zero_area_faces])
        ).astype(int)

        title = f"Degenerate faces: {len(problem_faces)}"

        if max_items is not None:
            problem_faces = problem_faces[:max_items]

        _add_face_edges_and_centroids(
            fig,
            vertices,
            faces,
            problem_faces,
            line_color="red",
            marker_color="red",
            name="Degenerate faces",
        )

    # ------------------------------------------------------------
    # Duplicate faces
    # ------------------------------------------------------------
    elif problem_type == "duplicate_faces":
        sorted_faces = np.sort(faces, axis=1)

        unique_faces, face_counts = np.unique(
            sorted_faces,
            axis=0,
            return_counts=True,
        )

        duplicate_face_keys = unique_faces[face_counts > 1]

        duplicate_face_ids = []

        for duplicate_key in duplicate_face_keys:
            ids = np.where(
                np.all(sorted_faces == duplicate_key, axis=1)
            )[0]
            duplicate_face_ids.extend(ids.tolist())

        duplicate_face_ids = np.asarray(duplicate_face_ids, dtype=int)

        title = f"Duplicate face entries: {len(duplicate_face_ids)}"

        if max_items is not None:
            duplicate_face_ids = duplicate_face_ids[:max_items]

        _add_face_edges_and_centroids(
            fig,
            vertices,
            faces,
            duplicate_face_ids,
            line_color="red",
            marker_color="red",
            name="Duplicate faces",
        )

    # ------------------------------------------------------------
    # Unused vertices
    # ------------------------------------------------------------
    elif problem_type == "unused_vertices":
        if len(faces) > 0:
            used_vertices = np.unique(faces.reshape(-1))
            all_vertices = np.arange(len(vertices))
            unused_vertices = np.setdiff1d(all_vertices, used_vertices)
        else:
            unused_vertices = np.arange(len(vertices))

        title = f"Unused vertices: {len(unused_vertices)}"

        if max_items is not None:
            unused_vertices = unused_vertices[:max_items]

        pts = vertices[unused_vertices]

        if len(pts) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers",
                    marker=dict(size=6, color="red"),
                    name="Unused vertices",
                    text=[f"vertex {idx}" for idx in unused_vertices],
                    hoverinfo="text+x+y+z",
                )
            )

    # ------------------------------------------------------------
    # Broken faces according to trimesh
    # ------------------------------------------------------------
    elif problem_type == "broken_faces":
        try:
            broken_faces = np.asarray(
                trimesh.repair.broken_faces(mesh),
                dtype=int,
            )
        except Exception:
            broken_faces = np.array([], dtype=int)

        title = f"Broken faces: {len(broken_faces)}"

        if max_items is not None:
            broken_faces = broken_faces[:max_items]

        _add_face_edges_and_centroids(
            fig,
            vertices,
            faces,
            broken_faces,
            line_color="red",
            marker_color="red",
            name="Broken faces",
        )

    # ------------------------------------------------------------
    # Connected components
    # ------------------------------------------------------------
    elif problem_type == "components":
        components = mesh.split(only_watertight=False)

        title = f"Connected components: {len(components)}"

        # Remove the base mesh, because components should be shown separately
        fig = go.Figure()

        for comp_id, comp in enumerate(components):
            v = np.asarray(comp.vertices)
            f = np.asarray(comp.faces)

            fig.add_trace(
                go.Mesh3d(
                    x=v[:, 0],
                    y=v[:, 1],
                    z=v[:, 2],
                    i=f[:, 0],
                    j=f[:, 1],
                    k=f[:, 2],
                    opacity=0.65,
                    name=f"Component {comp_id}",
                )
            )

    else:
        raise ValueError(
            "problem_type must be one of: "
            "'boundary_edges', 'non_manifold_edges', 'degenerate_faces', "
            "'duplicate_faces', 'unused_vertices', 'broken_faces', 'components'"
        )

    # ------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=True,
    )

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    fig.write_html(
        output_file,
        include_plotlyjs=include_plotlyjs,
        full_html=True,
    )

    return fig


def _add_face_edges_and_centroids(
        fig,
        vertices,
        faces,
        face_ids,
        line_color="red",
        marker_color="red",
        name="Problem faces",
):
    """
    Helper function to draw selected faces using their edges and centroids.
    """

    if len(face_ids) == 0:
        return

    x_lines, y_lines, z_lines = [], [], []
    cx, cy, cz = [], [], []
    labels = []

    for face_id in face_ids:
        tri = vertices[faces[face_id]]
        tri_closed = np.vstack([tri, tri[0]])

        x_lines += tri_closed[:, 0].tolist() + [None]
        y_lines += tri_closed[:, 1].tolist() + [None]
        z_lines += tri_closed[:, 2].tolist() + [None]

        centroid = tri.mean(axis=0)
        cx.append(centroid[0])
        cy.append(centroid[1])
        cz.append(centroid[2])
        labels.append(f"face {face_id}")

    fig.add_trace(
        go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode="lines",
            line=dict(width=6, color=line_color),
            name=name,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=cx,
            y=cy,
            z=cz,
            mode="markers",
            marker=dict(size=5, color=marker_color),
            name=f"{name} centroids",
            text=labels,
            hoverinfo="text+x+y+z",
        )
    )


def save_all_mesh_problem_html(mesh, output_dir="mesh_diagnostics_html"):
    """
    Save one HTML figure for each mesh diagnostic problem type.
    """

    os.makedirs(output_dir, exist_ok=True)

    problem_types = [
        "boundary_edges",
        "non_manifold_edges",
        "degenerate_faces",
        "duplicate_faces",
        "unused_vertices",
        "broken_faces",
        "components",
    ]

    output_files = {}

    for problem in problem_types:
        output_file = os.path.join(output_dir, f"{problem}.html")

        plot_mesh_problem_html(
            mesh,
            problem_type=problem,
            output_file=output_file,
        )

        output_files[problem] = output_file

    return output_files


def plot_layers_by_density(df, output_file=None):
    """
    Plot mascon points in 3D, coloring each layer according to density.

    Lower densities -> lighter colors
    Higher densities -> darker colors

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: x, y, z, layer_id, density_input
    output_file : str, optional
        If provided, saves the figure
    """

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    densities = df["density_input"].values

    # Normalize densities → [0,1]
    norm = colors.Normalize(vmin=densities.min(), vmax=densities.max())

    # Use a perceptually consistent colormap
    # "viridis" is a good default (scientifically recommended)
    cmap = cm.viridis

    # Plot layer by layer (so layers remain visually grouped)
    for layer in sorted(df["layer_id"].unique()):
        subset = df[df["layer_id"] == layer]

        density_value = subset["density_input"].iloc[0]
        color = cmap(norm(density_value))

        ax.scatter(
            subset["x"],
            subset["y"],
            subset["z"],
            color=color,
            s=5,
            label=f"Layer {layer} (ρ={density_value})"
        )

    # Add colorbar to show density scale
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    cbar = plt.colorbar(mappable, ax=ax, pad=0.1)
    cbar.set_label("Density")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Layered Mascon Model (Colored by Density)")

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(-0.05, 1.3),  # move outside (left, top)
        ncol=4,
        fontsize=8,
        frameon=False
    )

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    # plt.show()


def plot_layer_intersections(df, output_file=None, point_size=8):
    """
    Plot projections of mascon layers onto XY, XZ, and YZ planes
    using the same density-color mapping as the 3D plot.

    Internal layers are plotted in front of external layers.

    All subplots use the same axis limits so that XY, XZ, and YZ
    have the same physical and numerical scale.
    """

    required_cols = {"x", "y", "z", "layer_id", "density_input"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    densities = df["density_input"].values

    norm = colors.Normalize(vmin=densities.min(), vmax=densities.max())
    cmap = cm.viridis

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    plane_defs = [
        ("XY plane", "x", "y"),
        ("XZ plane", "x", "z"),
        ("YZ plane", "y", "z"),
    ]

    # Common axis limits using x, y, and z together
    all_coords = df[["x", "y", "z"]].values
    coord_min = all_coords.min()
    coord_max = all_coords.max()

    center = 0.5 * (coord_min + coord_max)
    half_range = 0.5 * (coord_max - coord_min)

    # Optional small margin
    margin = 0.05 * half_range
    lim_min = center - half_range - margin
    lim_max = center + half_range + margin

    # Plot outer layers first, inner layers last
    layer_order = sorted(df["layer_id"].unique(), reverse=True)

    for ax, (title, c1, c2) in zip(axes, plane_defs):
        for draw_order, layer in enumerate(layer_order):
            subset = df[df["layer_id"] == layer]

            density_value = subset["density_input"].iloc[0]
            color = cmap(norm(density_value))

            ax.scatter(
                subset[c1],
                subset[c2],
                s=point_size,
                color=color,
                alpha=0.9,
                zorder=draw_order + 1,
            )

        ax.set_title(title)
        ax.set_xlabel(c1.upper())
        ax.set_ylabel(c2.upper())

        # Same limits for all projections
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)

        # Same scale on both axes
        ax.set_aspect("equal", adjustable="box")

        ax.grid(True, linestyle="--", alpha=0.4)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    cbar = fig.colorbar(mappable, ax=axes, fraction=0.025, pad=0.04)
    cbar.set_label("Density")

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")



def compare_pot(asteroid, directory_path, compare_mascon=True, compare_expansion=True):
    """
    Compare potential models with respect to Werner.

    Parameters
    ----------
    asteroid : str
        Name of the asteroid.
    directory_path : str
        Base directory path containing the data.
    compare_mascon : bool, optional
        If True, compare Mascon potential with Werner. Default is True.
    compare_expansion : bool, optional
        If True, compare Expansion potential with Werner. Default is True.
    """

    # Build file paths using directory_path
    werner_file = f"{directory_path}/{asteroid}/pot_Werner.csv"
    mascon_file = f"{directory_path}/{asteroid}/pot_Mascon.csv"
    expansion_file = f"{directory_path}/{asteroid}/pot_Expansion.csv"

    faces_file = f"{directory_path}/{asteroid}/modified_f.dat"
    vertices_file = f"{directory_path}/{asteroid}/modified_v.dat"

    # --- Read Werner (reference)
    df_werner = pd.read_csv(werner_file)
    xyz_werner = jnp.asarray(df_werner[["x", "y", "z"]].values)
    pot_werner = jnp.asarray(df_werner["potential"].values)
    r = jnp.asarray(df_werner["r"].values)

    plt.figure()

    # --- Compare with Mascon if requested
    if compare_mascon:
        df_mascon = pd.read_csv(mascon_file)
        xyz_mascon = jnp.asarray(df_mascon[["x", "y", "z"]].values)
        pot_mascon = jnp.asarray(df_mascon["potential"].values)

        # Check shape
        if xyz_werner.shape != xyz_mascon.shape:
            raise ValueError("Werner and Mascon files have different number of points")

        # Check equality (with tolerance)
        same = jnp.all(jnp.isclose(xyz_werner, xyz_mascon, atol=1e-12))
        if not bool(same):
            raise ValueError("x, y, z columns are not identical between Werner and Mascon")

        # Compute difference relative to Werner
        d_pot_mascon = (pot_mascon - pot_werner) * 100 / pot_werner

        plt.scatter(r, d_pot_mascon, marker='.', color='blue', s=2)

    # --- Compare with Expansion if requested
    if compare_expansion:
        df_expansion = pd.read_csv(expansion_file)
        xyz_expansion = jnp.asarray(df_expansion[["x", "y", "z"]].values)
        pot_expansion = jnp.asarray(df_expansion["potential"].values)

        # Check shape
        if xyz_werner.shape != xyz_expansion.shape:
            raise ValueError("Werner and Expansion files have different number of points")

        # Check equality (with tolerance)
        same = jnp.all(jnp.isclose(xyz_werner, xyz_expansion, atol=1e-12))
        if not bool(same):
            raise ValueError("x, y, z columns are not identical between Werner and Expansion")

        # Compute difference relative to Werner
        d_pot_expansion = (pot_expansion - pot_werner) * 100 / pot_werner

        plt.scatter(r, d_pot_expansion, marker='.', color='green', s=2)

    # --- Read mesh for Brillouin radius
    faces = pd.read_table(
        faces_file,
        skiprows=0,
        header=None,
        sep=r"\s+",
        index_col=None,
        names=["f1", "f2", "f3"],
        low_memory=False,
        encoding="unicode_escape")

    vertices = pd.read_table(
        vertices_file,
        skiprows=0,
        header=None,
        sep=r"\s+",
        index_col=None,
        names=["x", "y", "z"],
        low_memory=False,
        encoding="unicode_escape",
    )

    plt.scatter(0, 100, marker='.', color='blue', s=20, label="Mascon vs Werner")
    plt.scatter(0, 100, marker='.', color='green', s=20, label="Expansion vs Werner")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    distances = cdist(mesh.vertices, np.array([mesh.center_mass]))
    R_brillouin = max(distances)[0]

    # --- Plot setup
    plt.axvline(x=R_brillouin, color='red', label='Brillouin radius')

    plt.xlabel(r"$r(\text{km})$")
    plt.ylabel("Relative Error (%)")

    plt.xlim(min(r), max(r))
    y_max = 2e-1
    plt.ylim(-y_max, y_max)

    plt.legend(loc='upper center', bbox_to_anchor=(0.45, 1.15), fancybox=True, shadow=True, ncol=6)

    plt.savefig(f"{directory_path}/{asteroid}/d_pot.png", dpi=300)
    plt.close()

