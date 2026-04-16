import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import cm, colors


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

    ax.legend(loc="upper right", fontsize=8)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    # plt.show()


def plot_layer_intersections(df, output_file=None, point_size=8):
    """
    Plot projections of mascon layers onto XY, XZ, and YZ planes
    using the same density-color mapping as the 3D plot.

    Internal layers are plotted in front of external layers.
    """

    required_cols = {"x", "y", "z", "layer_id", "density_input"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    densities = df["density_input"].values

    norm = colors.Normalize(vmin=densities.min(), vmax=densities.max())
    cmap = cm.viridis

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    plane_defs = [
        ("XY plane", "x", "y"),
        ("XZ plane", "x", "z"),
        ("YZ plane", "y", "z"),
    ]

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
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.4)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=axes, fraction=0.025, pad=0.04)
    cbar.set_label("Density")

    # plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    # plt.show()