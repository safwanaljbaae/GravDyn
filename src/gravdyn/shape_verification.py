from __future__ import annotations
import os
import sys
import trimesh
import numpy as np
from pathlib import Path
from gravdyn.shape_tools import load_vertices, load_faces
from gravdyn.plot_tools import save_mesh_projections, save_mesh_3d_html


class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def principal_axes(mesh):
    """
    Compute the principal axes of inertia of a triangular surface mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Triangular mesh object with attributes:
            - ``vertices`` of shape (N, 3)
            - ``faces`` of shape (M, 3)
        The mesh must have a valid mass distribution such that the inertia
        tensor can be computed via ``mesh.moment_inertia``.

    Returns
    -------
    eigenvectors : numpy.ndarray
        Array of shape (3, 3) whose columns are the orthonormal principal
        axes of inertia (sorted by increasing eigenvalue).
    M_4x4 : numpy.ndarray
        Homogeneous transformation matrix of shape (4, 4) that rotates the
        mesh from the original coordinate frame to the principal axes frame.
    angles : list of float
        Angles in degrees between each original coordinate axis and the
        corresponding principal axis direction.

    Notes
    -----
    The inertia tensor is diagonalized using eigen-decomposition. The
    eigenvectors define the principal directions, and the transformation
    matrix is constructed to align the mesh with these axes.

    The returned transformation can be directly applied using:
        ``mesh.apply_transform(M_4x4)``

    """

    # Calculate the moment of inertia tensor
    moment_inertia = mesh.moment_inertia

    # Diagonalize the moment of inertia tensor
    eigenvalues, eigenvectors = np.linalg.eigh(moment_inertia)

    # Sort the eigenvectors and eigenvalues by increasing moment of inertia
    indices = np.argsort(eigenvalues)[::1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    # fiend the angles between each axis of coordinates and the direction of the principal moments of inertia
    angles = []
    for i in range(len(eigenvectors)):
        a = eigenvectors[i]
        b = np.array([0, 0, 0])
        b[i] = 1
        # Compute the cosine of the angle between the vectors
        cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        # Convert the cosine to an angle in degrees
        angles.append(np.rad2deg(np.arccos(cos_angle)))

    # The principal axes of inertia are the columns of the eigenvector matrix
    axes_inertia = eigenvectors

    # Define the transformation matrix from the original coordinate system to the principal axes coordinate system
    T = eigenvectors.T  # the columns of the transpose matrix are the eigenvectors
    M_4x4 = np.eye(4)
    M_4x4[:3, :3] = T

    return eigenvectors, M_4x4, angles


def report_principal_axes(eigenvectors, new_eigenvectors, angles):
    """
    Print a diagnostic report for principal axes alignment.

    Parameters
    ----------
    eigenvectors : numpy.ndarray
        Original principal axes (shape (3, 3)), where columns represent
        eigenvectors of the inertia tensor before transformation.
    new_eigenvectors : numpy.ndarray
        Principal axes recomputed after applying the transformation.
        Ideally close to the identity matrix if alignment is successful.
    angles : list of float
        Rotation angles in degrees between the original coordinate axes
        and the principal axes.

    Returns
    -------
    None

    Notes
    -----
    The function evaluates the quality of alignment by computing the
    Frobenius norm:
        ``||R - I||``
    where ``R`` is the matrix of transformed eigenvectors and ``I`` is
    the identity matrix.

    A small error (e.g., < 1e-6) indicates successful alignment.
    """

    print("\n===================================================")
    print(" Principal Axes Transformation Report")
    print("===================================================")

    print("\nOriginal principal axes (columns = eigenvectors):")
    print(eigenvectors)

    print("\nRotation angles applied (deg):")
    print(angles)

    print("\nAfter transformation:")
    print("New principal axes (should be ~identity):")
    print(new_eigenvectors)

    # Check alignment quality
    identity = np.eye(3)
    error = np.linalg.norm(new_eigenvectors - identity)

    print("\nAlignment error (||R - I||): {:.3e}".format(error))

    if error < 1e-6:
        print("✔ Mesh successfully aligned with principal axes.")
    else:
        print("⚠ Warning: alignment may be inaccurate.")

    print("===================================================\n")


def shape_verification(
    asteroid_name: str,
    mass: float,
    density: float,
    base_dir: float,
    vertices_file: str,
    faces_file: str
) -> None:

    """
    Perform geometric and physical consistency checks on a triangular mesh
    representing an asteroid shape model.

    This function loads a mesh from vertex and face files, generates
    diagnostic visualizations, recenters the mesh at its center of mass,
    rescales it to match a reference volume derived from mass and density,
    aligns it with its principal axes of inertia, and saves the processed
    outputs.

    Parameters
    ----------
    asteroid_name : str
        Name of the target body. Used to create an output directory
        ``Data/<asteroid_name>/`` where all generated files are stored.
    mass : float
        Total mass of the body (in SI units). Used to compute a reference
        volume via ``V = M / rho``.
    density : float
        Bulk density of the body (in SI units). Must be consistent with
        ``mass``.
    base_dir : str
        Path to the directory containing the mesh data files (vertices and faces).
    vertices_file : str
        Name of the file containing vertex coordinates (not the full path).
        The file must exist inside ``data_folder`` and contain an array of
        shape ``(N, 3)`` with floating-point values representing ``(x, y, z)``.
    faces_file : str
        Name of the file containing triangular face connectivity (not the full path).
        The file must exist inside ``data_folder`` and contain an array of
        shape ``(M, 3)`` with integer indices referencing rows of the vertex array.

    Returns
    -------
    None
        This function does not return any value. It performs in-place
        transformations on the mesh and writes results (figures and
        processed vertex files) to disk.
    Raises
    ------
    FileNotFoundError
        If the vertices or faces file does not exist.
    ValueError
        If input files have invalid structure (e.g., incorrect number of columns).
    ZeroDivisionError
        If ``density`` is zero when computing the reference volume.
    RuntimeError
        If mesh construction fails due to invalid geometry (via ``trimesh``).

    Notes
    -----
    The mesh is assumed to be a triangular surface mesh with:
        - vertices: array of shape ``(N, 3)``
        - faces: array of shape ``(M, 3)``

    The workflow includes:
        1. Loading mesh geometry from disk.
        2. Generating initial 2D projections and 3D visualization.
        3. Translating the mesh to place the center of mass at the origin.
        4. Rescaling the mesh to match the physical volume ``V = M / rho``.
        5. Aligning the mesh with its principal axes of inertia.
        6. Generating updated visualizations after transformation.

    The volume scaling uses:
        ``scale_factor = (V_ref / V_mesh)^(1/3)``
    """

    log_file = Path(base_dir) / asteroid_name / "shape_verification.log"
    tee = Tee(log_file)
    sys.stdout = tee

    vertices_path = Path(base_dir) / asteroid_name / vertices_file
    faces_path = Path(base_dir) / asteroid_name / faces_file

    vertices = load_vertices(vertices_file=str(vertices_path))
    faces = load_faces(faces_file=str(faces_path))

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    save_mesh_projections(mesh, asteroid_name, base_dir, file_nam="shape_projection.png")
    save_mesh_3d_html(mesh, asteroid_name, base_dir, file_nam="shape_3d.html")
    
    # Compute center of mass (before)
    center_before = mesh.center_mass
    print(f"==========================")
    print(f"The name of the asteroid: {asteroid_name}")
    print(f"the considered mass is: {mass}")
    print(f"the considered density is: {density}")

    print("\n===== Centering Mesh =====")
    print(f"Original center of mass: {center_before}")

    # --- Apply translation to move COM to origin
    mesh.apply_translation(-center_before)

    # Compute center of mass (after)
    center_after = mesh.center_mass

    print(f"New center of mass: {center_after}")

    # Report result clearly
    if np.allclose(center_after, [0, 0, 0], atol=1e-8):
        print("✔ Mesh successfully recentered: center of mass is now at the origin (0, 0, 0).")
    else:
        print("Warning: mesh recentering may be inaccurate.")
        print("   Residual center of mass:", center_after)
    print("=================================\n")

    # --- Calculate the volume of the mesh
    volume = mesh.volume
    reference_volume = (mass / density) * 1.0e-12

    # Rescale factor
    scale_factor = (reference_volume / volume) ** (1 / 3)

    print("\n===== Mesh Rescaling Report =====")
    print(f"Original mesh volume        : {volume:.6e}")
    print(f"Reference volume (M/rho)    : {reference_volume:.6e}")
    print(f"Scaling factor applied      : {scale_factor:.6e}")

    # Apply scaling
    mesh.apply_scale(scale_factor)

    # New volume
    new_volume = mesh.volume

    print(f"Rescaled mesh volume        : {new_volume:.6e}")

    # Consistency check
    relative_error = abs(new_volume - reference_volume) / reference_volume
    print(f"Relative error              : {relative_error:.3e}")

    print("=================================\n")


    eigenvectors, M_4x4, angles = principal_axes(mesh)

    mesh.apply_transform(M_4x4)

    new_eigenvectors, new_M_4x4, new_angles = principal_axes(mesh)

    report_principal_axes(eigenvectors, new_eigenvectors, angles)
    
    save_mesh_projections(mesh, asteroid_name, base_dir, file_nam="modified_shape_projection.png")
    save_mesh_3d_html(mesh, asteroid_name, base_dir, file_nam="modified_shape_3d.html")

    # Build full output directory
    output_dir = Path(base_dir) / asteroid_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define full file path (with extension!)
    output_filename = output_dir / "modified_v.dat"
    precision = 5
    np.savetxt(
        output_filename,
        mesh.vertices,
        fmt=f"%.{precision}e"
    )
    print(f"Vertices saved to: {output_filename}")

    output_filename = output_dir / "modified_f.dat"
    np.savetxt(
        output_filename,
        mesh.faces,
        fmt="%d"
    )
    print(f"Faces saved to: {output_filename}")
    sys.stdout = tee.stdout
    tee.file.close()