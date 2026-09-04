from __future__ import annotations
import os
import sys
import trimesh
import numpy as np
from pathlib import Path
from gravdyn.shape_tools import load_vertices, load_faces
from gravdyn.plot_tools import save_mesh_projections, save_mesh_3d_html, save_all_mesh_problem_html


class Tee:
    def __init__(self, stdout, filename):
        self.stdout = stdout
        self.file = open(filename, "w", encoding="utf-8", errors="replace")

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


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

    # find the angles between each axis of coordinates and the direction of the principal moments of inertia
    angles = []
    for i in range(len(eigenvectors)):
        a = eigenvectors[i]
        b = np.array([0, 0, 0])
        b[i] = 1
        # Compute the cosine of the angle between the vectors  # **** To be verified ****
        cos_angle = abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
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
        print("Mesh successfully aligned with principal axes.")
    else:
        print("Warning: alignment may be inaccurate.")

    print("===================================================\n")


def diagnose_polyhedral_mesh(
        mesh: trimesh.Trimesh,
        tol: float = 1e-12,
        verbose: bool = True,
        return_elements: bool = True,
):
    """
    Diagnose common problems in a polyhedral triangular mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.

    tol : float
        Tolerance for detecting near-zero-area faces.

    verbose : bool
        If True, print the diagnostic report.

    return_elements : bool
        If True, include the actual problematic elements in the returned report.

    Returns
    -------
    report : dict
        Dictionary with diagnostic information.

        If return_elements=True, the dictionary also contains:

        report["problem_elements"]["boundary_edges"]
            Array of boundary/open edges, shape (n, 2).

        report["problem_elements"]["non_manifold_edges"]
            Array of non-manifold edges, shape (n, 2).

        report["problem_elements"]["degenerate_faces"]
            Array of degenerate face indices.

        report["problem_elements"]["duplicate_face_ids"]
            Array of duplicate face indices.

        report["problem_elements"]["unused_vertices"]
            Array of unused vertex indices.

        report["problem_elements"]["duplicate_vertex_coordinate_indices"]
            List of arrays. Each array contains vertex indices that share
            the same coordinates.

        report["problem_elements"]["broken_faces"]
            Array of broken face indices returned by trimesh.repair.broken_faces.

        report["problem_elements"]["components"]
            List of Trimesh components.
    """

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Input must be a trimesh.Trimesh object.")

    vertices = mesh.vertices
    faces = mesh.faces

    report = {}

    # ------------------------------------------------------------
    # Basic information
    # ------------------------------------------------------------
    report["num_vertices"] = int(len(vertices))
    report["num_faces"] = int(len(faces))

    report["is_watertight"] = bool(mesh.is_watertight)
    report["is_winding_consistent"] = bool(mesh.is_winding_consistent)
    report["is_volume"] = bool(mesh.is_volume)

    try:
        report["volume"] = float(mesh.volume)
    except Exception:
        report["volume"] = None

    # ------------------------------------------------------------
    # Edge topology
    # ------------------------------------------------------------
    edges = mesh.edges_sorted

    if len(edges) > 0:
        unique_edges, edge_counts = np.unique(
            edges,
            axis=0,
            return_counts=True,
        )

        boundary_edges = unique_edges[edge_counts == 1]
        non_manifold_edges = unique_edges[edge_counts > 2]
    else:
        boundary_edges = np.empty((0, 2), dtype=int)
        non_manifold_edges = np.empty((0, 2), dtype=int)

    report["num_boundary_edges"] = int(len(boundary_edges))
    report["num_non_manifold_edges"] = int(len(non_manifold_edges))

    # ------------------------------------------------------------
    # Degenerate faces
    # repeated vertices or near-zero area
    # ------------------------------------------------------------
    if len(faces) > 0:
        repeated_vertex_faces = np.where(
            (faces[:, 0] == faces[:, 1])
            | (faces[:, 1] == faces[:, 2])
            | (faces[:, 0] == faces[:, 2])
        )[0]

        zero_area_faces = np.where(mesh.area_faces <= tol)[0]
    else:
        repeated_vertex_faces = np.array([], dtype=int)
        zero_area_faces = np.array([], dtype=int)

    degenerate_faces = np.unique(
        np.concatenate([repeated_vertex_faces, zero_area_faces])
    ).astype(int)

    report["num_degenerate_faces"] = int(len(degenerate_faces))

    # ------------------------------------------------------------
    # Duplicate faces
    # independent of orientation
    # ------------------------------------------------------------
    if len(faces) > 0:
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

        duplicate_face_ids = np.array(duplicate_face_ids, dtype=int)
    else:
        duplicate_face_keys = np.empty((0, 3), dtype=int)
        duplicate_face_ids = np.array([], dtype=int)

    report["num_duplicate_faces"] = int(len(duplicate_face_ids))

    # ------------------------------------------------------------
    # Unused vertices
    # ------------------------------------------------------------
    if len(faces) > 0:
        used_vertices = np.unique(faces.reshape(-1))
        all_vertices = np.arange(len(vertices))
        unused_vertices = np.setdiff1d(all_vertices, used_vertices)
    else:
        unused_vertices = np.arange(len(vertices))

    report["num_unused_vertices"] = int(len(unused_vertices))

    # ------------------------------------------------------------
    # Exact duplicate vertex coordinates
    # ------------------------------------------------------------
    duplicate_vertex_coordinate_indices = []

    if len(vertices) > 0:
        unique_vertex_coords, inverse, vertex_coord_counts = np.unique(
            vertices,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )

        repeated_coord_ids = np.where(vertex_coord_counts > 1)[0]

        for coord_id in repeated_coord_ids:
            duplicate_ids = np.where(inverse == coord_id)[0]
            duplicate_vertex_coordinate_indices.append(duplicate_ids)

        num_duplicate_vertex_coordinate_groups = len(
            duplicate_vertex_coordinate_indices
        )

        num_duplicate_vertex_coordinates = sum(
            len(group) for group in duplicate_vertex_coordinate_indices
        )
    else:
        num_duplicate_vertex_coordinate_groups = 0
        num_duplicate_vertex_coordinates = 0

    report["num_duplicate_vertex_coordinate_groups"] = int(
        num_duplicate_vertex_coordinate_groups
    )
    report["num_exact_duplicate_vertex_coordinates"] = int(
        num_duplicate_vertex_coordinates
    )

    # ------------------------------------------------------------
    # Connected components
    # ------------------------------------------------------------
    try:
        components = mesh.split(only_watertight=False)
        report["num_connected_components"] = int(len(components))
        report["component_faces"] = [int(len(c.faces)) for c in components]
    except Exception:
        components = []
        report["num_connected_components"] = None
        report["component_faces"] = None

    # ------------------------------------------------------------
    # Broken faces according to trimesh repair
    # ------------------------------------------------------------
    try:
        broken_faces = trimesh.repair.broken_faces(mesh)
        broken_faces = np.asarray(broken_faces, dtype=int)
        report["num_broken_faces"] = int(len(broken_faces))
    except Exception:
        broken_faces = np.array([], dtype=int)
        report["num_broken_faces"] = None

    # ------------------------------------------------------------
    # Store examples for quick printing
    # ------------------------------------------------------------
    report["example_boundary_edges"] = boundary_edges[:5].tolist()
    report["example_non_manifold_edges"] = non_manifold_edges[:5].tolist()
    report["example_degenerate_faces"] = degenerate_faces[:5].tolist()
    report["example_duplicate_face_ids"] = duplicate_face_ids[:5].tolist()
    report["example_unused_vertices"] = unused_vertices[:5].tolist()
    report["example_broken_faces"] = broken_faces[:5].tolist()

    # ------------------------------------------------------------
    # Store full problematic elements
    # ------------------------------------------------------------
    if return_elements:
        report["problem_elements"] = {
            "boundary_edges": boundary_edges,
            "non_manifold_edges": non_manifold_edges,
            "degenerate_faces": degenerate_faces,
            "repeated_vertex_faces": repeated_vertex_faces,
            "zero_area_faces": zero_area_faces,
            "duplicate_face_keys": duplicate_face_keys,
            "duplicate_face_ids": duplicate_face_ids,
            "unused_vertices": unused_vertices,
            "duplicate_vertex_coordinate_indices": duplicate_vertex_coordinate_indices,
            "broken_faces": broken_faces,
            "components": components,
        }

    # ------------------------------------------------------------
    # Human-readable reasons
    # ------------------------------------------------------------
    problems = []

    if report["num_boundary_edges"] > 0:
        problems.append(
            f"Mesh has {report['num_boundary_edges']} boundary/open edges."
        )

    if report["num_non_manifold_edges"] > 0:
        problems.append(
            f"Mesh has {report['num_non_manifold_edges']} non-manifold edges."
        )

    if report["num_degenerate_faces"] > 0:
        problems.append(
            f"Mesh has {report['num_degenerate_faces']} degenerate faces."
        )

    if report["num_duplicate_faces"] > 0:
        problems.append(
            f"Mesh has {report['num_duplicate_faces']} duplicate face entries."
        )

    if report["num_unused_vertices"] > 0:
        problems.append(
            f"Mesh has {report['num_unused_vertices']} unused vertices."
        )

    if report["num_exact_duplicate_vertex_coordinates"] > 0:
        problems.append(
            f"Mesh has {report['num_exact_duplicate_vertex_coordinates']} vertices with exact duplicate coordinates."
        )

    if (
            report["num_connected_components"] is not None
            and report["num_connected_components"] > 1
    ):
        problems.append(
            f"Mesh has {report['num_connected_components']} disconnected components."
        )

    if not report["is_winding_consistent"]:
        problems.append(
            "Mesh has inconsistent face winding / normals."
        )

    if report["is_watertight"] and not report["is_volume"]:
        problems.append(
            "Mesh is watertight but not a valid volume. Normals or orientation may be wrong."
        )

    report["problems"] = problems

    # ------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------
    if verbose:
        print("Polyhedral mesh diagnostic")
        print("--------------------------")
        print(f"Vertices: {report['num_vertices']}")
        print(f"Faces: {report['num_faces']}")
        print()
        print(f"Watertight: {report['is_watertight']}")
        print(f"Winding consistent: {report['is_winding_consistent']}")
        print(f"Valid volume: {report['is_volume']}")
        print(f"Volume: {report['volume']}")
        print()
        print(f"Boundary edges: {report['num_boundary_edges']}")
        print(f"Non-manifold edges: {report['num_non_manifold_edges']}")
        print(f"Degenerate faces: {report['num_degenerate_faces']}")
        print(f"Duplicate faces: {report['num_duplicate_faces']}")
        print(f"Unused vertices: {report['num_unused_vertices']}")
        print(
            "Exact duplicate vertex coordinates: "
            f"{report['num_exact_duplicate_vertex_coordinates']}"
        )
        print(f"Connected components: {report['num_connected_components']}")
        print(f"Broken faces: {report['num_broken_faces']}")
        print()

        if problems:
            print("Problems found:")
            for problem in problems:
                print(f" - {problem}")
        else:
            print("No major topological problems found.")

    return report


def shape_verification(
        asteroid_name: str,
        mass: float,
        density: float,
        base_dir: str,
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

    original_stdout = sys.stdout
    tee = Tee(original_stdout, log_file)

    try:
        sys.stdout = tee

        vertices_path = Path(base_dir) / asteroid_name / vertices_file
        faces_path = Path(base_dir) / asteroid_name / faces_file

        vertices = load_vertices(vertices_file=str(vertices_path))
        faces = load_faces(faces_file=str(faces_path))

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        results_diagnose_polyhedral_mesh = diagnose_polyhedral_mesh(mesh, verbose=False)

        if len(results_diagnose_polyhedral_mesh['problems']):
            print(results_diagnose_polyhedral_mesh['problems'])
            output_dir = Path(base_dir) / asteroid_name
            output_files = save_all_mesh_problem_html(mesh, output_dir=output_dir)
            raise RuntimeError(f"Mesh has topological problems: {results_diagnose_polyhedral_mesh['problems']}")

        save_mesh_projections(mesh, asteroid_name, base_dir, file_nam="shape_projection.png")
        save_mesh_3d_html(mesh, asteroid_name, base_dir, file_nam="shape_3d.html")

        # Compute center of mass (before)
        center_before = mesh.center_mass
        print(f"==========================")
        print(f"The name of the asteroid: {asteroid_name}")
        print(f"the considered mass is: {mass}")
        print(f"the considered density is: {density}")
        print()
        print(f"Vertices: {len(vertices)}")
        print(f"Faces: {len(faces)}")

        print("\n===== Centering Mesh =====")
        print(f"Original center of mass: {center_before}")

        # --- Apply translation to move COM to origin
        mesh.apply_translation(-center_before)

        # Compute center of mass (after)
        center_after = mesh.center_mass

        print(f"New center of mass: {center_after}")

        # Report result clearly
        if np.allclose(center_after, [0, 0, 0], atol=1e-8):
            print("Mesh successfully recentered: center of mass is now at the origin (0, 0, 0).")
        else:
            print("Warning: mesh recentering may be inaccurate.")
            print("   Residual center of mass:", center_after)
        print("=================================\n")

        # --- Calculate the volume of the mesh
        volume = mesh.volume

        # Conversion factor
        conversion_factor = 1.0e-12

        reference_volume = (mass / density) * conversion_factor

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
        volume_equivalent_diameter = ((3.0*new_volume)/(4.0*np.pi)) **  (1 / 3)

        print(f"volume_equivalent_diameter  : {volume_equivalent_diameter:.6e}")

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

    finally:
        sys.stdout = original_stdout
        tee.close()