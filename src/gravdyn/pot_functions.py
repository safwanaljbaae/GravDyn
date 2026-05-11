# -*- coding: utf-8 -*-
"""
# !===============================================================
# !==   Dr. Safwan Aljbaae, Ph.D.                               ==
# !==   Assistant Researcher                                    ==
# !==   Instituto de Astronomía y Ciencias Planetarias - INCT   ==
# !==   Universidad de Atacama - UDA                            ==
# !==   Copayapu 485, Copiapó 1531772, Chile                    ==
# !==   safwan.aljbaae@uda.cl                                   ==
# !==   safwan.aljbaae@gmail.com                                ==
# !===============================================================
# python3 -m pip install -r requirements.txt                    ==
# !===============================================================
"""
import jax
import numpy as np
from jax import config
import jax.numpy as jax_np
from typing import Tuple, Union
from typing import Any, Mapping

ArrayLike = Union[list, tuple, jax_np.ndarray]

config.update("jax_enable_x64", True)


def pot_point_mass(
        mu: float,
        stat: ArrayLike
) -> Tuple[jax_np.ndarray, jax_np.ndarray]:
    """
    Compute gravitational potential and acceleration for one or more points.

    Parameters
    ----------
    mu : float
        Gravitational parameter GM.
    stat : ArrayLike
        Position vector(s), shape (3,) or (N, 3).

    Returns
    -------
    p : jnp.ndarray
        Potential(s), shape (N,) or scalar-like.
    acc : jnp.ndarray
        Acceleration vector(s), shape (N, 3) or (3,).
    """

    stat = jax_np.asarray(stat, dtype=jax_np.float64)

    single_point = stat.ndim == 1
    if single_point:
        stat = stat[None, :]

    if stat.shape[1] != 3:
        raise ValueError("stat must have shape (3,) or (N, 3)")

    r = jax_np.linalg.norm(stat, axis=1)
    eps = 1e-35

    p = -mu / (r + eps)
    acc = -mu * stat / (r[:, None] + eps) ** 3

    if single_point:
        return p[0], acc[0]

    return p, acc


def pot_expansion(stat, f_pot_expansion, f_d_pot_expansion):
    """
    Compute potential and acceleration from the expansion model.

    Parameters
    ----------
    stat : array-like
        A single point with shape (3,) or multiple points with shape (N, 3).
    f_pot_expansion : callable
        Function for the potential, expected as:
            f_pot_expansion(x, y, z)
    f_d_pot_expansion : sequence of callables
        Functions for the acceleration components, expected as:
            f_d_pot_expansion[0](x, y, z)
            f_d_pot_expansion[1](x, y, z)
            f_d_pot_expansion[2](x, y, z)

    Returns
    -------
    p, acc
        If input is a single point:
            p   : scalar
            acc : shape (3,)
        If input is multiple points:
            p   : shape (N,)
            acc : shape (N, 3)
    """
    stat = jax_np.asarray(stat, dtype=jax_np.float64)

    single_point = stat.ndim == 1
    if single_point:
        stat = stat[None, :]

    if stat.shape[1] != 3:
        raise ValueError("stat must have shape (3,) or (N, 3)")

    def eval_one(point):
        x, y, z = point[0], point[1], point[2]

        p = f_pot_expansion(x, y, z)
        acc = jax_np.array(
            [
                f_d_pot_expansion[0](x, y, z),
                f_d_pot_expansion[1](x, y, z),
                f_d_pot_expansion[2](x, y, z),
            ],
            dtype=jax_np.float64,
        )

        return p, acc

    p, acc = jax.vmap(eval_one)(stat)

    if single_point:
        return p[0], acc[0]

    return p, acc


@jax.jit
def pot_mascon_jax(
        stat: ArrayLike,
        data_shape: Mapping[str, Any],
        gm_body: float = 0.0,
) -> Tuple[jax_np.ndarray, jax_np.ndarray]:
    """
    Compute gravitational potential and acceleration from a mascon model
    for one point or multiple points.

    Parameters
    ----------
    stat : array-like
        Field point(s), shape (3,) or (N, 3).
    data_shape : dict
        Dictionary containing JAX-compatible arrays:
            - 'x': x coordinates of mascons
            - 'y': y coordinates of mascons
            - 'z': z coordinates of mascons
            - 'mu': GM contribution of each mascon
    gm_body : float, optional
        Additional GM term added to each mascon, by default 0.0.

    Returns
    -------
    p : jax.numpy.ndarray
        Potential(s):
            - scalar if input is a single point
            - shape (N,) if input is multiple points
    a : jax.numpy.ndarray
        Acceleration(s):
            - shape (3,) if input is a single point
            - shape (N, 3) if input is multiple points
    """

    x = jax_np.asarray(data_shape["x"])
    y = jax_np.asarray(data_shape["y"])
    z = jax_np.asarray(data_shape["z"])
    mu = jax_np.asarray(data_shape["mu"])

    stat = jax_np.asarray(stat, dtype=x.dtype)

    single_point = stat.ndim == 1
    if single_point:
        stat = stat[None, :]

    if stat.ndim != 2 or stat.shape[1] != 3:
        raise ValueError("stat must have shape (3,) or (N, 3)")

    # stat: (N, 3)
    sx = stat[:, 0:1]  # (N, 1)
    sy = stat[:, 1:2]
    sz = stat[:, 2:3]

    # mascons: (M,)
    dx = x[None, :] - sx  # (N, M)
    dy = y[None, :] - sy
    dz = z[None, :] - sz

    r2 = dx * dx + dy * dy + dz * dz

    eps = jax_np.asarray(1e-30, dtype=r2.dtype)
    r2 = jax_np.maximum(r2, eps)

    inv_r = jax.lax.rsqrt(r2)  # (N, M)
    inv_r3 = inv_r * inv_r * inv_r  # (N, M)

    GM_i = mu + jax_np.asarray(gm_body, dtype=mu.dtype)  # (M,)

    # Potential for each field point
    p = jax_np.sum(GM_i[None, :] * inv_r, axis=1)  # (N,)

    # Acceleration for each field point
    scale = GM_i[None, :] * inv_r3  # (N, M)
    ax = jax_np.sum(dx * scale, axis=1)  # (N,)
    ay = jax_np.sum(dy * scale, axis=1)
    az = jax_np.sum(dz * scale, axis=1)

    a = jax_np.stack((ax, ay, az), axis=1)  # (N, 3)

    if single_point:
        return p[0], a[0]

    return p, a


def batched_pot_mascon(
        stat,
        data_shape,
        gm_body=0.0,
        batch_size=2000,
):
    """
    Evaluate mascon potential/acceleration in batches.
    """
    stat = np.asarray(stat, dtype=np.float64)

    if stat.ndim == 1:
        return pot_mascon_jax(stat, data_shape, gm_body)

    p_batches = []
    a_batches = []

    for i in range(0, len(stat), batch_size):
        batch = stat[i:i + batch_size]
        p_batch, a_batch = pot_mascon_jax(batch, data_shape, gm_body)

        p_batch = jax.block_until_ready(p_batch)
        a_batch = jax.block_until_ready(a_batch)

        p_batches.append(np.asarray(p_batch))
        a_batches.append(np.asarray(a_batch))

    p_all = np.concatenate(p_batches, axis=0)
    a_all = np.concatenate(a_batches, axis=0)

    return p_all, a_all


@jax.jit
def _compute_werner_gravity_batch(
    points_batch: jax_np.ndarray,
    sigma: float,
    centroid_e_j: jax_np.ndarray,
    centroid_f_j: jax_np.ndarray,
    edge_len_j: jax_np.ndarray,
    n_f_j: jax_np.ndarray,
    n_f_e_j: jax_np.ndarray,
    n_fp_e_j: jax_np.ndarray,
    r_e1_j: jax_np.ndarray,
    r_e2_j: jax_np.ndarray,
    r_f1_j: jax_np.ndarray,
    r_f2_j: jax_np.ndarray,
    r_f3_j: jax_np.ndarray,
    n_f1_for_edge: jax_np.ndarray,
    n_f2_for_edge: jax_np.ndarray,
):
    r_vec = points_batch[:, None, :]  # (B, 1, 3)

    # Edge terms
    re = centroid_e_j - r_vec
    diff1 = r_e1_j - r_vec
    diff2 = r_e2_j - r_vec

    r1 = jax_np.linalg.norm(diff1, axis=2)
    r2 = jax_np.linalg.norm(diff2, axis=2)

    eps = 1.0e-15
    denom_edge = jax_np.maximum(r1 + r2 - edge_len_j, eps)
    numer_edge = r1 + r2 + edge_len_j
    L_e = jax_np.log(numer_edge / denom_edge)

    # Face terms
    rf = centroid_f_j - points_batch[:, None, :]
    rf1 = r_f1_j - r_vec
    rf2 = r_f2_j - r_vec
    rf3 = r_f3_j - r_vec

    d1 = jax_np.linalg.norm(rf1, axis=2)
    d2 = jax_np.linalg.norm(rf2, axis=2)
    d3 = jax_np.linalg.norm(rf3, axis=2)

    cross_23 = jax_np.cross(rf2, rf3, axis=2)
    numer = jax_np.sum(rf1 * cross_23, axis=2)
    dot23 = jax_np.sum(rf2 * rf3, axis=2)
    dot31 = jax_np.sum(rf3 * rf1, axis=2)
    dot12 = jax_np.sum(rf1 * rf2, axis=2)
    denom = d1 * d2 * d3 + d1 * dot23 + d2 * dot31 + d3 * dot12

    omega = 2.0 * jax_np.arctan2(numer, denom)

    # Edge contribution sums
    dot1 = jax_np.sum(re * n_f1_for_edge, axis=2)
    dot2 = jax_np.sum(re * n_f_e_j, axis=2)
    dot3 = jax_np.sum(re * n_f2_for_edge, axis=2)
    dot4 = jax_np.sum(re * n_fp_e_j, axis=2)

    sum_e_U = jax_np.sum((dot1 * dot2 + dot3 * dot4) * L_e, axis=1)
    sum_e_A = jax_np.sum(
        (dot2[:, :, None] * n_f1_for_edge + dot4[:, :, None] * n_f2_for_edge)
        * L_e[:, :, None],
        axis=1,
    )

    # Face contribution sums
    nf_dot_rf = jax_np.sum(n_f_j * rf, axis=2)
    sum_f_U = jax_np.sum((nf_dot_rf ** 2) * omega, axis=1)
    sum_f_A = jax_np.sum(
        (nf_dot_rf[:, :, None] * n_f_j) * omega[:, :, None],
        axis=1,
    )
    sum_omega = jax_np.sum(omega, axis=1)

    U_batch = 0.5 * sigma * (sum_e_U - sum_f_U)
    A_batch = -sigma * (sum_e_A - sum_f_A)
    Lap_batch = -sigma * sum_omega

    return U_batch, A_batch, Lap_batch


def pot_werner_model(
        gm_body: float,
        stat: list[float] | list[list[float]] | np.ndarray | jax_np.ndarray,
        polyhedral_data: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute gravitational potential and acceleration with the polyhedral model.

    Parameters
    ----------
    gm_body : float
        Gravitational parameter of the body (G*M).
    stat : list[float] | list[list[float]] | np.ndarray | jax.numpy.ndarray
        Field point(s) in body-fixed coordinates.
        Accepted shapes:
            - (3,) for a single point [x, y, z]
            - (N, 3) for multiple points
    polyhedral_data : Mapping[str, Any]
        Dictionary with precomputed polyhedral geometry.

    Returns
    -------
    U : np.ndarray
        Potential at each point.
        Shape:
            - scalar-like / (1,) reduced to scalar for one point
            - (N,) for multiple points
    A : np.ndarray
        Acceleration at each point.
        Shape:
            - (3,) for one point
            - (N, 3) for multiple points
    """

    faces = polyhedral_data["faces"]
    vertices = polyhedral_data["vertices"]
    edges = polyhedral_data["edges"]
    centroid_edges = polyhedral_data["centroid_edges"]
    centroid_faces = polyhedral_data["centroid_faces"]
    e_e = polyhedral_data["e_e"]
    n_f = polyhedral_data["n_f"]
    n_f_e = polyhedral_data["n_f_e"]
    n_fp_e = polyhedral_data["n_fp_e"]
    r_e_1 = polyhedral_data["r_e_1"]
    r_e_2 = polyhedral_data["r_e_2"]
    r_f_1 = polyhedral_data["r_f_1"]
    r_f_2 = polyhedral_data["r_f_2"]
    r_f_3 = polyhedral_data["r_f_3"]

    stat = jax_np.asarray(stat, dtype=jax_np.float64)

    single_point = False
    if stat.ndim == 1:
        if stat.shape[0] != 3:
            raise ValueError("For a single point, stat must have shape (3,)")
        stat = stat[None, :]  # (1, 3)
        single_point = True
    elif stat.ndim == 2:
        if stat.shape[1] != 3:
            raise ValueError("For multiple points, stat must have shape (N, 3)")
    else:
        raise ValueError("stat must have shape (3,) or (N, 3)")

    # Volume and density computation
    v = vertices[faces]  # (M, 3, 3)                                                                                                                                                       LSP
    volume = np.abs(np.sum(v[:, 0] * np.cross(v[:, 1], v[:, 2]))) / 6.0

    sigma = gm_body / volume

    # Convert geometry arrays to JAX
    centroid_e_j = jax_np.asarray(centroid_edges, dtype=jax_np.float64)
    centroid_f_j = jax_np.asarray(centroid_faces, dtype=jax_np.float64)
    edge_len_j = jax_np.asarray(e_e, dtype=jax_np.float64)
    n_f_j = jax_np.asarray(n_f, dtype=jax_np.float64)
    n_f_e_j = jax_np.asarray(n_f_e, dtype=jax_np.float64)
    n_fp_e_j = jax_np.asarray(n_fp_e, dtype=jax_np.float64)
    r_e1_j = jax_np.asarray(r_e_1, dtype=jax_np.float64)
    r_e2_j = jax_np.asarray(r_e_2, dtype=jax_np.float64)
    r_f1_j = jax_np.asarray(r_f_1, dtype=jax_np.float64)
    r_f2_j = jax_np.asarray(r_f_2, dtype=jax_np.float64)
    r_f3_j = jax_np.asarray(r_f_3, dtype=jax_np.float64)

    face1_idx = edges[:, 2] - 1
    face2_idx = edges[:, 3] - 1
    n_f1_for_edge = jax_np.asarray(n_f[face1_idx], dtype=jax_np.float64)
    n_f2_for_edge = jax_np.asarray(n_f[face2_idx], dtype=jax_np.float64)

    U_b, A_b, _ = _compute_werner_gravity_batch(
        stat, sigma,
        centroid_e_j, centroid_f_j, edge_len_j,
        n_f_j, n_f_e_j, n_fp_e_j,
        r_e1_j, r_e2_j,
        r_f1_j, r_f2_j, r_f3_j,
        n_f1_for_edge, n_f2_for_edge,
    )

    U_b = np.asarray(U_b)
    A_b = np.asarray(A_b)

    if single_point:
        return U_b[0], A_b[0]

    return U_b, A_b


def batched_werner_potential(
        stat,
        gm_body,
        polyhedral_data,
        batch_size=2000,
):
    """
    Evaluate polyhedral potential and acceleration in batches.

    Parameters
    ----------
    stat : np.ndarray
        Array of shape (N, 3)
    gm_body : float
        Gravitational parameter
    polyhedral_data : dict
        Prepared polyhedral model data
    batch_size : int
        Number of points per batch

    Returns
    -------
    p_all : np.ndarray
        Potential array of shape (N,)
    acc_all : np.ndarray
        Acceleration array of shape (N, 3)
    """
    n_total = stat.shape[0]

    p_batches = []
    acc_batches = []

    for i in range(0, n_total, batch_size):
        batch = stat[i:i + batch_size]

        p_batch, acc_batch = pot_werner_model(
            gm_body=gm_body,
            stat=batch,
            polyhedral_data=polyhedral_data,
        )

        # Important for JAX: force execution before storing
        p_batch = jax.block_until_ready(p_batch)
        acc_batch = jax.block_until_ready(acc_batch)

        p_batches.append(np.asarray(p_batch))
        acc_batches.append(np.asarray(acc_batch))

        # print(f"Processed batch {i // batch_size + 1} / {(n_total - 1) // batch_size + 1}")

    p_all = np.concatenate(p_batches, axis=0)
    acc_all = np.concatenate(acc_batches, axis=0)

    return p_all, acc_all


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def compute_pseudo_potential(
    stat: np.ndarray,
    pot: np.ndarray,
    rot_period_hours: float,
) -> np.ndarray:
    """
    Compute the pseudo potential by combining gravitational and centrifugal potentials.

    The pseudo potential is computed as:
        pseudo = -fat - pot
    where:
        fat = (omega**2) * (x**2 + y**2) / 2.0  (centrifugal potential, rotation around Z)
        omega = 2*pi / (rot_period_hours * 3600)
        pot is the gravitational potential

    Parameters
    ----------
    stat : np.ndarray
        Array of shape (N, 3) containing the x, y, z coordinates of the points.
    pot : np.ndarray
        Gravitational potential array of shape (N,).
    rot_period_hours : float
        Rotation period in hours.

    Returns
    -------
    pseudo_pot : np.ndarray
        Pseudo potential array of shape (N,).
    """
    omega = 2.0 * np.pi / (rot_period_hours * 3600.0)
    x = stat[:, 0]
    y = stat[:, 1]

    fat = (omega ** 2) * (x ** 2 + y ** 2) / 2.0

    pseudo_pot = -fat - pot

    return pseudo_pot


def save_potential_to_file(
    stat: np.ndarray,
    pot: np.ndarray,
    pseudo_pot: np.ndarray | None,
    output_path: str,
) -> None:
    """
    Save potential data to a file.

    Parameters
    ----------
    stat : np.ndarray
        Array of shape (N, 3) containing the x, y, z coordinates.
    pot : np.ndarray
        Gravitational potential array of shape (N,).
    pseudo_pot : np.ndarray | None
        Pseudo potential array of shape (N,), or None to skip.
    output_path : str
        Path to the output file.

    Returns
    -------
    None
    """
    data = {
        'x': stat[:, 0],
        'y': stat[:, 1],
        'z': stat[:, 2],
        'potential': pot,
    }

    if pseudo_pot is not None:
        data['pseudo_potential'] = pseudo_pot

    np.savez_compressed(output_path, **data)
