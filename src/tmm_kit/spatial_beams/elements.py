from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from tmm_kit.spatial_beams.constants import (
    SPATIAL_BEAM_DISPLACEMENT_B,
    SPATIAL_BEAM_DISPLACEMENT_N,
    SPATIAL_BEAM_DISPLACEMENT_T,
    SPATIAL_BEAM_FORCE_B,
    SPATIAL_BEAM_FORCE_N,
    SPATIAL_BEAM_FORCE_T,
    SPATIAL_BEAM_MOMENT_B,
    SPATIAL_BEAM_MOMENT_N,
    SPATIAL_BEAM_MOMENT_T,
    SPATIAL_BEAM_ROTATION_B,
    SPATIAL_BEAM_ROTATION_N,
    SPATIAL_BEAM_ROTATION_T,
    SPATIAL_BEAM_STATE_SIZE,
)
from tmm_kit.tmm.base import TMMElement

# Short aliases matching the notation of the model equations
U, W_N, W_B = (
    SPATIAL_BEAM_DISPLACEMENT_T,
    SPATIAL_BEAM_DISPLACEMENT_N,
    SPATIAL_BEAM_DISPLACEMENT_B,
)
PHI, THETA_N, THETA_B = (
    SPATIAL_BEAM_ROTATION_T,
    SPATIAL_BEAM_ROTATION_N,
    SPATIAL_BEAM_ROTATION_B,
)
M_T, M_N, M_B = SPATIAL_BEAM_MOMENT_T, SPATIAL_BEAM_MOMENT_N, SPATIAL_BEAM_MOMENT_B
N, Q_N, Q_B = SPATIAL_BEAM_FORCE_T, SPATIAL_BEAM_FORCE_N, SPATIAL_BEAM_FORCE_B


@dataclass(slots=True)
class SpatialBeamElement(TMMElement):
    """Straight spatial beam segment with constant distributed loads.

    Attributes
    ----------
    axial_stiffness
        Axial stiffness SE.
    torsional_stiffness
        Torsional stiffness GJ.
    bending_stiffness_n
        Stiffness EI_n of bending about the normal (deflection along the binormal).
    bending_stiffness_b
        Stiffness EI_beta of bending about the binormal (deflection along the normal).
    basis
        Local right-handed orthonormal basis of shape ``(3, 3)``. The rows are
        the tangent, normal and binormal vectors in global coordinates.
    f_t
        Distributed load along the tangent.
    f_rot
        Distributed torque about the tangent.
    q_n
        Distributed load along the normal.
    q_b
        Distributed load along the binormal.

    Notes
    -----
    Any stiffness can be ``np.inf``, which makes the beam rigid
    in the corresponding deformation mode.
    """

    vectorized: ClassVar[bool] = True

    axial_stiffness: float
    torsional_stiffness: float
    bending_stiffness_n: float
    bending_stiffness_b: float
    basis: npt.ArrayLike = field(default_factory=lambda: np.eye(3))
    f_t: float = 0.0
    f_rot: float = 0.0
    q_n: float = 0.0
    q_b: float = 0.0

    def __post_init__(self):
        """Normalize the basis into a float array."""
        self.basis = np.asarray(self.basis, dtype=np.float64)

    def get_transfer_eq(self, s: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return the transfer matrix K and right vector f."""
        s_arr = np.asarray(s, dtype=np.float64)
        s2 = s_arr**2 / 2.0
        s3 = s_arr**3 / 6.0
        s4 = s_arr**4 / 24.0

        # Compliances turn infinite stiffness into exact zeros
        c_axial = 1.0 / self.axial_stiffness
        c_torsion = 1.0 / self.torsional_stiffness
        c_n = 1.0 / self.bending_stiffness_n
        c_b = 1.0 / self.bending_stiffness_b

        K = np.zeros(
            (*s_arr.shape, SPATIAL_BEAM_STATE_SIZE, SPATIAL_BEAM_STATE_SIZE),
            dtype=np.float64,
        )
        f = np.zeros((*s_arr.shape, SPATIAL_BEAM_STATE_SIZE), dtype=np.float64)

        diagonal = np.arange(SPATIAL_BEAM_STATE_SIZE)
        K[..., diagonal, diagonal] = 1.0

        # Compression
        K[..., U, N] = -s_arr * c_axial
        f[..., U] = -self.f_t * s2 * c_axial
        f[..., N] = self.f_t * s_arr

        # Torsion
        K[..., PHI, M_T] = -s_arr * c_torsion
        f[..., PHI] = -self.f_rot * s2 * c_torsion
        f[..., M_T] = self.f_rot * s_arr

        # Bending in the (t, n) plane
        K[..., W_N, THETA_B] = s_arr
        K[..., W_N, M_B] = -s2 * c_b
        K[..., W_N, Q_N] = s3 * c_b
        K[..., THETA_B, M_B] = -s_arr * c_b
        K[..., THETA_B, Q_N] = s2 * c_b
        K[..., M_B, Q_N] = -s_arr
        f[..., W_N] = self.q_n * s4 * c_b
        f[..., THETA_B] = self.q_n * s3 * c_b
        f[..., M_B] = -self.q_n * s2
        f[..., Q_N] = self.q_n * s_arr

        # Bending in the (t, beta) plane
        K[..., W_B, THETA_N] = -s_arr
        K[..., W_B, M_N] = s2 * c_n
        K[..., W_B, Q_B] = s3 * c_n
        K[..., THETA_N, M_N] = -s_arr * c_n
        K[..., THETA_N, Q_B] = -s2 * c_n
        K[..., M_N, Q_B] = s_arr
        f[..., W_B] = self.q_b * s4 * c_n
        f[..., THETA_N] = -self.q_b * s3 * c_n
        f[..., M_N] = self.q_b * s2
        f[..., Q_B] = self.q_b * s_arr

        return K, f


def rotate_basis(basis: npt.ArrayLike, axis: int, angle: float) -> np.ndarray:
    """Rotate a basis about one of its own axes by the right-hand rule.

    Parameters
    ----------
    basis
        Basis of shape ``(3, 3)`` whose rows are the tangent, normal
        and binormal vectors in global coordinates.
    axis
        Index of the basis vector to rotate about
        (``AXIS_T``, ``AXIS_N`` or ``AXIS_B``).
    angle
        Rotation angle in radians.

    Returns
    -------
    rotated_basis
        Rotated basis of shape ``(3, 3)``.
    """
    basis_arr = np.asarray(basis, dtype=np.float64)

    # The two rotating vectors, in the cyclic order after the axis
    i, j = (axis + 1) % 3, (axis + 2) % 3

    a = np.eye(3)
    a[i, i] = a[j, j] = np.cos(angle)
    a[i, j] = np.sin(angle)
    a[j, i] = -np.sin(angle)

    return a @ basis_arr
