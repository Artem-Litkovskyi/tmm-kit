import numpy as np
import pytest

import tmm_kit.spatial_beams as sb
from tmm_kit import ElementEdge, TMMSolver

SE, GJ, EI_N, EI_B = 3.0, 0.8, 1.5, 2.5
F_T, F_ROT, Q_N, Q_B = 1.0, 0.5, -0.3, 0.7


def test_transfer_eq_satisfies_model():
    element = sb.SpatialBeamElement(
        2.0, SE, GJ, EI_N, EI_B, f_t=F_T, f_rot=F_ROT, q_n=Q_N, q_b=Q_B
    )
    y0 = np.random.default_rng(0).normal(size=sb.SPATIAL_BEAM_STATE_SIZE)

    def state(s):
        K, f = element.get_transfer_eq(s)
        return K @ y0 + f

    np.testing.assert_allclose(state(0.0), y0)

    s = np.linspace(0.1, 1.9, 7)
    h = 1e-4
    u, w_n, w_b, phi, theta_n, theta_b, M_t, M_n, M_b, N, Q_n, Q_b = state(s).T
    d = ((state(s + h) - state(s - h)) / (2 * h)).T

    expected = [
        -N / SE,  # u'
        theta_b,  # w_n'
        -theta_n,  # w_b'
        -M_t / GJ,  # phi'
        -M_n / EI_N,  # theta_n'
        -M_b / EI_B,  # theta_b'
        np.full_like(s, F_ROT),  # M_t'
        Q_b,  # M_n'
        -Q_n,  # M_b'
        np.full_like(s, F_T),  # N'
        np.full_like(s, Q_N),  # Q_n'
        np.full_like(s, Q_B),  # Q_b'
    ]
    np.testing.assert_allclose(d, expected, rtol=1e-6, atol=1e-6)


def test_infinite_stiffness_is_finite():
    inf = np.inf
    element = sb.SpatialBeamElement(
        1.0, inf, inf, inf, inf, f_t=F_T, f_rot=F_ROT, q_n=Q_N, q_b=Q_B
    )

    K, f = element.get_transfer_eq([0.0, 1.0])

    assert np.all(np.isfinite(K))
    assert np.all(np.isfinite(f))
    # Rigid body: no deformation caused by loads
    np.testing.assert_allclose(f[:, : sb.SPATIAL_BEAM_MOMENT], 0.0)
    np.testing.assert_allclose(
        K[:, : sb.SPATIAL_BEAM_MOMENT, sb.SPATIAL_BEAM_MOMENT :], 0.0
    )


def test_basis_accepts_nested_list():
    element = sb.SpatialBeamElement(
        1.0, SE, GJ, EI_N, EI_B, basis=[[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    )

    assert element.basis.dtype == np.float64
    assert element.basis.shape == (3, 3)
    np.testing.assert_allclose(
        sb.SpatialBeamElement(1.0, SE, GJ, EI_N, EI_B).basis, np.eye(3)
    )


def test_rotate_basis_follows_right_hand_rule():
    eye = np.eye(3)
    quarter = np.pi / 2

    about_b = sb.rotate_basis(eye, sb.AXIS_B, quarter)
    np.testing.assert_allclose(about_b, [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=1e-15)

    about_n = sb.rotate_basis(eye, sb.AXIS_N, quarter)
    np.testing.assert_allclose(about_n, [[0, 0, -1], [0, 1, 0], [1, 0, 0]], atol=1e-15)

    about_t = sb.rotate_basis(eye, sb.AXIS_T, quarter)
    np.testing.assert_allclose(about_t, [[1, 0, 0], [0, 0, 1], [0, -1, 0]], atol=1e-15)


@pytest.mark.parametrize("axis", [sb.AXIS_T, sb.AXIS_N, sb.AXIS_B])
def test_rotate_basis_keeps_right_handed_orthonormal_basis(axis):
    basis = sb.rotate_basis(sb.rotate_basis(np.eye(3), sb.AXIS_B, 0.4), sb.AXIS_T, -1.1)

    rotated = sb.rotate_basis(basis, axis, 0.77)

    np.testing.assert_allclose(rotated @ rotated.T, np.eye(3), atol=1e-15)
    np.testing.assert_allclose(np.linalg.det(rotated), 1.0)
    # The rotation axis itself does not move
    np.testing.assert_allclose(rotated[axis], basis[axis])


def _solve(elements, nodes, bcs):
    return TMMSolver(sb.SPATIAL_BEAM_STATE_SIZE, elements, nodes, bcs).solve()


def test_professor_problem():
    stiffness = (np.inf, 0.8, 1.0, 1.0)
    elements = [
        sb.SpatialBeamElement(1.0, *stiffness),
        sb.SpatialBeamElement(
            1.0, *stiffness, basis=sb.rotate_basis(np.eye(3), sb.AXIS_B, np.pi / 3)
        ),
        sb.SpatialBeamElement(1.0, *stiffness, f_t=1.0),
    ]
    nodes = [sb.SpatialBeamRigidNode([0], [1, 2], elements)]
    bcs = [
        *sb.clamped_end(0, ElementEdge.START),
        *sb.free_end(1, ElementEdge.END),
        *sb.sliding_end(2, ElementEdge.END, sb.AXIS_N),
    ]

    result = _solve(elements, nodes, bcs)

    s = np.linspace(0.0, 1.0, 5)
    expected = np.zeros((3, s.size, sb.SPATIAL_BEAM_STATE_SIZE))
    expected[0, :, sb.SPATIAL_BEAM_FORCE_T] = -1.0
    expected[2, :, sb.SPATIAL_BEAM_FORCE_T] = s - 1.0
    states = [result.evaluate(element_i, s) for element_i in range(3)]
    np.testing.assert_allclose(states, expected, atol=1e-12)


def _l_cantilever(rotation=None, node_elements=(0, 1)):
    """Clamped arm a along t, then arm b turned by 90 degrees with a load q_b."""
    a, b, EI, q = 1.5, 2.0, 1.3, 0.7
    bases = [np.eye(3), sb.rotate_basis(np.eye(3), sb.AXIS_B, np.pi / 2)]
    if rotation is not None:
        bases = [basis @ rotation for basis in bases]

    elements = [
        sb.SpatialBeamElement(a, 5.0, GJ, EI, EI, basis=bases[0]),
        sb.SpatialBeamElement(b, 5.0, GJ, EI, EI, basis=bases[1], q_b=q),
    ]
    in_el, out_el = node_elements
    nodes = [sb.SpatialBeamRigidNode([in_el], [out_el], elements)]
    bcs = [*sb.clamped_end(0, ElementEdge.START), *sb.free_end(1, ElementEdge.END)]

    tip = q * b**4 / (8 * EI) + q * b * a**3 / (3 * EI) + q * b**3 * a / (2 * GJ)
    return _solve(elements, nodes, bcs), tip


def test_l_cantilever_out_of_plane():
    result, tip = _l_cantilever()

    tip_state = result.get_edge_state(1, ElementEdge.END)

    np.testing.assert_allclose(tip_state[sb.SPATIAL_BEAM_DISPLACEMENT_B], tip)


def test_tee_with_unequal_branch_loads():
    """Clamped stem with two opposite branches loaded along the common normal."""
    a, b, q1, q2 = 1.5, 2.0, 0.7, 0.2
    stiffness = (SE, GJ, EI_N, EI_B)
    elements = [
        sb.SpatialBeamElement(a, *stiffness),
        sb.SpatialBeamElement(
            b,
            *stiffness,
            basis=sb.rotate_basis(np.eye(3), sb.AXIS_N, np.pi / 2),
            q_n=q1,
        ),
        sb.SpatialBeamElement(
            b,
            *stiffness,
            basis=sb.rotate_basis(np.eye(3), sb.AXIS_N, -np.pi / 2),
            q_n=q2,
        ),
    ]
    nodes = [sb.SpatialBeamRigidNode([0], [1, 2], elements)]
    bcs = [
        *sb.clamped_end(0, ElementEdge.START),
        *sb.free_end(1, ElementEdge.END),
        *sb.free_end(2, ElementEdge.END),
    ]

    result = _solve(elements, nodes, bcs)

    # The stem bends under the total load and twists under the load difference
    stem_deflection = (q1 + q2) * b * a**3 / (3 * EI_B)
    stem_twist = (q1 - q2) * b**2 * a / (2 * GJ)
    stem_end = result.get_edge_state(0, ElementEdge.END)
    np.testing.assert_allclose(
        stem_end[sb.SPATIAL_BEAM_DISPLACEMENT_N], stem_deflection
    )
    np.testing.assert_allclose(stem_end[sb.SPATIAL_BEAM_ROTATION_T], stem_twist)

    for element_i, q, twist_sign in ((1, q1, 1.0), (2, q2, -1.0)):
        np.testing.assert_allclose(
            result.get_edge_state(element_i, ElementEdge.END)[
                sb.SPATIAL_BEAM_DISPLACEMENT_N
            ],
            stem_deflection + twist_sign * stem_twist * b + q * b**4 / (8 * EI_B),
        )


def test_rotation_invariance():
    rotation = sb.rotate_basis(
        sb.rotate_basis(sb.rotate_basis(np.eye(3), sb.AXIS_B, 0.4), sb.AXIS_T, -1.1),
        sb.AXIS_N,
        2.3,
    )

    result, _ = _l_cantilever()
    rotated_result, _ = _l_cantilever(rotation, node_elements=(-2, -1))

    np.testing.assert_allclose(
        rotated_result.edge_states, result.edge_states, atol=1e-12
    )


def test_split_invariance():
    stiffness = (SE, GJ, EI_N, EI_B)
    loads = {"f_t": F_T, "f_rot": F_ROT, "q_n": Q_N, "q_b": Q_B}
    clamped = sb.clamped_end(0, ElementEdge.START)

    whole = [sb.SpatialBeamElement(2.0, *stiffness, **loads)]
    whole_result = _solve(whole, [], [*clamped, *sb.free_end(0, ElementEdge.END)])

    halves = [sb.SpatialBeamElement(1.0, *stiffness, **loads) for _ in range(2)]
    halves_result = _solve(
        halves,
        [sb.SpatialBeamRigidNode([0], [1], halves)],
        [*clamped, *sb.free_end(1, ElementEdge.END)],
    )

    np.testing.assert_allclose(
        halves_result.get_edge_state(0, ElementEdge.END),
        whole_result.evaluate(0, 1.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        halves_result.get_edge_state(1, ElementEdge.END),
        whole_result.get_edge_state(0, ElementEdge.END),
        atol=1e-12,
    )


def test_two_inbound_node():
    L, EI, q = 1.2, 1.3, 0.7
    stiffness = (SE, GJ, EI, EI)
    elements = [
        sb.SpatialBeamElement(L, *stiffness, q_b=q),
        sb.SpatialBeamElement(
            L, *stiffness, basis=sb.rotate_basis(np.eye(3), sb.AXIS_B, np.pi), q_b=q
        ),
    ]
    node = sb.SpatialBeamRigidNode([0, 1], [], elements)
    bcs = [*sb.clamped_end(0, ElementEdge.START), *sb.clamped_end(1, ElementEdge.START)]

    assert node.get_conjugation_eq_n() == 12
    assert len(node.get_conjugation_eq()) == 12

    result = _solve(elements, [node], bcs)

    # Clamped-clamped beam of length 2L under a uniform load
    midpoint = q * (2 * L) ** 4 / (384 * EI)
    for element_i in range(2):
        np.testing.assert_allclose(
            result.get_edge_state(element_i, ElementEdge.END)[
                sb.SPATIAL_BEAM_DISPLACEMENT_B
            ],
            midpoint,
        )


def test_pinned_end_propped_cantilever():
    L, EI, q = 2.0, 1.3, 0.7
    elements = [sb.SpatialBeamElement(L, SE, GJ, EI, EI, q_n=q)]
    bcs = [*sb.clamped_end(0, ElementEdge.START), *sb.pinned_end(0, ElementEdge.END)]

    result = _solve(elements, [], bcs)

    np.testing.assert_allclose(
        result.evaluate(0, L / 2)[sb.SPATIAL_BEAM_DISPLACEMENT_N], q * L**4 / (192 * EI)
    )
    np.testing.assert_allclose(
        result.get_edge_state(0, ElementEdge.END)[sb.SPATIAL_BEAM_MOMENT_B],
        0.0,
        atol=1e-12,
    )


def test_sliding_end_constrains_only_its_axis():
    L, EI, q = 2.0, 1.3, 0.7
    elements = [sb.SpatialBeamElement(L, SE, GJ, EI, EI, q_n=q, q_b=q)]
    bcs = [
        *sb.clamped_end(0, ElementEdge.START),
        *sb.sliding_end(0, ElementEdge.END, sb.AXIS_N),
    ]

    end_state = _solve(elements, [], bcs).get_edge_state(0, ElementEdge.END)

    # Propped along n, free cantilever along beta
    np.testing.assert_allclose(
        end_state[sb.SPATIAL_BEAM_DISPLACEMENT_N], 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        end_state[sb.SPATIAL_BEAM_DISPLACEMENT_B], q * L**4 / (8 * EI)
    )
    np.testing.assert_allclose(end_state[sb.SPATIAL_BEAM_FORCE_N], 3 * q * L / 8)
