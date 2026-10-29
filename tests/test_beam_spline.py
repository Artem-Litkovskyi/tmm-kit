import numpy as np

from tmm_kit import TMMSolver
from tmm_kit.beam_splines import (
    BEAM_STATE_SIZE,
    BeamElement,
    BeamPinnedSupport,
    BeamSpline,
    create_beam_elements_and_nodes,
    pinned_ends,
)

# Three supports (two elements conjugated at one node)
SUPPORT_T = [2, 4, 5]
SUPPORT_W = [-1, 4, 2]

# Start and end states for each element
DESIRED_EDGE_STATES = np.array(
    [
        [-1, 4, 0, -2.25],  # Element 1
        [4, -0.5, -4.5, -2.25],
        [4, -0.5, -4.5, 4.5],  # Element 2
        [2, -2.75, 0, 4.5],
    ],
    dtype=np.float64,
)

# Solver setup
ELEMENTS, NODES = create_beam_elements_and_nodes(
    support_t=SUPPORT_T,
    element_class=BeamElement,
    node_class=BeamPinnedSupport,
    node_data={"displacement": SUPPORT_W[1:-1]},
)

BCS = pinned_ends(SUPPORT_W[0], SUPPORT_W[-1])

SOLVER = TMMSolver(
    BEAM_STATE_SIZE,
    ELEMENTS,
    NODES,
    BCS,
)


def test_tmm_solver_beam_edge_states():
    result = SOLVER.solve()

    np.testing.assert_allclose(
        result.get_all_edge_states(),
        DESIRED_EDGE_STATES,
    )


def test_beam_spline_element_lengths():
    result = SOLVER.solve()
    spline = BeamSpline(result, SUPPORT_T)

    np.testing.assert_allclose(
        spline.element_lengths,
        np.diff(SUPPORT_T),
    )


def test_beam_spline_evaluate_element():
    result = SOLVER.solve()
    spline = BeamSpline(result, SUPPORT_T)

    element_lengths = np.diff(SUPPORT_T)

    np.testing.assert_allclose(
        spline.evaluate_element_state(0, [0, element_lengths[0]]),
        DESIRED_EDGE_STATES[[0, 1]],
    )
    np.testing.assert_allclose(
        spline.evaluate_element_state(1, [0, element_lengths[1]]),
        DESIRED_EDGE_STATES[[2, 3]],
    )


def test_beam_spline_evaluate_global():
    result = SOLVER.solve()
    spline = BeamSpline(result, SUPPORT_T)

    np.testing.assert_allclose(
        spline.evaluate_spline_state(SUPPORT_T),
        DESIRED_EDGE_STATES[[0, 2, 3]],
    )
