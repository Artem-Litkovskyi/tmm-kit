import numpy as np
import pytest

import tmm_kit.electrical as el
from tmm_kit import ElementEdge, TMMEquation, TMMResult, TMMSolver

# Electrical circuit setup
ELEMENTS = [
    el.ElectricalElement(1, 2, 2),  # 0. 2R resistor, 2E battery
    el.ElectricalElement(1, 6, 0),  # 1. 6R resistor
    el.ElectricalElement(1, 0, 1),  # 2. E battery
    el.ElectricalElement(1, 0, 0),  # 3. Wire
    el.ElectricalElement(1, 2, 0),  # 4. 2R resistor
    el.ElectricalElement(1, 4, 0),  # 5. 4R resistor
]

NODES = [
    el.ElectricalNode([2], [0, 3]),
    el.ElectricalNode([0, 4], [1]),
    el.ElectricalNode([1, 5], [2]),
]

# Replace the last node with boundary conditions with a fixed potential
# so that the system of equations has exactly one solution.
BCS = el.electrical_circuit_boundary_node([3], [4, 5], 1)

I1 = 9 / 14
I2 = 15 / 28
I3 = 2 / 7

E1 = 1.0
E2 = 1.714285
E3 = 0.0

# Start and end states for each element
DESIRED_EDGE_STATES = np.array(
    [
        [E1, I1],  # Element 0
        [E2, I1],
        [E2, I3],  # Element 1
        [E3, I3],
        [E3, I2],  # Element 2
        [E1, I2],
        [E1, I2 - I1],  # Element 3
        [E1, I2 - I1],
        [E1, I3 - I1],  # Element 4
        [E2, I3 - I1],
        [E1, I2 - I3],  # Element 5
        [E3, I2 - I3],
    ]
)


def test_tmm_solver():
    solver = TMMSolver(
        el.ELECTRICAL_STATE_SIZE,
        ELEMENTS,
        NODES,
        BCS,
    )

    result = solver.solve()

    np.testing.assert_allclose(
        result.get_all_edge_states(), DESIRED_EDGE_STATES, rtol=1e-6, atol=1e-6
    )


def test_tmm_result_evaluate_at_ends():
    result = TMMResult(ELEMENTS, DESIRED_EDGE_STATES)

    end_states = []

    for i in range(len(ELEMENTS)):
        end_states.append(result.evaluate(i, 1))

    np.testing.assert_allclose(
        np.array(end_states),
        DESIRED_EDGE_STATES[1::2],
        rtol=1e-6,
        atol=1e-6,
    )


def test_tmm_result_get_edge_state():
    result = TMMResult(ELEMENTS, DESIRED_EDGE_STATES)

    np.testing.assert_allclose(
        result.get_edge_state(0, ElementEdge.START),
        DESIRED_EDGE_STATES[0],
    )
    np.testing.assert_allclose(
        result.get_edge_state(0, ElementEdge.END),
        DESIRED_EDGE_STATES[1],
    )
    np.testing.assert_allclose(
        result.get_edge_state(0),
        DESIRED_EDGE_STATES[0:2],
    )
    np.testing.assert_allclose(
        result.get_edge_state([0, 2], ElementEdge.START),
        DESIRED_EDGE_STATES[[0, 4]],
    )


def test_tmm_result_get_all_edge_states():
    result = TMMResult(ELEMENTS, DESIRED_EDGE_STATES)

    np.testing.assert_allclose(
        result.get_all_edge_states(),
        DESIRED_EDGE_STATES,
    )

    np.testing.assert_allclose(
        result.get_all_edge_states(ElementEdge.START),
        DESIRED_EDGE_STATES[::2],
    )

    np.testing.assert_allclose(
        result.get_all_edge_states(ElementEdge.END),
        DESIRED_EDGE_STATES[1::2],
    )


def _solve_single_resistor(boundary_conditions):
    # A single 2R resistor, so the transfer equations give E_end = E_start - 2 I
    solver = TMMSolver(
        el.ELECTRICAL_STATE_SIZE,
        [el.ElectricalElement(1, 2, 0)],
        [],
        boundary_conditions,
    )
    return solver.solve().get_all_edge_states()


def test_tmm_solver_sums_repeated_addends():
    current_bc = TMMEquation((0, ElementEdge.START, 1, 1.0), 3.0)

    split_states = _solve_single_resistor(
        [
            TMMEquation(
                [(0, ElementEdge.START, 0, 0.5), (0, ElementEdge.START, 0, 0.5)],
                1.0,
            ),
            current_bc,
        ]
    )
    states = _solve_single_resistor(
        [TMMEquation((0, ElementEdge.START, 0, 1.0), 1.0), current_bc]
    )

    np.testing.assert_allclose(split_states, states)
    np.testing.assert_allclose(states, [[1.0, 3.0], [-5.0, 3.0]])


def test_tmm_solver_negative_element_index_counts_from_end():
    current_bc = TMMEquation((0, ElementEdge.START, 1, 1.0), 3.0)

    np.testing.assert_allclose(
        _solve_single_resistor(
            [TMMEquation((-1, ElementEdge.END, 0, 1.0), 1.0), current_bc]
        ),
        _solve_single_resistor(
            [TMMEquation((0, ElementEdge.END, 0, 1.0), 1.0), current_bc]
        ),
    )


def test_tmm_solver_rejects_element_index_out_of_range():
    current_bc = TMMEquation((0, ElementEdge.START, 1, 1.0), 3.0)

    with pytest.raises(ValueError):
        _solve_single_resistor(
            [TMMEquation((1, ElementEdge.START, 0, 1.0), 1.0), current_bc]
        )
