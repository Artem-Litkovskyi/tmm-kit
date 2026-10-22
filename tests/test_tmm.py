import numpy as np

from tmm_kit import ElementEdge, TMMResult, TMMSolver
from tmm_kit.electrical import (
    ELECTRICAL_STATE_SIZE,
    ElectricalElement,
    ElectricalNode,
    electrical_circuit_boundary_node,
)

# Electrical circuit setup
ELEMENTS = [
    ElectricalElement(1, 2, 2),  # 0. 2R resistor, 2E battery
    ElectricalElement(1, 6, 0),  # 1. 6R resistor
    ElectricalElement(1, 0, 1),  # 2. E battery
    ElectricalElement(1, 0, 0),  # 3. Wire
    ElectricalElement(1, 2, 0),  # 4. 2R resistor
    ElectricalElement(1, 4, 0),  # 5. 4R resistor
]

NODES = [
    ElectricalNode([2], [0, 3]),
    ElectricalNode([0, 4], [1]),
    ElectricalNode([1, 5], [2]),
]

# Replace the last node with boundary conditions with a fixed potential
# so that the system of equations has exactly one solution.
BCS = electrical_circuit_boundary_node([3], [4, 5], 1)

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
        ELECTRICAL_STATE_SIZE,
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
