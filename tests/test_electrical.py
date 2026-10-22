import numpy as np

from tmm_kit import ElementEdge
from tmm_kit.electrical import STATE_E, STATE_I, ElectricalElement, ElectricalNode


def test_electrical_element_transfer_eq_for_resistor():
    element = ElectricalElement(2.5, 12.0, 0.0)

    matrix, right_vector = element.get_transfer_eq(s=element.length)

    np.testing.assert_array_equal(
        matrix,
        np.array([[1.0, -12.0], [0.0, 1.0]], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        right_vector,
        np.zeros(2, dtype=np.float64),
    )


def test_electrical_element_transfer_eq_for_battery():
    element = ElectricalElement(2.5, 0.0, 9.0)

    matrix, right_vector = element.get_transfer_eq(s=element.length)

    np.testing.assert_array_equal(
        matrix,
        np.eye(2, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        right_vector,
        np.array([9.0, 0.0], dtype=np.float64),
    )


def test_electrical_node_conjugation_equations():
    node = ElectricalNode([3, 4], [6, 7])

    equations = node.get_conjugation_eq()

    assert len(equations) == node.get_conjugation_eq_n() == 4

    # Equal potentials
    assert equations[0].left_side_addends == [
        (3, ElementEdge.END, STATE_E, 1),
        (4, ElementEdge.END, STATE_E, -1),
    ]
    assert equations[0].right_side == 0

    assert equations[1].left_side_addends == [
        (6, ElementEdge.START, STATE_E, 1),
        (7, ElementEdge.START, STATE_E, -1),
    ]
    assert equations[1].right_side == 0

    assert equations[2].left_side_addends == [
        (3, ElementEdge.END, STATE_E, 1),
        (6, ElementEdge.START, STATE_E, -1),
    ]
    assert equations[2].right_side == 0

    # The sum of inbound currents is equal to the sum of outbound currents
    assert equations[3].left_side_addends == [
        (3, ElementEdge.END, STATE_I, 1),
        (4, ElementEdge.END, STATE_I, 1),
        (6, ElementEdge.START, STATE_I, -1),
        (7, ElementEdge.START, STATE_I, -1),
    ]
    assert equations[3].right_side == 0
