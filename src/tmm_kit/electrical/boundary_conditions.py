from tmm_kit.electrical.constants import STATE_E
from tmm_kit.tmm.base import ElementEdge, TMMEquation


def electrical_circuit_boundary_node(
    inbound_elements,
    outbound_elements,
    fixed_potential,
):
    """Boundary conditions that must replace one circuit node.

    It fixes the absolute potential level of the circuit to avoid
    getting an infinite number of valid solutions (which results
    in a singular global matrix error).
    """
    eqs = []

    # Fix inbound elements equal E
    for i in inbound_elements:
        eqs.append(
            TMMEquation((i, ElementEdge.END, STATE_E, 1), fixed_potential),
        )

    # Fix outbound elements equal E
    for i in outbound_elements:
        eqs.append(
            TMMEquation((i, ElementEdge.START, STATE_E, 1), fixed_potential),
        )

    return eqs
