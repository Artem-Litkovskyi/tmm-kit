from dataclasses import dataclass

from tmm_kit.electrical.constants import STATE_E, STATE_I
from tmm_kit.tmm.base import ElementEdge, TMMEquation, TMMNode


@dataclass(slots=True)
class ElectricalNode(TMMNode):
    """Electrical circuit node."""

    def get_conjugation_eq(self) -> list[TMMEquation]:
        """Return the conjugation equations connecting elements."""
        eqs = []

        # Inbound elements equal E
        if self.inbound_elements:
            el_0 = self.inbound_elements[0]
            for el_i in self.inbound_elements[1:]:
                eqs.append(
                    TMMEquation(
                        [
                            (el_0, ElementEdge.END, STATE_E, 1),
                            (el_i, ElementEdge.END, STATE_E, -1),
                        ],
                        0,
                    ),
                )

        # Outbound elements equal E
        if self.outbound_elements:
            el_0 = self.outbound_elements[0]
            for el_i in self.outbound_elements[1:]:
                eqs.append(
                    TMMEquation(
                        [
                            (el_0, ElementEdge.START, STATE_E, 1),
                            (el_i, ElementEdge.START, STATE_E, -1),
                        ],
                        0,
                    ),
                )

        # Connect inbound and outbound elements E
        if self.inbound_elements and self.outbound_elements:
            eqs.append(
                TMMEquation(
                    [
                        (self.inbound_elements[0], ElementEdge.END, STATE_E, 1),
                        (self.outbound_elements[0], ElementEdge.START, STATE_E, -1),
                    ],
                    0,
                ),
            )

        # Kirchhoff’s Current Law (sum of inbound I = sum of outbound I)
        eqs.append(
            TMMEquation(
                [
                    *((i, ElementEdge.END, STATE_I, 1) for i in self.inbound_elements),
                    *(
                        (i, ElementEdge.START, STATE_I, -1)
                        for i in self.outbound_elements
                    ),
                ],
                0,
            ),
        )

        return eqs

    def get_conjugation_eq_n(self) -> int:
        """Return the number of conjugation equations."""
        inbound_n = len(self.inbound_elements)
        outbound_n = len(self.outbound_elements)
        return inbound_n + outbound_n
