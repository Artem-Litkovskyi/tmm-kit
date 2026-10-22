from abc import abstractmethod
from dataclasses import dataclass

from tmm_kit.beam_splines.constants import (
    BEAM_DISPLACEMENT,
    BEAM_FORCE,
    BEAM_MOMENT,
    BEAM_SLOPE,
    BEAM_STATE_SIZE,
)
from tmm_kit.tmm.base import (
    ElementEdge,
    TMMEquation,
    TMMNode,
    continuity_equation,
    offset_equation,
)


@dataclass(slots=True, init=False)
class BeamSplineNode(TMMNode):
    """Base class for beam spline nodes.

    Can have only one inbound and one outbound elements.

    Attributes
    ----------
    angle_misalignment
        Needed for compatibility with corotational beam splines.
    """

    angle_misalignment: float

    # Custom init limits the number of inbound and outbound elements
    def __init__(
        self,
        inbound_element: int,
        outbound_element: int,
        angle_misalignment: float | None = None,
        **kwargs,
    ):
        super(BeamSplineNode, self).__init__(
            inbound_elements=[inbound_element],
            outbound_elements=[outbound_element],
        )

        if angle_misalignment is None:
            self.angle_misalignment = 0.0
        else:
            self.angle_misalignment = angle_misalignment

        # kwargs allow the init to be inherited by subclasses with additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def get_conjugation_eq(self) -> list[TMMEquation]:
        """Return the conjugation equations connecting elements."""
        pass

    def get_conjugation_eq_n(self) -> int:
        """Return the number of conjugation equations."""
        return BEAM_STATE_SIZE


@dataclass(slots=True, init=False)
class BeamPinnedSupport(BeamSplineNode):
    """Beam spline pinned support node.

    Set displacement.
    Enforce continuity of slope and moment.

    Attributes
    ----------
    displacement
        Support displacement at the node.
    """

    displacement: float

    def get_conjugation_eq(self) -> list[TMMEquation]:
        """Return the conjugation equations connecting elements."""
        in_el = self.inbound_elements[0]
        out_el = self.outbound_elements[0]

        return [
            TMMEquation(
                (in_el, ElementEdge.END, BEAM_DISPLACEMENT, 1),
                self.displacement,
            ),
            TMMEquation(
                (out_el, ElementEdge.START, BEAM_DISPLACEMENT, 1),
                self.displacement,
            ),
            offset_equation(in_el, out_el, BEAM_SLOPE, self.angle_misalignment),
            continuity_equation(in_el, out_el, BEAM_MOMENT),
        ]


@dataclass(slots=True, init=False)
class BeamElasticSupport(BeamSplineNode):
    """Beam spline elastic support node.

    Set the force change proportional to the difference between
    the support displacement and the beam displacement.
    Enforce continuity of displacement, slope and moment.

    Attributes
    ----------
    displacement
        Support displacement at the node.
    stiffness
        Spring stiffness of the support.
    """

    displacement: float
    stiffness: float

    def get_conjugation_eq(self) -> list[TMMEquation]:
        """Return the conjugation equations connecting elements."""
        in_el = self.inbound_elements[0]
        out_el = self.outbound_elements[0]

        return [
            continuity_equation(in_el, out_el, BEAM_DISPLACEMENT),
            offset_equation(in_el, out_el, BEAM_SLOPE, self.angle_misalignment),
            continuity_equation(in_el, out_el, BEAM_MOMENT),
            TMMEquation(
                [
                    (out_el, ElementEdge.START, BEAM_FORCE, 1),
                    (in_el, ElementEdge.END, BEAM_FORCE, -1),
                    (in_el, ElementEdge.END, BEAM_DISPLACEMENT, self.stiffness),
                ],
                self.stiffness * self.displacement,
            ),
        ]


@dataclass(slots=True, init=False)
class BeamElasticSupportInverted(BeamSplineNode):
    """Beam spline elastic support node.

    A very stiff BeamElasticSupport may result in high error.
    In such cases use BeamElasticSupportInverted instead.

    Attributes
    ----------
    displacement
        Support displacement at the node.
    compliance
        Spring compliance of the support (compliance = 1 / stiffness).
    """

    displacement: float
    compliance: float

    def get_conjugation_eq(self) -> list[TMMEquation]:
        """Return the conjugation equations connecting elements."""
        in_el = self.inbound_elements[0]
        out_el = self.outbound_elements[0]

        return [
            continuity_equation(in_el, out_el, BEAM_DISPLACEMENT),
            offset_equation(in_el, out_el, BEAM_SLOPE, self.angle_misalignment),
            continuity_equation(in_el, out_el, BEAM_MOMENT),
            TMMEquation(
                [
                    (in_el, ElementEdge.END, BEAM_DISPLACEMENT, 1),
                    (out_el, ElementEdge.START, BEAM_FORCE, -self.compliance),
                    (in_el, ElementEdge.END, BEAM_FORCE, self.compliance),
                ],
                self.displacement,
            ),
        ]


@dataclass(slots=True, init=False)
class BeamDummyNode(BeamSplineNode):
    """Beam spline dummy node.

    Enforce continuity of all beam state variables.
    """

    def get_conjugation_eq(self) -> list[TMMEquation]:
        """Return the conjugation equations connecting elements."""
        in_el = self.inbound_elements[0]
        out_el = self.outbound_elements[0]

        return [
            continuity_equation(in_el, out_el, BEAM_DISPLACEMENT),
            offset_equation(in_el, out_el, BEAM_SLOPE, self.angle_misalignment),
            continuity_equation(in_el, out_el, BEAM_MOMENT),
            continuity_equation(in_el, out_el, BEAM_FORCE),
        ]
