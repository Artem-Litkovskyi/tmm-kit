from collections.abc import Sequence, Sized

import numpy as np

from tmm_kit.beam_splines.elements import BeamElement
from tmm_kit.beam_splines.nodes import BeamDummyNode, BeamSplineNode
from tmm_kit.beam_splines.utils import angle_misalignment
from tmm_kit.tmm.base import TMMElement


def create_beam_elements_and_nodes(
    support_t: Sequence[float] | np.ndarray,
    element_class: type[TMMElement],
    node_class: type[BeamSplineNode],
    element_data: dict[str, Sequence] | None = None,
    node_data: dict[str, Sequence] | None = None,
) -> tuple[list[BeamElement], list[BeamSplineNode]]:
    """Populate elements and nodes of the same type.

    Parameters
    ----------
    support_t
        Coordinates of the support points.
    element_class
        TMM element class used for every spline segment.
    node_class
        Beam spline node class used at every spline junction.
    element_data
        Struct of Arrays for element instances.
        Key - argument name.
        Value - array of argument values per element.
    node_data
        Struct of Arrays for node instances.
        Key - argument name.
        Value - array of argument values per node.
    """
    # Process support t
    support_t_arr = np.asarray(support_t, dtype=np.float64)
    if support_t_arr.ndim != 1:
        raise ValueError("The support t coordinates must be 1D points")
    if support_t_arr.size < 2:
        raise ValueError("The number of support points must be greater or equal to 2")

    element_lengths = np.diff(support_t_arr)
    if np.any(element_lengths <= 0):
        raise ValueError(
            "Supports can't share the same t coordinate"
            "and must be sorted in the increasing order"
        )

    elements_n = support_t_arr.size - 1
    nodes_n = elements_n - 1

    # Process data
    element_data, node_data = _validate_data(
        element_data, node_data, elements_n, nodes_n
    )

    # Populate
    elements = []
    nodes = []
    for element_i, length in enumerate(element_lengths):
        element_kwargs = {k: v[element_i] for k, v in element_data.items()}
        elements.append(
            element_class(length, **element_kwargs)  # type: ignore[call-arg]
        )

        if element_i < nodes_n:
            node_kwargs = {k: v[element_i] for k, v in node_data.items()}
            nodes.append(node_class(element_i, element_i + 1, **node_kwargs))

    return elements, nodes


def create_cbs_elements_and_nodes(
    polyline: Sequence[Sequence[float]] | np.ndarray,
    element_class: type[TMMElement],
    node_class: type[BeamSplineNode],
    element_data: dict[str, Sequence] | None = None,
    node_data: dict[str, Sequence] | None = None,
    dummy_nodes_per_element: int | Sequence[int] | np.ndarray = 0,
) -> tuple[list[BeamElement], list[BeamSplineNode], np.ndarray, np.ndarray]:
    """Populate elements and nodes of the same type.

    Parameters
    ----------
    polyline
        Coordinates of the polyline points.
    element_class
        TMM element class used for every spline segment.
    node_class
        Beam spline node class used at every spline junction.
    element_data
        Struct of Arrays for element instances.
        Key - argument name.
        Value - array of argument values per element.
    node_data
        Struct of Arrays for node instances.
        Key - argument name.
        Value - array of argument values per node.
    dummy_nodes_per_element
        Number of dummy nodes inserted into each element:
        a scalar applies the same number to every element;
        an array specifies a number for each element.

    Return
    ------
        Elements list, nodes list, support t array, angle misalignment array
    """
    # Process polyline
    polyline_arr = np.asarray(polyline, dtype=np.float64)
    if polyline_arr.ndim != 2 or polyline_arr.shape[1] != 2:
        raise ValueError("The array points must be 2D vectors")
    if polyline_arr.shape[0] < 2:
        raise ValueError("The number of array points must be greater or equal to 2")

    element_lengths = np.linalg.norm(np.diff(polyline_arr, axis=0), axis=1)
    if np.any(element_lengths <= 0):
        raise ValueError(
            "Supports can't share the same t coordinate"
            "and must be sorted in the increasing order"
        )

    support_t = np.concatenate(([0.0], np.cumsum(element_lengths)))

    # Process dummy nodes number
    if isinstance(dummy_nodes_per_element, Sized):
        real_elements_n = len(dummy_nodes_per_element)
    else:
        real_elements_n = (support_t.size - 1) // (dummy_nodes_per_element + 1)
    real_nodes_n = real_elements_n - 1

    dummies_per_el_arr = np.asarray(dummy_nodes_per_element, dtype=int)
    if dummies_per_el_arr.size == 1:
        dummies_per_el_arr = np.repeat(dummies_per_el_arr.item(), real_elements_n)
    elif dummies_per_el_arr.size != real_elements_n:
        raise ValueError(
            f"Length of dummy_nodes_per_element ({dummies_per_el_arr.size}) "
            f"must match the number of elements ({real_elements_n})."
        )

    # Process data
    element_data, node_data = _validate_data(
        element_data, node_data, real_elements_n, real_nodes_n
    )

    if "displacement" not in node_data:
        node_data["displacement"] = np.repeat(0.0, real_nodes_n)

    if "angle_misalignment" in node_data:
        raise ValueError(
            'Do not include "angle_misalignment" in node_data: '
            "it's calculated automatically."
        )
    angle_diffs = angle_misalignment(polyline_arr)

    # Populate
    elements = []
    nodes = []

    element_i = 0
    last_sub_element_i = element_lengths.size - 1
    dummy_nodes_remaining = dummies_per_el_arr[element_i]

    for i, length in enumerate(element_lengths):
        element_kwargs = {k: v[element_i] for k, v in element_data.items()}
        elements.append(
            element_class(length, **element_kwargs)  # type: ignore[call-arg]
        )

        if i == last_sub_element_i:
            continue

        if dummy_nodes_remaining:
            nodes.append(BeamDummyNode(i, i + 1, angle_misalignment=angle_diffs[i + 1]))
            dummy_nodes_remaining -= 1
        else:
            node_kwargs = {k: v[element_i] for k, v in node_data.items()}
            node_kwargs["angle_misalignment"] = angle_diffs[i + 1]
            nodes.append(node_class(i, i + 1, **node_kwargs))
            element_i += 1
            dummy_nodes_remaining = dummies_per_el_arr[element_i]

    return elements, nodes, support_t, angle_diffs


def _validate_data(
    element_data: dict[str, Sequence] | None,
    node_data: dict[str, Sequence] | None,
    elements_n: int,
    nodes_n: int,
) -> tuple[dict[str, Sequence], dict[str, Sequence]]:
    if element_data is None:
        element_data = {}
    if node_data is None:
        node_data = {}

    for data, data_n, kind in (
        (element_data, elements_n, "elements"),
        (node_data, nodes_n, "nodes"),
    ):
        for key, values in data.items():
            values_n = len(values)
            if values_n != data_n:
                raise ValueError(
                    f'The number of values ({values_n}) for the "{key}" key '
                    f"doesn't match the number of {kind} ({data_n})"
                )

    return element_data, node_data
