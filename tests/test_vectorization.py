from typing import ClassVar

import numpy as np

import tmm_kit.beam_splines as bs
from tmm_kit import TMMSolver

POLYLINE = np.array(
    [[0, 0], [1, 1], [2, 1], [3, 2], [4, 4], [5, 4.5], [6, 4], [7, 2]], dtype=float
)
ELEMENTS_N = len(POLYLINE) - 1
STIFFNESS = np.linspace(0.5, 2.0, ELEMENTS_N - 1)
BANDWIDTH = np.linspace(0.6, 1.2, ELEMENTS_N)


# Copies of the library classes that the solver must process one by one
class LoopElement(bs.BeamElement):
    vectorized: ClassVar[bool] = False


class LoopElementOnElasticFoundation(bs.BeamElementOnElasticFoundation):
    vectorized: ClassVar[bool] = False


class LoopElasticSupport(bs.BeamElasticSupport):
    vectorized: ClassVar[bool] = False


class LoopDummyNode(bs.BeamDummyNode):
    vectorized: ClassVar[bool] = False


LOOP_CLASSES = {
    bs.BeamElement: LoopElement,
    bs.BeamElementOnElasticFoundation: LoopElementOnElasticFoundation,
    bs.BeamElasticSupport: LoopElasticSupport,
    bs.BeamDummyNode: LoopDummyNode,
}


def _create_system():
    """Create a system with two element classes and two node classes."""
    elements, nodes, support_t, _ = bs.create_cbs_elements_and_nodes(
        polyline=POLYLINE,
        element_class=bs.BeamElementOnElasticFoundation,
        node_class=bs.BeamElasticSupport,
        element_data={
            "start_foundation": np.linspace(-0.1, 0.1, ELEMENTS_N),
            "end_foundation": np.linspace(0.2, -0.2, ELEMENTS_N),
            "bandwidth": BANDWIDTH,
        },
        node_data={"stiffness": STIFFNESS},
    )
    # Mix the classes so that the groups are not contiguous
    for i in (1, 4):
        elements[i] = bs.BeamElement(elements[i].length)
    for i in (0, 3):
        nodes[i] = bs.BeamDummyNode(i, i + 1, nodes[i].angle_misalignment)

    return elements, nodes, support_t


def _as_loop(objects):
    """Replace vectorized objects with their one-by-one copies."""
    loop_objects = []
    for obj in objects:
        loop_obj = object.__new__(LOOP_CLASSES[type(obj)])
        for name in type(obj).__dataclass_fields__:
            setattr(loop_obj, name, getattr(obj, name))
        loop_objects.append(loop_obj)
    return loop_objects


def _solve(elements, nodes):
    bcs = bs.elastic_ends(0.0, 0.1, 1.0, 2.0)
    return TMMSolver(bs.BEAM_STATE_SIZE, elements, nodes, bcs).solve()


def test_library_classes_are_vectorized():
    assert all(cls.vectorized for cls in LOOP_CLASSES)


def test_vectorized_solver_matches_loop():
    elements, nodes, _ = _create_system()

    result = _solve(elements, nodes)
    loop_result = _solve(_as_loop(elements), _as_loop(nodes))

    assert np.any(result.edge_states != 0.0)
    np.testing.assert_allclose(
        result.edge_states, loop_result.edge_states, rtol=1e-10, atol=1e-12
    )


def test_evaluate_with_element_array_matches_scalar_elements():
    elements, nodes, _ = _create_system()
    result = _solve(elements, nodes)
    loop_result = _solve(_as_loop(elements), _as_loop(nodes))

    element_i = np.array([[5, 0, 1], [4, 5, 1]])
    s = np.array([[0.0, 0.3, 0.9], [1.1, 0.2, 0.5]])

    states = result.evaluate(element_i, s)
    assert states.shape == (2, 3, bs.BEAM_STATE_SIZE)

    for index in np.ndindex(s.shape):
        np.testing.assert_allclose(
            states[index], result.evaluate(int(element_i[index]), float(s[index]))
        )
    np.testing.assert_allclose(
        loop_result.evaluate(element_i, s), states, rtol=1e-10, atol=1e-12
    )


def test_evaluate_broadcasts_element_array_with_scalar_s():
    elements, nodes, _ = _create_system()
    result = _solve(elements, nodes)

    np.testing.assert_allclose(
        result.evaluate([0, 3], 0.4),
        [result.evaluate(0, 0.4), result.evaluate(3, 0.4)],
    )


def test_spline_evaluation_with_unsorted_t_matches_elements():
    elements, nodes, support_t = _create_system()
    spline = bs.CorotationalBeamSpline(_solve(elements, nodes), support_t, POLYLINE)

    t = np.array([support_t[5] + 0.2, 0.1, support_t[2] + 0.3, support_t[5] + 0.1])
    element_i = [5, 0, 2, 5]

    for evaluate_spline, evaluate_element in (
        (spline.evaluate_spline_state, spline.evaluate_element_state),
        (spline.evaluate_spline_point, spline.evaluate_element_point),
        (spline.evaluate_spline_normal, spline.evaluate_element_normal),
        (spline.evaluate_spline_curvature, spline.evaluate_element_curvature),
    ):
        values = evaluate_spline(t)
        for value, el_i, t_i in zip(values, element_i, t):
            np.testing.assert_allclose(
                value, evaluate_element(el_i, [t_i - support_t[el_i]])[0]
            )


def test_refine_polyline_with_different_dummy_points_per_segment():
    elements, nodes, support_t = _create_system()
    spline = bs.CorotationalBeamSpline(_solve(elements, nodes), support_t, POLYLINE)
    dummies = [0, 2, 1, 0, 3, 0, 1]

    refined = bs.refine_polyline(spline, dummies)

    mask = bs.real_points_mask(len(POLYLINE), dummies)
    np.testing.assert_allclose(refined[mask], spline.evaluate_spline_point(support_t))

    lengths = spline.element_lengths
    desired_dummies = np.concatenate(
        [
            spline.evaluate_element_point(i, lengths[i] * np.arange(1, n + 1) / (n + 1))
            for i, n in enumerate(dummies)
        ]
    )
    np.testing.assert_allclose(refined[~mask], desired_dummies)
