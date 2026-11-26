from dataclasses import dataclass

import numpy as np
import pytest

import tmm_kit.beam_splines as bs
import tmm_kit.electrical as el
import tmm_kit.spatial_beams as sb
from tmm_kit import TMMElement, TMMResult

ELEMENTS = [
    bs.BeamElement(2.0),
    bs.BeamElementStretched(2.0, 0.7),
    bs.BeamElementCompressed(2.0, 0.7),
    bs.BeamElementOnElasticFoundation(2.0, 0.3, -0.5, 0.8),
    el.ElectricalElement(1.0, 2.0, 3.0),
    sb.SpatialBeamElement(
        2.0, 3.0, 0.8, 1.5, 2.5, f_t=1.0, f_rot=0.5, q_n=-0.3, q_b=0.7
    ),
]


@pytest.mark.parametrize("element", ELEMENTS, ids=lambda el: type(el).__name__)
def test_transfer_eq_array_matches_scalar(element):
    s = np.linspace(0.0, element.length, 6).reshape(2, 3)

    K, f = element.get_transfer_eq(s)
    state_size = f.shape[-1]

    assert K.shape == (2, 3, state_size, state_size)
    assert f.shape == (2, 3, state_size)

    for index in np.ndindex(s.shape):
        K_scalar, f_scalar = element.get_transfer_eq(float(s[index]))
        assert K_scalar.shape == (state_size, state_size)
        assert f_scalar.shape == (state_size,)
        np.testing.assert_allclose(K[index], K_scalar)
        np.testing.assert_allclose(f[index], f_scalar)


@dataclass(slots=True)
class _ScalarOnlyElement(TMMElement):
    def get_transfer_eq(self, s):
        return np.eye(2), np.zeros(2)


def test_evaluate_rejects_wrong_transfer_eq_shape():
    result = TMMResult(elements=[_ScalarOnlyElement(1.0)], edge_states=np.zeros(4))

    with pytest.raises(ValueError, match="instead of"):
        result.evaluate(0, [0.0, 0.5])
