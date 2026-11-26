# Custom Elements and Nodes

Any domain can be modeled by subclassing `TMMElement` and `TMMNode` and building `TMMEquation`s
(see `docs/math/tmm/transfer_matrix_method.md` for the math and `src/tmm_kit/electrical` for a small example).

## Vectorized Classes

By default the solver and the evaluators call each custom element and node one by one.
This is simple but slow for systems with hundreds of elements.

A class can set the class variable `vectorized = True`. Then all objects of the class are processed
in a single call, made on one object whose fields are arrays with the values of all $N$ objects.
All library elements, as well as the beam spline nodes, are vectorized.

### Elements

`get_transfer_eq(s)` is called with the fields and `s` being arrays of shape `(N,)`.
It must use only element-wise operations on them and return `K` of shape `(N, S, S)` and `f` of shape `(N, S)`,
which is what `stack_transfer_eq` does:

```python
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from tmm_kit import TMMElement, stack_transfer_eq


@dataclass(slots=True)
class Resistor(TMMElement):
    vectorized: ClassVar[bool] = True

    resistance: float

    def get_transfer_eq(self, s: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        s_arr = np.asarray(s, dtype=np.float64)
        one, zero = np.ones_like(s_arr), np.zeros_like(s_arr)

        K = [
            [one, -self.resistance * one],
            [zero, one],
        ]
        f = [zero, zero]

        return stack_transfer_eq(K, f)
```

Avoid anything that works only with scalars: `if self.resistance > 0`, `math.sqrt`, `float(...)`.

### Nodes

`get_conjugation_eq()` is called with the fields stacked along the last axis, e.g., `self.inbound_elements[0]`
is an array of shape `(N,)`. The element indices, coefficients and right sides of the returned equations
can be such arrays: each equation is added once per node. All nodes of the class must return
the same number of equations, so nodes with a variable number of connected elements can't be vectorized.

### Inheritance

A subclass inherits `vectorized` from its parent. If you subclass a vectorized library class
and override its method with scalar-only code, set `vectorized = False` explicitly.
