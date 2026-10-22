# Math Coding Guidelines for AI Agents

This document outlines the code style and architectural patterns for math project development.

## 1. Core Stack
* **Language:** Python 3.13
* **Libraries:** `numpy` (arrays), `scipy` (solvers). Versions are pinned in `pyproject.toml`.
  Add new dependencies only with the user's approval.

## 2. Code Style
* **Readable and less verbose code:** It must represent the idea, the details should be encapsulated by the library
  functions. Example: use `np.outer` instead of plain `*` multiplication that requires manual array reshaping.
* **Universality vs. simplicity:** Design the code that allows solving different problems by passing different
  arguments. Example: `scipy.integrate.odeint` gets the right-hand side of the equation as a callable `f` and
  accepts additional `args` that are then passed into `f` internally, so that these parameters do not have to
  be hardcoded. However, avoid speculative generality and let the user decide if adding complexity is really needed.
* **Performance oriented:** Use matrix operations instead of iterating through items.

## 3. Type Hinting & Precision
* **Scalars:** Use Python's built-in `float` for scalar type hints, not `np.float64`.
* **Arrays:** Allow flexible sequenc[README.md](README.md)e inputs `Sequence[float] | np.ndarray`. Normalize them internally using
  `np.asarray(points, dtype=np.float64)` with `dtype` specified to avoid accidental integers.
* **Scalar OR Array:** Use `np.typing.ArrayLike` when the function expects either a single value or an array of them.
  Use `np.isscalar(val)` for scalar check. Use `scalar = float(arr.item())` to get the scalar value.

## 4. Documentation
* **Format:** Strictly use **NumPy-style** docstrings. Do not duplicate types in the docstring (type hints are the
  single source of truth).
* **Public functions:** Require comprehensive docstrings detailing mathematical constraints and expected array shapes
  (e.g., `(N, 3)`). If the function is simple and its arguments are self-explanatory, the `Parameters` and `Returns`
  sections can be omitted.
* **Private functions:** The docstring can be absent.
* **Classes:** For simple classes, document only `Parameters` and explain derived values in `Notes`.
  Add `Attributes` when the stored public state is an important part of the class API.
* **Dataclasses:** Fields are both constructor parameters and public state, so document them in `Attributes`.

## 5. Verification
* **Formatting:** Run `ruff format .` before finishing.
* **Linting:** Run `ruff check --fix .` to automatically fix trivial rule breaks.
* **Tests:** Run `python -m pytest`. Compare arrays with `np.testing.assert_allclose`.

## 6. Standard Code Template Example

```python
from collections.abc import Sequence

import numpy as np


class SomeClass:
    """Class docstring."""

    def __init__(self, points: Sequence[float] | np.ndarray):
        """Magic methods first."""
        self.points = np.asarray(points, dtype=np.float64)

    def a_method(self):
        """Then normal methods, in order of importance."""

    def _private_method(self):
        pass


def process_coordinates(x: np.typing.ArrayLike) -> float | np.ndarray:
    """Do something.

    A description with more details.

    Parameters
    ----------
    x
        Coordinate (coordinates).
    """
    x_arr = np.asarray(x, dtype=np.float64)

    if np.isscalar(x):
        return float(x_arr.item())
    return x_arr
```