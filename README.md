# tmm-kit — Transfer Matrix Method

This library is created for these courses at the Faculty of Applied Math of KPI:
1. **Modeling of Complex Systems**
2. **Geometric Modeling**

## 🧱 What You Can Build

### 1. `tmm_kit` — General TMM [1]

- **Branched systems of connected elements:**  electrical circuits, pipeline networks.
Only basic electrical presets are included as an example (`tmm_kit.electrical`) — any other domain is
implemented by defining your own elements, nodes and boundary conditions.

### 2. `tmm_kit.beam_splines` — TMM-based beam splines [2]

#### 2.1 Simple beam splines (in a functional form: `W = W(t)`)

- **Interpolation:** the curve passes exactly through every support point.
- **Approximation / smoothing:** supports act as springs.

#### 2.2 Corotational beam splines

- **Geometric extension**: interpolate or approximate the points of a 2D polyline.
- Supports **closed curves** and polyline refinement with **dummy points**.

## 🚀 Getting Started

Requires **Python 3.13+**.

### Normal Setup

If you want to use this library as-is in your project, just install it from GitHub:

```bash
python -m pip install git+https://github.com/Artem-Litkovskyi/tmm-kit.git
```

### Development Setup

If you want to download the source code and modify it, follow these steps:

<details>

<summary>Clone, install in editable mode, run tests</summary>

1. Clone the repository and navigate to its root directory.

   ```bash
   git clone https://github.com/Artem-Litkovskyi/tmm-kit.git
   cd tmm-kit
   ```

2. Create and activate a virtual environment.

   ```bash
   python -m venv .venv
   ```

   On Linux/macOS:

   ```bash
   source .venv/bin/activate
   ```

   On Windows:

   ```powershell
   .venv\Scripts\activate
   ```

3. Install the library and its dependencies. The argument `-e` makes an editable library installation.
   It allows you to modify the source code without reinstalling the library after each change.

   ```bash
   python -m pip install -e .
   ```

4. Install the development dependencies. It includes tools for testing and linting.
   The `--group` option requires pip 25.1+ (upgrade with `python -m pip install -U pip`).

   ```bash
   python -m pip install --group dev
   ```

5. Run the tests.

   ```bash
   python -m pytest
   ```

</details>

## 💡 Documentation

### Quickstart

An interpolating beam spline through three support points:

```python
import numpy as np

import tmm_kit.beam_splines as bs
from tmm_kit import TMMSolver

support_t = [2, 4, 5]
support_w = [-1, 4, 2]

elements, nodes = bs.create_beam_elements_and_nodes(
    support_t=support_t,
    element_class=bs.BeamElement,
    node_class=bs.BeamPinnedSupport,
    node_data={"displacement": support_w[1:-1]},
)
bcs = bs.pinned_ends(support_w[0], support_w[-1])

solver = TMMSolver(bs.BEAM_STATE_SIZE, elements, nodes, bcs)
result = solver.solve()

spline = bs.BeamSpline(result, support_t)
states = spline.evaluate_spline_state(t=np.linspace(support_t[0], support_t[-1], 128))
```

Swap `element_class`, `node_class` and the boundary condition preset (`bs.pinned_ends`)
to change what the spline does. There are presets for common use-cases:

<details>

<summary>Presets description</summary>

#### Beam Elements

- `BeamElement` — basic Euler-Bernoulli beam with $EI=1$.
- `BeamElementStretched` — reduces beam length by applying a force outwards.
- `BeamElementCompressed` — reduces beam curvature by applying a force inwards.
- `BeamElementOnElasticFoundation` — smoothing spline.

#### Beam Nodes

- `BeamPinnedSupport` — sets $W$; used for interpolation and hard constraints.
- `BeamElasticSupport` — sets $Q$ jump; used for approximation, smoothing and soft constraints.
- `BeamElasticSupportInverted` — same, via compliance = 1 / stiffness; used for very stiff supports.
- `BeamDummyNode` — used for CBS refinement by lowering angle misalignment `ψ`.

#### Boundary Conditions

Boundary condition presets follow a naming convention — `*_ends` for open curves,
`*_loop` for closed / periodic ones: `pinned_*`, `clamped_*`, `elastic_*`, `elastic_*_inverted`, plus `free_ends`.

Custom conditions are built with `TMMEquation`.

</details>

Public API is re-exported at the package level, so there's no need to import from individual modules:

- `tmm_kit` — core TMM: `TMMSolver`, `TMMResult`, `TMMEquation`, `TMMElement`, `TMMNode`, `ElementEdge`;
- `tmm_kit.beam_splines` — beam spline elements, nodes, boundary conditions, factories, evaluation and utils
  (conventionally imported as `bs`);
- `tmm_kit.electrical` — electrical presets.

All classes and functions have docstrings in the source code. For more usage examples see:
- `docs/beam_spline_examples.md` — beam splines, including CBS with dummy node refinement;
- `docs/electrical_example.md` — general TMM on a branched electrical circuit.

### Math Documentation

The underlying math is documented here. The second column maps each topic to the code that implements it.

| Document                                        | Implemented by                                                                       | Ref            |
|-------------------------------------------------|--------------------------------------------------------------------------------------|----------------|
| `docs/tmm/transfer_matrix_method.md`            | `TMMSolver`, `TMMEquation`                                                           | [1, lecture 1] |
| `docs/tmm/krylov_method.md`                     | background for deriving transfer matrices for elements                               | [1, lecture 2] |
| `docs/beam_splines/beam_spline.md`              | `BeamElement`, `BeamPinnedSupport`, `BeamSpline`                                     | [2]            |
| `docs/beam_splines/beam_spline_smoothing.md`    | `BeamElementOnElasticFoundation`, `BeamElasticSupport`, `BeamElasticSupportInverted` | [2]            |
| `docs/beam_splines/corotational_beam_spline.md` | `CorotationalBeamSpline`                                                             | [2]            |

## 📚 Literature

1. І.В. Ориняк. РОЗРАХУНКИ СКЛАДНИХ СИСТЕМ МЕТОДОМ ПОЧАТКОВИХ ПАРАМЕТРІВ. Рекомендовано Методичною радою КПІ ім. Ігоря Сікорського як навчальний посібник для студентів, які навчаються за спеціальністю 113 «Прикладна математика» Київ, КПІ ім. Ігоря Сікорського, 2022, 252 с. Режим доступу <https://ela.kpi.ua/bitstream/123456789/48744/1/Rozrakhunky_skladnykh_system>
2. І.В. Ориняк. ДИФЕРЕНЦІЙНА ГЕОМЕТРІЯ І СПЛАЙНИ [the link will be added]
