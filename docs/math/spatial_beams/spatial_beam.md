# Spatial Beam

## Overview

**Spatial beam** is a straight beam element in 3D that resists compression, torsion and bending in two planes.
Elements are joined at rigid nodes, which may branch and change direction, so frames can be modeled.

## Local Basis

Each element has a right-handed orthonormal basis given in global coordinates:
*   **$\vec{t}$**: Tangent vector (the element axis)
*   **$\vec{n}$**: Normal vector
*   **$\vec{\beta}$**: Binormal vector

A basis can be obtained from another one by rotating it about one of its own axes (`rotate_basis`).
For example, rotation by an angle $\psi$ about $\vec{\beta}$:

$$\begin{pmatrix} \vec{t}' \\ \vec{n}' \\ \vec{\beta}' \end{pmatrix} = \begin{pmatrix} \cos\psi & \sin\psi & 0 \\ -\sin\psi & \cos\psi & 0 \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} \vec{t} \\ \vec{n} \\ \vec{\beta} \end{pmatrix}$$

## Model

The state consists of four vectors in the local basis, 12 variables in total:

| Vector       | $\vec{t}$ | $\vec{n}$  | $\vec{\beta}$    | Constants                       |
|--------------|-----------|------------|------------------|---------------------------------|
| Displacement | $u$       | $w_n$      | $w_\beta$        | `SPATIAL_BEAM_DISPLACEMENT_T/N/B` |
| Rotation     | $\phi$    | $\theta_n$ | $\theta_\beta$   | `SPATIAL_BEAM_ROTATION_T/N/B`     |
| Moment       | $M_{rot}$ | $M_n$      | $M_\beta$        | `SPATIAL_BEAM_MOMENT_T/N/B`       |
| Force        | $N$       | $Q_n$      | $Q_\beta$        | `SPATIAL_BEAM_FORCE_T/N/B`        |

The variables are stored in this order (row by row).

Compression:

$$\frac{du}{dx} = -\frac{N}{SE}, \quad \frac{dN}{dx} = f_t$$

Torsion:

$$\frac{d\phi}{dx} = -\frac{M_{rot}}{GJ_p}, \quad \frac{dM_{rot}}{dx} = f_{rot}$$

Bending in the $(\vec{t}, \vec{n})$ plane:

$$\frac{dw_n}{dx} = \theta_\beta, \quad \frac{d\theta_\beta}{dx} = -\frac{M_\beta}{EI_\beta}, \quad \frac{dM_\beta}{dx} = -Q_n, \quad \frac{dQ_n}{dx} = q_n$$

Bending in the $(\vec{t}, \vec{\beta})$ plane:

$$\frac{dw_\beta}{dx} = -\theta_n, \quad \frac{d\theta_n}{dx} = -\frac{M_n}{EI_n}, \quad \frac{dM_n}{dx} = Q_\beta, \quad \frac{dQ_\beta}{dx} = q_\beta$$

The distributed loads $f_t$, $f_{rot}$, $q_n$, $q_\beta$ are constant along the element.

The signs differ between the two bending planes because the rotation, the moment and the force are true vectors.
In the vector form the equations are the same for all components:

$$\frac{d\vec{F}}{dx} = \vec{q}, \quad \frac{d\vec{M}}{dx} = -\vec{t} \times \vec{F}$$

Thus $EI \cdot w'''' = q$ holds in both planes, and all four vectors are transformed between bases in the same way.

Any stiffness can be infinite (e.g., $\frac{1}{SE} = 0$ for an incompressible beam).

## Transfer Equation

The transfer matrix is built by the Krylov method (see `docs/math/tmm/krylov_method.md`). For the bending in the $(\vec{t}, \vec{n})$ plane:

$$\begin{pmatrix} w_n \\ \theta_\beta \\ M_\beta \\ Q_n \end{pmatrix} = \begin{pmatrix} 1 & s & -\frac{s^2}{2EI_\beta} & \frac{s^3}{6EI_\beta} \\ 0 & 1 & -\frac{s}{EI_\beta} & \frac{s^2}{2EI_\beta} \\ 0 & 0 & 1 & -s \\ 0 & 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} w_n \\ \theta_\beta \\ M_\beta \\ Q_n \end{pmatrix}_0 + q_n \begin{pmatrix} \frac{s^4}{24EI_\beta} \\ \frac{s^3}{6EI_\beta} \\ -\frac{s^2}{2} \\ s \end{pmatrix}$$

The other blocks are constructed the same way.

## Conjugation Equations

A rigid node connects the ends of $k_{in}$ inbound elements and the starts of $k_{out}$ outbound elements.
Let $R_i$ be the basis matrix of an $i$-th element (its rows are $\vec{t}$, $\vec{n}$, $\vec{\beta}$).
Then a local vector $\vec{v}$ has global coordinates $R_i^T \vec{v}$.

The displacement and rotation vectors $\vec{d}$ of all the connected edges are equal in global coordinates:

$$R_i^T \vec{d}^i - R_1^T \vec{d}^1 = 0 \quad (2 \leq i \leq k_{in} + k_{out})$$

The force and moment vectors $\vec{F}$ are in equilibrium:

$$\sum_{outbound}{R_i^T \vec{F}^i_0} - \sum_{inbound}{R_i^T \vec{F}^i_e} = 0$$

That gives $6(k_{in} + k_{out})$ equations per node.

## Boundary Conditions

Six equations per element edge:

| Support   | Function      | Equations                                                                     |
|-----------|---------------|-------------------------------------------------------------------------------|
| Pinned    | `pinned_end`  | Zero displacements and moments                                                |
| Clamped   | `clamped_end` | Zero displacements and rotations                                              |
| Free      | `free_end`    | Zero forces and moments                                                       |
| Sliding   | `sliding_end` | Zero displacement along one local axis, zero other two forces and all moments |

For a code example see `docs/usage/spatial_beam_examples.md`.
