# Krylov Functions

A method proposed for construction of the transfer equation for the TMM.

## Definition

For a linear differential equation with initial conditions:

$$k_p u^{(p)}(t) + k_{p-1} u^{(p-1)}(t) + ... + k_1 u'(t) + k_0 u(t) = P(t)$$

$$u^{(i)}(0) = u_{i0} \quad (1 \leq i \leq p)$$

**The Krylov function $K_j(t)$** is a linear combination of **characteristic modes $\{F_i(t)\}$**, such that its $k$-th derivative has this property:

$$K_j^{(k)}(0) = \begin{cases} 1 & \text{if } k = j \\ 0 & \text{if } k \neq j \end{cases}$$

## Homogenous Case ($P(t) = 0$)

Thanks to the Krylov function property, the homogenous solution of the initial condition problem can be written simply as:

$$u_h(t) = \sum_{i = 1}^{i \leq p} {u_{i0} K_i(t)}$$

## Non-Homogenous Case ($P(t) \neq 0$)

The sum of the homogenous and the partial solutions $u_h(t) + u_p(t)$ doesn't satisfy the initial conditions as is. To fix this, let's use a "zero partial solution":

$$u_{zp}(t) = \sum_{i = 1}^{i \leq p} {C_i K_i(t)} + u_p(t)$$

Where ${C_i}$ constants must satisfy zero initial conditions:

$$u_{zp}^{(i)}(t) = 0$$

Given the Krylov function property, the "zero partial solution" can be calcuated as follows:

$$u_{zp}(t) = u_p(t) - \sum_{i = 1}^{i \leq p} {u_p^{(i)}(0) K_i(t)}$$

Then the complete solution is:

$$u_c(t) = u_h(t) + u_{zp}(t)$$

## TMM Application

Transfer matrix:

$$A(t) = \begin{pmatrix} K_1(t) \quad \cdots \quad K_p(t) \\ \vdots \quad \ddots \quad \vdots \\ K_1^{(p)}(t) \quad \cdots \quad K_p^{(p)}(t) \end{pmatrix}$$

Free term of the transfer equation:

$$\vec{B}(t) = (u_{zp}(t), ..., u_{zp}^{(p)}(t))^T$$

| Math                                                        | Code                                                    |
|-------------------------------------------------------------|---------------------------------------------------------|
| $A(t)$ and $\vec{B}(t)$                                     | `K` and `f` returned by `TMMElement.get_transfer_eq(s)` |
| Stacking the entries of $A(t)$ and $\vec{B}(t)$ into arrays | `stack_transfer_eq`                                     |
