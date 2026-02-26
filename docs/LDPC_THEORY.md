# LDPC Theory (for Notebook P4)

## 1) Core objects

An LDPC code is defined by a sparse parity-check matrix $H \in \{0,1\}^{M \times N}$.

- $N$: number of transmitted bits
- $M$: number of parity checks
- Row $m$ of $H$ indicates which bits participate in check $m$

A valid codeword $x \in \{0,1\}^N$ satisfies:

$$
Hx \equiv 0 \pmod 2
$$

Each row enforces an even-parity XOR constraint.

## 2) Factor graph representation

The LDPC model in the notebook includes:

- Bit variables: $x_0,\dots,x_{N-1}$
- Observation variables: $y_0,\dots,y_{N-1}$
- Parity factors $\psi_m(x_{N(m)})$
- Channel factors $\phi_n(x_n,y_n)$

The joint factors as:

$$
p(x,y) \propto \prod_{m=1}^{M} \psi_m(x_{N(m)})\prod_{n=0}^{N-1}\phi_n(x_n,y_n)
$$

## 3) Parity-check factors

For check $m$ over neighborhood $N(m)$:

$$
\psi_m(x_{N(m)}) =
\begin{cases}
1 & \text{if } \sum_{n\in N(m)}x_n \equiv 0 \; (\mathrm{mod}\;2)\\
0 & \text{otherwise}
\end{cases}
$$

So these factors are hard constraints: assignments violating parity get probability mass 0.

## 4) Channel model (BSC)

Binary Symmetric Channel with flip probability $f$:

$$
\phi_n(x_n,y_n)=p(y_n\mid x_n)=
\begin{cases}
1-f & y_n=x_n\\
f & y_n\neq x_n
\end{cases}
$$

This injects soft evidence into each bit from observations.

## 5) Why sparsity matters

LDPC stands for low-density parity-check:

- each row has few ones (small check degree $k$)
- each column has few ones (small variable degree $j$)

Sparsity gives efficient message passing while still providing strong error correction for large block lengths.

## 6) Validation ideas in P4

- Structural: expected numbers of variables/factors and degrees
- Semantic: for random assignments $x$, parity-factor product agrees with syndrome test $Hx \mod 2$
- Visualization: inspect wiring consistency between $H$ and factor neighborhoods

## 7) Treewidth intuition

Exact inference complexity grows exponentially with treewidth. As $N$ grows, LDPC graphs become highly loopy and larger, so exact methods (junction tree) quickly become infeasible.

This motivates approximate methods used in Notebook P5.
