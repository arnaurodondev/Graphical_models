# Loopy Belief Propagation Intuition (for Notebook P5)

## 1) From tree BP to loopy BP

On trees, belief propagation converges in finite passes and gives exact marginals.
On graphs with cycles, the same local update equations are used iteratively:

- variable-to-factor messages
- factor-to-variable messages

This is loopy BP.

## 2) Why it can still work for LDPC

Even with cycles, LDPC graphs are sparse and locally tree-like for short neighborhoods.
That often makes loopy BP very effective in practice for decoding.

## 3) Message update view

Each message is a local summary of uncertainty.

- Variable $\to$ Factor: combines incoming information from other connected factors
- Factor $\to$ Variable: combines constraint compatibility with incoming neighbor messages

Messages are normalized each update for numerical stability.

## 4) Sequential updates and convergence

In practice, sequential/asynchronous updates often converge faster than fully synchronous updates.

Common stopping rule:

$$
\max_{m\in\text{all messages}} \|m^{(t)}-m^{(t-1)}\|_\infty < \text{tol}
$$

If not below tolerance, stop at `max_iters`.

## 5) Decoding with marginals

After convergence (or max iterations), compute bit marginals $p(x_i\mid y)$ and take hard decisions:

$$
\hat{x}_i = \arg\max_{b\in\{0,1\}} p(x_i=b\mid y)
$$

Then evaluate BER:

$$
\mathrm{BER} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\hat{x}_i \neq x_i]
$$

## 6) Expected behavior vs noise

- Small flip probability $f$: low BER, easy decoding
- Medium $f$: some errors remain, convergence can slow
- Large $f$: evidence is weak/noisy, BER rises sharply

Plotting BER against $f$ reveals decoder robustness.

## 7) Intractable large instances

Exact inference requires triangulation / junction tree construction and can explode in memory/time on large LDPC graphs.
Loopy BP stays tractable because each iteration scales roughly with graph size and local factor degree.

That tradeoff—approximate but scalable—is the key practical reason BP is standard in LDPC decoding.
