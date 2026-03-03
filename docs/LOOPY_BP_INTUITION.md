# Loopy Belief Propagation (LBP): Intuition and Practical Use

## 1) From Tree BP to Loopy BP

On a tree-structured graphical model, belief propagation (BP) computes exact marginals in a finite number of passes. The reason is structural: once a message is sent across an edge, it never indirectly depends on itself. Information flows inward and outward without forming feedback loops, so dynamic programming applies.

On graphs with cycles, we use *exactly the same local update equations*, but we can no longer terminate after a single forward–backward sweep. Because cycles create feedback, messages become mutually dependent. The algorithm therefore becomes iterative: we repeatedly update

- variable-to-factor messages  
- factor-to-variable messages  

until convergence (or until a maximum number of iterations is reached). This iterative application of the standard BP equations on a cyclic graph is called **loopy belief propagation (LBP)**.

Mathematically, LBP is a fixed-point iteration for the BP message equations.

---

## 2) Why LBP Works Well for LDPC Codes

LDPC factor graphs contain cycles, but they are:

- sparse  
- locally tree-like over short neighborhoods  

For a limited number of iterations, messages propagate in regions that resemble trees. In early iterations, information has not yet traversed long cycles, so the approximation behaves similarly to exact tree BP.

As a result, LBP performs extremely well for LDPC decoding in practice, even though the graph is not acyclic. Its empirical success relies on sparsity and large girth (long shortest cycles), which delay harmful feedback effects.

---

## 3) Message-Passing Interpretation

Each message represents a local summary of uncertainty.

A variable-to-factor message aggregates all information about that variable coming from *other* neighboring factors:
\[
m_{i \to a}(x_i) = \prod_{b \in \mathcal{N}(i)\setminus a} m_{b \to i}(x_i).
\]

A factor-to-variable message combines:
- the compatibility function (constraint), and  
- incoming messages from all other connected variables:
\[
m_{a \to i}(x_i) =
\sum_{x_a \setminus x_i}
\psi_a(x_a)
\prod_{j \in \mathcal{N}(a)\setminus i}
m_{j \to a}(x_j).
\]

Messages are typically normalized at each update:
\[
m(x) \leftarrow \frac{m(x)}{\sum_x m(x)},
\]
which preserves numerical stability without affecting the fixed point.

---

## 4) Iterative Updates and Convergence

Because of cycles, the message equations form a coupled nonlinear system. LBP performs iterative updates over all edges.

Two common schedules:

- **Synchronous:** all messages at iteration \(t\) use values from iteration \(t-1\).
- **Asynchronous (sequential):** messages are updated one at a time and immediately reused.

Asynchronous updates often converge faster in practice because fresh information is propagated immediately.

A common stopping rule monitors message change:
\[
\max_{m} \|m^{(t)} - m^{(t-1)}\|_\infty < \text{tol}.
\]

If this condition is not satisfied, the algorithm stops after `max_iters`.

Convergence is not guaranteed in general, but when it occurs, messages satisfy the BP fixed-point equations.

---

## 5) Decoding via Marginals

Once messages have converged (or after the final iteration), approximate marginal beliefs are computed:

\[
p(x_i \mid y) \propto
\prod_{a \in \mathcal{N}(i)} m_{a \to i}(x_i).
\]

Hard decisions are obtained by maximum posterior decoding:
\[
\hat{x}_i = \arg\max_{b \in \{0,1\}} p(x_i=b \mid y).
\]

The bit error rate (BER) is then
\[
\mathrm{BER} = \frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[\hat{x}_i \neq x_i].
\]

---

## 6) Behavior as Noise Increases

Let \(f\) denote the flip probability of the channel.

For small \(f\), the local evidence is strong and consistent; LBP typically converges rapidly to low BER.

For moderate \(f\), conflicting evidence increases. Convergence may slow, and some residual decoding errors remain.

For large \(f\), observations become weak or misleading. The posterior becomes flatter and decoding performance degrades sharply.

Plotting BER versus \(f\) typically reveals a threshold-like transition, reflecting decoder robustness.

---

## 7) Why Not Exact Inference?

Exact inference on general graphs requires graph triangulation and junction tree construction. For LDPC-sized graphs, the induced treewidth becomes large, and computational and memory costs grow exponentially.

LBP avoids this explosion because each iteration scales linearly with:

- number of edges  
- local factor degrees  

Thus, LBP offers a critical tradeoff:

- approximate inference  
- scalable complexity  

This scalability is the fundamental reason belief propagation is the standard decoding algorithm for LDPC codes.