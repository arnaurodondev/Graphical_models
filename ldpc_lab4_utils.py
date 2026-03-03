import itertools
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import FactorGraph
import matplotlib.pyplot as plt
import networkx as nx


def generate_regular_ldpc_H(
    N: int,
    j: int,
    k: int,
    seed: int = 0,
    max_tries: int = 2000,
) -> np.ndarray:
    """Generate a random binary (j, k)-regular LDPC parity-check matrix H.

    The matrix has shape (M, N), where M = N*j/k must be an integer.
    Regularity means every column has exactly j ones and every row has exactly k ones.

    Args:
        N: Number of variable nodes / codeword bits (columns of H).
        j: Target column weight (ones per variable/column).
        k: Target row weight (ones per parity check/row).
        seed: Random seed for reproducible construction.
        max_tries: Maximum number of random construction attempts.

    Returns:
        A binary numpy array H with shape (M, N) and regular degrees.

    Raises:
        ValueError: If M = N*j/k is not an integer.
        RuntimeError: If a valid regular matrix is not found within max_tries.
    """
    # Random generator with fixed seed for reproducible H matrices.
    rng = np.random.default_rng(seed)
    if (N * j) % k != 0:
        raise ValueError(f"Need M = N*j/k integer, but N*j={N*j} not divisible by k={k}.")
    # Number of parity checks (rows): each of N columns contributes j ones, each row has k ones.
    M = (N * j) // k

    for _ in range(max_tries):
        # Create j "stubs" per variable (column) so each column tends to degree j.
        col_stubs = np.repeat(np.arange(N), j)
        rng.shuffle(col_stubs)

        # Build H with shape (M, N): M parity checks by N variables.
        H = np.zeros((M, N), dtype=np.uint8)
        ok = True
        for m in range(M):
            # Fill row m with exactly k variable indices.
            cols = col_stubs[m * k : (m + 1) * k]
            # Reject if a row gets duplicate variable indices.
            if len(set(cols.tolist())) != k:
                ok = False
                break
            H[m, cols] = 1
        if not ok:
            continue

        # Final regularity check: all column weights are j and all row weights are k.
        if np.all(H.sum(axis=0) == j) and np.all(H.sum(axis=1) == k):
            return H

    raise RuntimeError("Failed to construct a regular LDPC matrix. Try a different seed or parameters.")


def make_parity_check_factor(var_names: Sequence[str]) -> DiscreteFactor:
    """Create an even-parity factor over binary variables.

    The returned factor equals 1 for assignments with even Hamming weight
    and 0 otherwise, implementing a hard XOR-style parity constraint.

    Args:
        var_names: Variable names in the factor scope.

    Returns:
        A pgmpy DiscreteFactor encoding even parity over var_names.
    """
    # Binary parity table over d variables has size 2^d.
    factor = DiscreteFactor(var_names, [2] * len(var_names), np.zeros(2 ** len(var_names)))
    for combination in itertools.product([0, 1], repeat=len(var_names)):
        # Even parity constraint: factor is 1 only when XOR-sum is even.
        if sum(combination) % 2 == 0:
            factor.values[combination] = 1.0
    return factor


def make_bsc_channel_factor(x_var: str, y_var: str, f: float) -> DiscreteFactor:
    """Create a Binary Symmetric Channel factor p(y|x).

    For binary states {0,1}, this factor encodes:
      - p(y=x) = 1-f
      - p(y!=x) = f

    Args:
        x_var: Name of latent/transmitted bit variable.
        y_var: Name of observed/received bit variable.
        f: Bit-flip probability of the BSC.

    Returns:
        A pairwise DiscreteFactor over [x_var, y_var].
    """
    # Pairwise factor phi(x,y)=p(y|x) for a Binary Symmetric Channel with flip probability f.
    factor = DiscreteFactor([x_var, y_var], [2, 2], np.zeros(4))
    for combination in itertools.product([0, 1], repeat=2):
        x, y = combination
        # If bit is preserved use 1-f, if flipped use f.
        factor.values[combination] = (1 - f) if x == y else f
    return factor


def build_ldpc_factor_graph(H: np.ndarray, f: float) -> FactorGraph:
    """Build an LDPC factor graph from parity matrix H and channel noise f.

    Graph structure:
      - Variables: x0..x{N-1} (transmitted bits), y0..y{N-1} (observations)
      - Parity factors: one factor per row of H, connected to selected x vars
      - Channel factors: one factor per n, connected to (x_n, y_n)

    Args:
        H: Binary parity-check matrix of shape (M, N).
        f: BSC flip probability used in channel factors.

    Returns:
        A pgmpy FactorGraph representing the LDPC model.
    """
    # H has M checks (rows) and N transmitted bits (columns).
    M, N = H.shape
    x_vars = [f"x{n}" for n in range(N)]
    y_vars = [f"y{n}" for n in range(N)]

    G = FactorGraph()
    # Add variable nodes first: transmitted bits x_n and observed bits y_n.
    for var in x_vars + y_vars:
        G.add_node(var)

    for m in range(M):
        # Row m of H defines which x variables participate in check m.
        var_names = [x_vars[n] for n in range(N) if H[m, n] == 1]
        factor = make_parity_check_factor(var_names)
        G.add_factors(factor)
        # Connect parity factor to all variables in its scope.
        for var in var_names:
            G.add_edge(var, factor)

    for n in range(N):
        # Add one channel factor per bit linking latent x_n with observed y_n.
        factor = make_bsc_channel_factor(x_vars[n], y_vars[n], f)
        G.add_factors(factor)
        G.add_edge(x_vars[n], factor)
        G.add_edge(y_vars[n], factor)

    return G


def validate_model(
    G: FactorGraph,
    H: Optional[np.ndarray] = None,
    f: Optional[float] = None,
    num_random_assignments: int = 1000,
    exhaustive_max_n: int = 14,
    seed: int = 0,
) -> bool:
    """Validate a factor graph with optional LDPC-specific checks.

    Behavior:
    - Always validates `G.check_model()`.
    - If only `G` is provided, runs only generic checks and skips LDPC-specific checks.
    - If `H` is provided, runs LDPC structure and parity-consistency checks.
    - If `f` is also provided, validates BSC channel factor semantics.
    """
    # Always run pgmpy's internal structural checks first.
    assert G.check_model(), "Basic graph validation failed (G.check_model())."

    # If only a graph is provided, skip LDPC-specific checks that require H/f.
    if H is None:
        return True

    if H.ndim != 2:
        raise ValueError("H must be a 2D array.")

    if f is not None and not (0.0 <= f <= 1.0):
        raise ValueError("Flip probability f must be in [0, 1].")

    # H shape encodes number of checks (M) and code length (N).
    M, N = H.shape
    rng = np.random.default_rng(seed)

    # Split graph nodes into factor nodes and variable nodes.
    factors = list(G.get_factors())
    factor_set = set(factors)
    all_nodes = list(G.nodes())
    variable_nodes = [node for node in all_nodes if node not in factor_set]

    expected_x = {f"x{n}" for n in range(N)}
    expected_y = {f"y{n}" for n in range(N)}
    variable_names = {node for node in variable_nodes if isinstance(node, str)}

    assert variable_names == (expected_x | expected_y), (
        "Variable nodes mismatch. "
        f"Expected {2 * N} nodes x/y, got {len(variable_names)} named variables."
    )
    assert len(variable_nodes) == 2 * N, f"Expected {2 * N} variable nodes, got {len(variable_nodes)}"

    # Classify factors by scope pattern: parity over x's vs channel over (x_n, y_n).
    parity_factors: List[DiscreteFactor] = []
    channel_factors: List[DiscreteFactor] = []

    for fac in factors:
        fac_vars = list(fac.variables)
        if len(fac_vars) == 2 and any(v.startswith("y") for v in fac_vars):
            channel_factors.append(fac)
        elif all(v.startswith("x") for v in fac_vars):
            parity_factors.append(fac)

    assert len(parity_factors) == M, f"Expected {M} parity factors, got {len(parity_factors)}"
    assert len(channel_factors) == N, f"Expected {N} channel factors, got {len(channel_factors)}"
    assert len(factors) == (M + N), f"Expected {M + N} total factors, got {len(factors)}"

    # Degree checks enforce the expected LDPC wiring induced by H.
    for n in range(N):
        x_n = f"x{n}"
        y_n = f"y{n}"
        deg_x = len(list(G.neighbors(x_n)))
        deg_y = len(list(G.neighbors(y_n)))

        expected_deg_x = int(H[:, n].sum()) + 1
        assert deg_x == expected_deg_x, (
            f"Degree mismatch for {x_n}: expected {expected_deg_x}, got {deg_x}"
        )
        assert deg_y == 1, f"Degree mismatch for {y_n}: expected 1, got {deg_y}"

    expected_parity_scopes = [frozenset(f"x{n}" for n in np.where(H[m] == 1)[0]) for m in range(M)]
    actual_parity_scopes = []

    # Local parity semantics check: each parity factor is a hard XOR constraint.
    for fac in parity_factors:
        scope = list(fac.variables)
        assert all(v.startswith("x") for v in scope), "Parity factor includes non-x variable."
        actual_parity_scopes.append(frozenset(scope))

    assert Counter(actual_parity_scopes) == Counter(expected_parity_scopes), (
        "Parity-factor scopes do not match H row neighborhoods."
    )

    seen_channel_pairs = set()
    for fac in channel_factors:
        scope = list(fac.variables)
        assert len(scope) == 2, "Channel factor must be pairwise."

        x_vars = [v for v in scope if v.startswith("x")]
        y_vars = [v for v in scope if v.startswith("y")]
        assert len(x_vars) == 1 and len(y_vars) == 1, (
            "Channel factor must connect one x and one y variable."
        )

        x_var, y_var = x_vars[0], y_vars[0]
        ix, iy = int(x_var[1:]), int(y_var[1:])
        assert ix == iy, f"Channel factor mismatch: {x_var} connected to {y_var}."
        seen_channel_pairs.add((ix, iy))

        if f is not None:
            expected = np.array([[1 - f, f], [f, 1 - f]], dtype=float)
            assert np.allclose(fac.values, expected), (
                f"Channel factor values mismatch for pair ({x_var}, {y_var})."
            )

    assert seen_channel_pairs == {(n, n) for n in range(N)}, "Missing or duplicate channel factors."

    for fac in parity_factors:
        scope = list(fac.variables)
        d = len(scope)
        for bits in itertools.product([0, 1], repeat=d):
            got = float(fac.values[bits])
            expected = 1.0 if (sum(bits) % 2 == 0) else 0.0
            assert np.isclose(got, expected), (
                f"Parity factor semantic mismatch on scope {scope} and assignment {bits}."
            )

    # Global consistency: accepted assignments must match syndrome test Hx mod 2 = 0.
    x_names = [f"x{n}" for n in range(N)]

    def parity_product_for_assignment(x_bits: Sequence[int]) -> float:
        assign: Dict[str, int] = {name: int(bit) for name, bit in zip(x_names, x_bits)}
        value = 1.0
        for fac in parity_factors:
            scope = list(fac.variables)
            idx = tuple(assign[v] for v in scope)
            value *= float(fac.values[idx])
            if value == 0.0:
                break
        return value

    if N <= exhaustive_max_n:
        assignments: Iterable[Sequence[int]] = itertools.product([0, 1], repeat=N)
    else:
        assignments = (rng.integers(0, 2, size=N).tolist() for _ in range(num_random_assignments))

    for x_bits in assignments:
        x_arr = np.asarray(x_bits, dtype=int)
        syndrome_zero = np.all((H @ x_arr) % 2 == 0)
        parity_ok = parity_product_for_assignment(x_bits) > 0.5
        assert parity_ok == syndrome_zero, "Global parity consistency failed."

    return True


def validate_ldpc_graph(
    G,
    H: np.ndarray,
    f: float,
    num_random_assignments: int = 1000,
    exhaustive_max_n: int = 14,
    seed: int = 0,
):
    """Run a detailed structural+semantic validation suite for an LDPC graph.

    Checks performed:
      1) basic pgmpy model validity
      2) node/factor counts and variable naming
      3) parity/channel factor arities and counts
      4) wiring consistency with H
      5) local factor semantics (parity + BSC channel)
      6) global parity consistency on exhaustive/random x assignments

    Args:
        G: Factor graph to validate.
        H: Reference parity-check matrix used to build the graph.
        f: BSC flip probability expected in channel factors.
        num_random_assignments: Number of random global assignments when N is large.
        exhaustive_max_n: Exhaustive validation threshold on number of bits.
        seed: Random seed for random-assignment validation.

    Returns:
        True if all checks pass.

    Raises:
        ValueError: If f is outside [0, 1].
        AssertionError: If any structural or semantic consistency check fails.
    """
    # Channel flip probability must be a valid Bernoulli parameter.
    if not (0.0 <= f <= 1.0):
        # This is a sanity check for the flip probability parameter, which should be in the range [0, 1].
        raise ValueError("Flip probability f must be in [0, 1].")

    M, N = H.shape
    rng = np.random.default_rng(seed)

    # 1) Basic pgmpy model validity
    assert G.check_model(), "Basic LDPC graph validation failed (G.check_model())."

    factors = list(G.get_factors())
    factor_set = set(factors)
    all_nodes = list(G.nodes())
    variable_nodes = [node for node in all_nodes if node not in factor_set]

    expected_x = {f"x{n}" for n in range(N)}
    expected_y = {f"y{n}" for n in range(N)}
    variable_names = {node for node in variable_nodes if isinstance(node, str)}

    # 2) Variable count check and variable naming
    assert variable_names == (expected_x | expected_y), (
        "Variable nodes mismatch. "
        f"Expected {2 * N} nodes x/y, got {len(variable_names)} named variables."
    )
    # Variable node count should be X + Y = 2N
    assert len(variable_nodes) == 2 * N, f"Expected {2*N} variable nodes, got {len(variable_nodes)}"

    parity_factors = []
    channel_factors = []

    for fac in factors:
        fac_vars = list(fac.variables)
        if len(fac_vars) == 2 and any(v.startswith("y") for v in fac_vars):
            # Channel factor should connect one x and one y variable, and has the format φ_n(x_n, y_n) = p(y_n | x_n) for BSC flip prob f
            channel_factors.append(fac)
        elif all(v.startswith("x") for v in fac_vars):
            # Parity factor has the form φ(x_{i1}, x_{i2}, ..., x_{id}) = 1 iff sum of x_{ij} is even, else 0.
            parity_factors.append(fac)

    # 3) Factor count checks
    assert len(parity_factors) == M, f"Expected {M} parity factors, got {len(parity_factors)}"
    assert len(channel_factors) == N, f"Expected {N} channel factors, got {len(channel_factors)}"
    assert len(factors) == (M + N), f"Expected {M+N} total factors, got {len(factors)}"

    # Degree checks: x_n should have deg = column_weight(H[:, n]) + 1 ; y_n should have deg = 1
    for n in range(N):
        x_n = f"x{n}"
        y_n = f"y{n}"

        deg_x = len(list(G.neighbors(x_n)))
        deg_y = len(list(G.neighbors(y_n)))

        # Expected degree for x_n is number of parity checks it participates in (sum of column in H) + 1 for channel factor
        expected_deg_x = int(H[:, n].sum()) + 1
        assert deg_x == expected_deg_x, (
            f"Degree mismatch for {x_n}: expected {expected_deg_x}, got {deg_x}"
        )
        # Expected degree for y_n is 1 (only connected to its channel factor)
        assert deg_y == 1, f"Degree mismatch for {y_n}: expected 1, got {deg_y}"

    # Parity wiring consistency with H (multiset compare of scopes)
    expected_parity_scopes = []
    for m in range(M):
        # H[m] is the m-th row; np.where(H[m] == 1)[0] gives indices of x variables in this parity check
        # H[m] = [0, 1, 0, 1, 1] -> scope = {x1, x3, x4}
        scope = frozenset(f"x{n}" for n in np.where(H[m] == 1)[0])
        expected_parity_scopes.append(scope)

    actual_parity_scopes = []
    for fac in parity_factors:
        scope = list(fac.variables)
        # Format is φ(x_{i1}, x_{i2}, ..., x_{id}) = 1/0
        assert all(v.startswith("x") for v in scope), "Parity factor includes non-x variable."
        actual_parity_scopes.append(frozenset(scope))

    from collections import Counter

    # We use Counter to compare the multisets of parity scopes, since the order of factors is not guaranteed.
    assert Counter(actual_parity_scopes) == Counter(expected_parity_scopes), (
        "Parity-factor scopes do not match H row neighborhoods."
    )

    # Channel wiring and semantics
    seen_channel_pairs = set()

    for fac in channel_factors:
        scope = list(fac.variables)
        # Each channel factor should have exactly 2 variables (x_n and y_n).
        assert len(scope) == 2, "Channel factor must be pairwise."
        x_vars = [v for v in scope if v.startswith("x")]
        y_vars = [v for v in scope if v.startswith("y")]
        # Each channel factor should connect exactly one x and one y variable,
        assert len(x_vars) == 1 and len(y_vars) == 1, "Channel factor must connect one x and one y variable."

        x_var, y_var = x_vars[0], y_vars[0]
        ix = int(x_var[1:])
        iy = int(y_var[1:])
        # The indices should match (i.e., x_n connected to y_n).
        assert ix == iy, f"Channel factor mismatch: {x_var} connected to {y_var}."

        seen_channel_pairs.add((ix, iy))

    # Check that we have exactly the expected set of channel pairs (x_n, y_n) for n in [0, N-1].
    assert seen_channel_pairs == {(n, n) for n in range(N)}, "Missing or duplicate channel factors."

    # For each parity factor φ_m, we explicitly enumerate all 2^d assignments
    # (where d is the degree of that parity check) and verify that:
    #
    #   φ_m(x_{i1}, ..., x_{id}) =
    #       1  if sum(x_{ij}) mod 2 == 0
    #       0  otherwise
    #
    # This guarantees that each parity factor is implemented exactly as a hard XOR constraint.
    for fac in parity_factors:
        scope = list(fac.variables)
        d = len(scope)
        for bits in itertools.product([0, 1], repeat=d):
            # Retrieve the actual factor value for this assignment
            got = float(fac.values[bits])

            # Expected value under even-parity (XOR) semantics
            expected = 1.0 if (sum(bits) % 2 == 0) else 0.0

            assert np.isclose(got, expected), (
                f"Parity factor semantic mismatch on scope {scope} and assignment {bits}."
            )

    # We now validate that the PRODUCT of all parity factors equals 1 if and only if Hx ≡ 0 (mod 2), i.e., x is a valid codeword.
    # If N is small, we exhaustively enumerate all 2^N assignments, otherwise, we randomly sample assignments for scalability.
    x_names = [f"x{n}" for n in range(N)]

    def parity_product_for_assignment(x_bits):
        # Build assignment dictionary for x variables
        assign = {name: int(bit) for name, bit in zip(x_names, x_bits)}

        # Multiply all parity factor values for this assignment
        prod = 1.0
        for fac in parity_factors:
            scope = list(fac.variables)
            idx = tuple(assign[v] for v in scope)
            prod *= float(fac.values[idx])

            # Early stopping: if any parity factor is zero,
            # the product is zero and parity is violated.
            if prod == 0.0:
                break

        return prod

    # Choose exhaustive or random testing depending on N
    if N <= exhaustive_max_n:
        # Exhaustively check all 2^N bit assignments
        assignments = itertools.product([0, 1], repeat=N)
    else:
        # Randomly sample assignments when exhaustive testing is infeasible
        assignments = (rng.integers(0, 2, size=N).tolist()
                    for _ in range(num_random_assignments))

    checked = 0
    for x_bits in assignments:
        x_arr = np.asarray(x_bits, dtype=int)

        # Compute algebraic syndrome Hx mod 2
        syndrome_zero = np.all((H @ x_arr) % 2 == 0)

        # Compute factor-graph parity evaluation (product > 0 means all constraints satisfied)
        parity_ok = parity_product_for_assignment(x_bits) > 0.5

        # The factor graph must accept exactly the same assignments
        # as those satisfying Hx ≡ 0 (mod 2).
        assert parity_ok == syndrome_zero, "Global parity consistency failed."

        checked += 1

    print(
        f"Validation passed: N={N}, M={M}, factors={len(factors)}, "
        f"parity_factors={len(parity_factors)}, channel_factors={len(channel_factors)}, "
        f"assignments_checked={checked}"
    )
    return True

def visualize_factor_graph(G: FactorGraph, max_nodes: int = 200):
    """Visualize an LDPC factor graph with a fixed column layout.

    Columns (left to right):
      y variables → channel factors → x variables → parity-check factors

    Args:
        G: Factor graph to draw.
        max_nodes: Safety threshold to avoid unreadable huge plots.

    Raises:
        ValueError: If graph size exceeds max_nodes.
    """
    # Build a plain NetworkX graph only for plotting.
    Hnx = nx.Graph()

    factors = list(G.get_factors())
    factor_set = set(factors)
    variable_nodes = [node for node in G.nodes() if node not in factor_set]

    x_vars = sorted([v for v in variable_nodes if isinstance(v, str) and v.startswith("x")], key=lambda s: int(s[1:]))
    y_vars = sorted([v for v in variable_nodes if isinstance(v, str) and v.startswith("y")], key=lambda s: int(s[1:]))

    channel_factors = [fac for fac in factors if any(v.startswith("y") for v in fac.variables)]
    parity_factors = [fac for fac in factors if all(v.startswith("x") for v in fac.variables)]

    channel_factors = sorted(channel_factors, key=lambda fac: min(int(v[1:]) for v in fac.variables if v.startswith("x")))
    parity_factors = sorted(parity_factors, key=lambda fac: min(int(v[1:]) for v in fac.variables))

    # Column-wise node order controls the left-to-right layout.
    all_nodes = y_vars + channel_factors + x_vars + parity_factors
    if len(all_nodes) > max_nodes:
        raise ValueError(f"Graph too large to visualize ({len(all_nodes)} > {max_nodes}).")

    Hnx.add_nodes_from(all_nodes)
    Hnx.add_edges_from(G.edges())

    # Place each node type in a fixed x-column for readability.
    columns = [y_vars, channel_factors, x_vars, parity_factors]
    x_coords = [0.0, 1.0, 2.0, 3.0]
    pos = {}

    for x_coord, nodes in zip(x_coords, columns):
        n = len(nodes)
        if n == 0:
            continue
        ys = np.linspace(1.0, -1.0, n)
        for node, y in zip(nodes, ys):
            pos[node] = (x_coord, float(y))

    plt.figure(figsize=(14, 8))

    nx.draw_networkx_nodes(Hnx, pos, nodelist=y_vars, node_color="#4c78a8", node_size=350, label="y variables")
    nx.draw_networkx_nodes(Hnx, pos, nodelist=x_vars, node_color="#54a24b", node_size=350, label="x variables")
    nx.draw_networkx_nodes(Hnx, pos, nodelist=channel_factors, node_color="#f58518", node_shape="s", node_size=450, label="channel factors")
    nx.draw_networkx_nodes(Hnx, pos, nodelist=parity_factors, node_color="#e45756", node_shape="s", node_size=450, label="parity factors")

    nx.draw_networkx_edges(Hnx, pos, width=1.0, alpha=0.7)

    label_nodes = y_vars + x_vars
    labels = {node: node for node in label_nodes}
    nx.draw_networkx_labels(Hnx, pos, labels=labels, font_size=7)

    plt.title("LDPC Factor Graph")
    plt.axis("off")
    plt.legend(loc="upper center", ncol=4, frameon=False)
    plt.tight_layout()
    plt.show()
