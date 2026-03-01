# Graphical Models — LDPC Labs (P4 & P5)

## Owners

- Arnau Rodon Comas (U212914)
- Jesús Santos Esteban (u186668)

This repository contains two Jupyter notebooks for understanding LDPC codes from a probabilistic graphical models perspective:

- `P4_ LDPC representation (STUDENT VERSION).ipynb`
- `P5_LDPC_decoding_(STUDENT_VERSION).ipynb`

## What you will learn

- How to represent an LDPC code as a factor graph
- How parity-check and channel factors are constructed
- Why exact inference becomes intractable as graph complexity grows
- How loopy belief propagation can still provide useful approximate decoding

## Quick start

1. Create/activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch Jupyter:

```bash
jupyter notebook
```

4. Run notebook 1 first, then notebook 2:
   - `P4_ LDPC representation (STUDENT VERSION).ipynb`
   - `P5_LDPC_decoding_(STUDENT_VERSION).ipynb`

## Notebook roadmap

### Notebook 1 (P4): LDPC representation

- Generates regular LDPC parity-check matrices
- Builds parity-check and BSC channel factors
- Constructs an LDPC factor graph in `pgmpy`
- Adds validation, visualization, and treewidth analysis

### Notebook 2 (P5): LDPC decoding

- Implements loopy belief propagation on loopy graphs
- Compares approximate marginals against exact inference on small instances
- Decodes LDPC codewords under different channel noise levels
- Applies loopy BP to a larger, intractable instance

## Theory notes

See:

- `docs/LDPC_THEORY.md`
- `docs/LOOPY_BP_INTUITION.md`

These notes summarize the math and practical intuition behind both notebooks.

## Suggested workflow

- Start with small `N` and inspect graph structure carefully.
- Validate each factor before running full BP loops.
- Track convergence diagnostics (max message difference).
- Evaluate decoding quality with BER vs noise level.

## Requirements

Dependencies are listed in `requirements.txt` and extracted from notebook imports.
