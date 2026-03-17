<<<<<<< HEAD
#  Evolutionary Algorithm Variants
### SVKM's NMIMS | Mukesh Patel School of Technology Management & Engineering
**Course:** Evolutionary Computing | **Semester:** VI | **Department:** AI

---

## Overview
Implementation of two Evolutionary Algorithm variants:
- **Genetic Programming (GP)** — Symbolic regression to find expression for `f(x) = x² + x + 1`
- **Differential Evolution (DE)** — Minimization of `f(x) = x₁ + x₂ + x₃`

---

## Files

| File | Description |
|---|---|
| `genetic_programming.py` | GP implementation with tree-based representation |
| `differential_evolution.py` | DE/rand/1/bin implementation |
| `performance_analysis.py` | Comparison of GP vs DE |
| `Lab6_EvolutionaryAlgorithmVariants.ipynb` | Interactive Jupyter notebook with user inputs |

---

## How to Run

### Option 1 — Jupyter Notebook (Recommended)
```bash
jupyter notebook Lab6_EvolutionaryAlgorithmVariants.ipynb
```
Run cells top to bottom. Press **Enter** at each prompt to use default parameters.

### Option 2 — Individual Python Scripts
```bash
# Run Genetic Programming
python genetic_programming.py

# Run Differential Evolution
python differential_evolution.py

# Run Performance Analysis (run above two first)
python performance_analysis.py
```

---

## Parameters

### Genetic Programming
| Parameter | Default | Description |
|---|---|---|
| Population Size | 200 | Number of trees per generation |
| Max Generations | 100 | Training iterations |
| Max Tree Depth | 4 | Max depth of expression tree |
| Crossover Rate | 0.8 | Probability of subtree crossover |
| Mutation Rate | 0.2 | Probability of subtree mutation |
| Tournament Size | 7 | Selection pressure |

### Differential Evolution
| Parameter | Default | Description |
|---|---|---|
| N (Population) | 30 | Number of vectors |
| F (Mutation Factor) | 0.8 | Differential weight `[0, 2]` |
| CR (Crossover Rate) | 0.9 | Binomial crossover probability |
| Max Generations | 200 | Termination criterion |

---

## Results

### GP — Regression Problem
- **Target:** `f(x) = x² + x + 1`, `x ∈ [-1, 1]`
- **Function set:** `{+, −, *, /}`
- **Terminal set:** `{x, R}` where `R ∈ [-5, 5]`

### DE — Minimization Problem
- **Objective:** Minimize `f(x) = x₁ + x₂ + x₃`
- **Domain:** `x₁, x₂, x₃ ∈ (0, 1)`
- **Global minimum:** `f(0, 0, 0) = 0`

---

## Key Observations
- DE converges **faster** (≈16 generations) vs GP (≈35 generations)
- GP produces a **human-readable expression** — useful for interpretability
- DE produces **optimal numbers** — useful for continuous optimization
- Both reach similar final fitness quality

---

## Dependencies
```bash
pip install -r requirements.txt
```
No external datasets required — both problems use programmatically generated data.

---

## References
- Eiben & Smith, *Introduction to Evolutionary Computing*, 2nd Ed., Springer 2015
- Storn & Price, *Differential Evolution*, Journal of Global Optimization, 1997
- Koza, *Genetic Programming*, MIT Press, 1992
=======
# evolutionary-algorithms
Genetic Programming (GP) and Differential Evolution (DE) are both evolutionary algorithms inspired by natural selection, but they solve problems in different ways.  Genetic Programming (GP) focuses on evolving computer programs or mathematical expressions. It starts with a population of randomly generated programs.
>>>>>>> f6ac06ddf8d2f80dce6e6be72873881e0c917a9f
