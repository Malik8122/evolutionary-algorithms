"""
Lab 6 - Task iii: Performance Analysis
Compares Genetic Programming vs Differential Evolution
"""

import random
import json
import time
import math
from genetic_programming import run_gp
from differential_evolution import run_de


def analyze_performance():
    print("\n" + "=" * 65)
    print("  PERFORMANCE ANALYSIS: Genetic Programming vs Differential Evolution")
    print("=" * 65)

    # ── Run GP ──────────────────────────────────────────────────────────────
    print("\n📌 Running Genetic Programming (Regression: x^2 + x + 1)...")
    random.seed(42)
    start = time.time()
    gp_history, gp_best_mse, gp_best_expr = run_gp(verbose=False)
    gp_time = time.time() - start

    # ── Run DE ──────────────────────────────────────────────────────────────
    print("📌 Running Differential Evolution (Minimize: f(x) = x1+x2+x3)...")
    random.seed(42)
    start = time.time()
    de_history, de_best_val, de_best_sol = run_de(verbose=False)
    de_time = time.time() - start

    # ── Convergence Analysis ─────────────────────────────────────────────────
    gp_convergence_gen = next(
        (h["generation"] for h in gp_history if h["best_fitness"] < 0.1), None
    )
    de_convergence_gen = next(
        (h["generation"] for h in de_history if h["best_fitness"] < 0.01), None
    )

    # ── Print Comparison Table ────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"{'Metric':<35} {'GP':>12} {'DE':>12}")
    print(f"{'─'*65}")
    print(f"{'Problem Type':<35} {'Regression':>12} {'Minimization':>12}")
    print(f"{'Representation':<35} {'Tree (AST)':>12} {'Real Vector':>12}")
    print(f"{'Population Size':<35} {'100':>12} {'30':>12}")
    print(f"{'Generations Run':<35} {len(gp_history):>12} {len(de_history):>12}")
    print(f"{'Best Fitness (MSE / f(x))':<35} {gp_best_mse:>12.6f} {de_best_val:>12.8f}")
    print(f"{'Convergence Generation (<threshold)':<35} {str(gp_convergence_gen or 'N/A'):>12} {str(de_convergence_gen or 'N/A'):>12}")
    print(f"{'Runtime (seconds)':<35} {gp_time:>12.3f} {de_time:>12.3f}")
    print(f"{'─'*65}")

    print(f"\n🌳 GP Best Expression Found: {gp_best_expr}")
    print(f"🎯 DE Best Solution Found:   x = {[round(v, 6) for v in de_best_sol]}, f(x) = {de_best_val:.8f}")

    # ── Observations ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  OBSERVATIONS")
    print("=" * 65)
    print("""
1. REPRESENTATION:
   GP uses tree-based representation (expression trees), which is ideal
   for symbolic regression. DE uses real-valued vectors, suited for
   continuous numerical optimization.

2. CONVERGENCE SPEED:
   DE typically converges faster per generation due to simpler operators
   (mutation + crossover on fixed-length vectors). GP needs more
   generations as it must evolve both structure and constants.

3. SOLUTION QUALITY:
   GP can find exact symbolic solutions (expressions) which are
   interpretable. DE finds precise numerical optima efficiently.

4. PARAMETER SENSITIVITY:
   DE requires careful tuning of F (mutation factor) and CR (crossover rate).
   GP requires tuning of tree depth, crossover/mutation rates, and population.

5. SCALABILITY:
   DE scales well with dimensionality for fixed structure problems.
   GP complexity grows with the search space of possible programs.

6. USE CASES:
   - GP: Symbolic regression, program synthesis, formula discovery
   - DE: Continuous parameter optimization, neural network tuning,
         engineering design problems
""")

    # ── Save combined results ──────────────────────────────────────────────────
    combined = {
        "gp": {
            "history": gp_history,
            "best_mse": gp_best_mse,
            "best_expression": gp_best_expr,
            "runtime": round(gp_time, 4)
        },
        "de": {
            "history": de_history,
            "best_fitness": de_best_val,
            "best_solution": de_best_sol,
            "runtime": round(de_time, 4)
        }
    }
    with open("performance_analysis.json", "w") as f:
        json.dump(combined, f, indent=2)
    print("✅ Performance data saved to performance_analysis.json")
    return combined


if __name__ == "__main__":
    analyze_performance()