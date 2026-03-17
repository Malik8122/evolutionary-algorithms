"""
Lab 6 - Task ii: Differential Evolution for Minimization
Problem: Minimize f(x) = x1 + x2 + x3
         where x = [x1, x2, x3] and each xi in (0, 1)
"""

import random
import json
import math

# ─── DE Parameters (explicitly defined as per lab requirement) ───────────────
F = 0.8            # Differential weighting factor (mutation factor) [0, 2]
CR = 0.9           # Crossover Rate
N = 30             # Population size
D = 3              # Dimensionality (x1, x2, x3)
LOWER = 0.0        # Lower bound
UPPER = 1.0        # Upper bound
MAX_GENERATIONS = 200   # Termination criteria: max generations
TOLERANCE = 1e-6        # Termination criteria: fitness tolerance


# ─── Objective Function ───────────────────────────────────────────────────────
def objective(x):
    """f(x) = x1 + x2 + x3"""
    return sum(x)


# ─── Initialization ────────────────────────────────────────────────────────────
def initialize_population():
    """Uniformly sample population within bounds"""
    return [[random.uniform(LOWER, UPPER) for _ in range(D)] for _ in range(N)]


# ─── Mutation (DE/rand/1) ──────────────────────────────────────────────────────
def mutate(population, target_idx):
    """
    Select 3 distinct random vectors r1, r2, r3 (≠ target)
    Donor vector: v = x_r1 + F * (x_r2 - x_r3)
    """
    indices = list(range(N))
    indices.remove(target_idx)
    r1, r2, r3 = random.sample(indices, 3)

    donor = []
    for j in range(D):
        val = population[r1][j] + F * (population[r2][j] - population[r3][j])
        # Bound handling: clamp to [LOWER, UPPER]
        val = max(LOWER, min(UPPER, val))
        donor.append(val)
    return donor


# ─── Recombination (Binomial Crossover) ───────────────────────────────────────
def recombine(target, donor):
    """
    Trial vector: u_j = v_j if rand < CR or j == I_rand, else x_j
    I_rand ensures at least one dimension comes from donor
    """
    I_rand = random.randint(0, D - 1)
    trial = []
    for j in range(D):
        if random.random() <= CR or j == I_rand:
            trial.append(donor[j])
        else:
            trial.append(target[j])
    return trial


# ─── Selection ────────────────────────────────────────────────────────────────
def select(target, trial):
    """Greedy selection: keep better (lower) fitness"""
    return trial if objective(trial) < objective(target) else target


# ─── Main DE Loop ─────────────────────────────────────────────────────────────
def run_de(verbose=True):
    random.seed(42)
    population = initialize_population()
    history = []

    if verbose:
        print("=" * 60)
        print("  Differential Evolution — Minimization of f(x) = x1+x2+x3")
        print(f"  F={F}, CR={CR}, N={N}, D={D}, MaxGen={MAX_GENERATIONS}")
        print("=" * 60)

    best_solution = None
    best_value = float('inf')

    for gen in range(MAX_GENERATIONS):
        fitnesses = [objective(ind) for ind in population]
        gen_best_val = min(fitnesses)
        gen_best_sol = population[fitnesses.index(gen_best_val)]
        gen_avg = sum(fitnesses) / N

        if gen_best_val < best_value:
            best_value = gen_best_val
            best_solution = gen_best_sol[:]

        history.append({
            "generation": gen + 1,
            "best_fitness": round(gen_best_val, 8),
            "avg_fitness": round(gen_avg, 6),
            "best_x": [round(v, 6) for v in gen_best_sol]
        })

        if verbose and (gen % 20 == 0 or gen == 0):
            print(f"Gen {gen+1:3d} | Best f(x): {gen_best_val:.8f} | "
                  f"x = [{', '.join(f'{v:.4f}' for v in gen_best_sol)}]")

        # Termination: tolerance check
        if gen_best_val < TOLERANCE:
            if verbose:
                print(f"\n✅ Converged at generation {gen+1}! f(x) < {TOLERANCE}")
            break

        # Evolution step: apply mutation, crossover, selection for each vector
        new_population = []
        for i in range(N):
            donor = mutate(population, i)
            trial = recombine(population[i], donor)
            winner = select(population[i], trial)
            new_population.append(winner)
        population = new_population

    if verbose:
        print(f"\n{'='*60}")
        print(f"🏆 FINAL RESULT:")
        print(f"   Best f(x)  = {best_value:.10f}")
        print(f"   Solution   = {[round(v, 8) for v in best_solution]}")
        print(f"   Theoretical minimum: f(x) → 0 as x1,x2,x3 → 0")
        print(f"   (Global min = 0 at x=[0,0,0], boundary of domain)")
        print(f"{'='*60}")

    return history, best_value, best_solution


if __name__ == "__main__":
    history, best_val, best_sol = run_de(verbose=True)

    result = {
        "algorithm": "Differential Evolution",
        "problem": "Minimize f(x) = x1 + x2 + x3",
        "parameters": {"F": F, "CR": CR, "N": N, "D": D, "max_generations": MAX_GENERATIONS},
        "history": history,
        "best_fitness": best_val,
        "best_solution": best_sol
    }
    with open("de_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n✅ Results saved to de_results.json")