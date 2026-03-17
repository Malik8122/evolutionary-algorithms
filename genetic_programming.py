"""
Lab 6 - Task i: Genetic Programming for Regression
Problem: Find expression approximating f(x) = x^2 + x + 1 where -1 <= x <= 1
Function set: {*, /, -, +}
Terminal set: {x, R} where R is a random constant in [-5, 5]
"""

import random
import math
import json
import sys
from copy import deepcopy

# ─── Configuration ───────────────────────────────────────────────────────────
POPULATION_SIZE = 200
MAX_GENERATIONS = 100
MAX_DEPTH = 5
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 7
DATASET_SIZE = 20

FUNCTIONS = ['+', '-', '*', '/']
TERMINALS = ['x']

# ─── Dataset: x^2 + x + 1, x in [-1, 1] ─────────────────────────────────────
def generate_dataset(n=DATASET_SIZE):
    dataset = []
    for i in range(n):
        x = -1.0 + 2.0 * i / (n - 1)
        y = x**2 + x + 1
        dataset.append((x, y))
    return dataset

DATASET = generate_dataset()

# ─── Tree Representation ─────────────────────────────────────────────────────
class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value   # function or terminal
        self.left = left
        self.right = right

    def is_function(self):
        return self.value in FUNCTIONS

    def is_terminal(self):
        return not self.is_function()

    def to_string(self):
        if self.is_terminal():
            if isinstance(self.value, float):
                return f"{self.value:.3f}"
            return str(self.value)
        return f"({self.left.to_string()} {self.value} {self.right.to_string()})"

    def evaluate(self, x_val):
        if self.value == 'x':
            return x_val
        if isinstance(self.value, float):
            return self.value
        l = self.left.evaluate(x_val)
        r = self.right.evaluate(x_val)
        if self.value == '+': return l + r
        if self.value == '-': return l - r
        if self.value == '*': return l * r
        if self.value == '/': return l / r if abs(r) > 1e-6 else 1.0
        return 0.0


def random_terminal():
    if random.random() < 0.5:
        return Node('x')
    else:
        return Node(round(random.uniform(-5, 5), 3))


def random_tree(max_depth):
    if max_depth == 0:
        return random_terminal()
    if random.random() < 0.5:
        return random_terminal()
    func = random.choice(FUNCTIONS)
    return Node(func, random_tree(max_depth - 1), random_tree(max_depth - 1))


# ─── Fitness ──────────────────────────────────────────────────────────────────
def fitness(tree):
    total_error = 0
    for x_val, y_true in DATASET:
        try:
            y_pred = tree.evaluate(x_val)
            if math.isnan(y_pred) or math.isinf(y_pred):
                return float('inf')
            total_error += (y_pred - y_true) ** 2
        except:
            return float('inf')
    return total_error / len(DATASET)  # MSE (lower = better)


# ─── Genetic Operators ────────────────────────────────────────────────────────
def collect_nodes(tree):
    nodes = [tree]
    if tree.is_function():
        nodes += collect_nodes(tree.left)
        nodes += collect_nodes(tree.right)
    return nodes


def crossover(t1, t2):
    t1c, t2c = deepcopy(t1), deepcopy(t2)
    nodes1 = collect_nodes(t1c)
    nodes2 = collect_nodes(t2c)
    n1 = random.choice(nodes1)
    n2 = deepcopy(random.choice(nodes2))
    n1.value = n2.value
    n1.left = n2.left
    n1.right = n2.right
    return t1c


def mutate(tree):
    tc = deepcopy(tree)
    nodes = collect_nodes(tc)
    node = random.choice(nodes)
    sub = random_tree(2)
    node.value, node.left, node.right = sub.value, sub.left, sub.right
    return tc


def tournament_select(population, fitnesses):
    candidates = random.sample(range(len(population)), TOURNAMENT_SIZE)
    best = min(candidates, key=lambda i: fitnesses[i])
    return population[best]


# ─── Main GP Loop ─────────────────────────────────────────────────────────────
def run_gp(verbose=True):
    population = [random_tree(MAX_DEPTH) for _ in range(POPULATION_SIZE)]
    history = []

    for gen in range(MAX_GENERATIONS):
        fitnesses = [fitness(ind) for ind in population]
        best_fit = min(fitnesses)
        avg_fit = sum(f for f in fitnesses if f != float('inf')) / POPULATION_SIZE
        best_tree = population[fitnesses.index(best_fit)]

        history.append({
            "generation": gen + 1,
            "best_fitness": round(best_fit, 6),
            "avg_fitness": round(avg_fit, 4),
            "best_expr": best_tree.to_string()
        })

        if verbose:
            print(f"Gen {gen+1:3d} | Best MSE: {best_fit:.6f} | Best Expr: {best_tree.to_string()}")

        if best_fit < 1e-4:
            if verbose:
                print(f"\n✅ Converged at generation {gen+1}!")
            break

        new_pop = [best_tree]  # elitism
        while len(new_pop) < POPULATION_SIZE:
            p1 = tournament_select(population, fitnesses)
            if random.random() < CROSSOVER_RATE:
                p2 = tournament_select(population, fitnesses)
                child = crossover(p1, p2)
            else:
                child = deepcopy(p1)
            if random.random() < MUTATION_RATE:
                child = mutate(child)
            new_pop.append(child)
        population = new_pop

    best_fit = min(fitness(ind) for ind in population)
    best_tree = min(population, key=fitness)

    if verbose:
        print(f"\n🏆 Final Best MSE: {best_fit:.6f}")
        print(f"🌳 Best Expression: {best_tree.to_string()}")

        # Validate on dataset
        print("\n📊 Sample Predictions:")
        print(f"{'x':>8} {'True y':>10} {'Predicted':>12} {'Error':>10}")
        print("-" * 45)
        for x_val, y_true in DATASET[::4]:
            y_pred = best_tree.evaluate(x_val)
            print(f"{x_val:8.3f} {y_true:10.4f} {y_pred:12.4f} {abs(y_pred - y_true):10.6f}")

    return history, best_fit, best_tree.to_string()


if __name__ == "__main__":
    random.seed(42)
    history, best_mse, best_expr = run_gp(verbose=True)

    # Save results
    result = {
        "algorithm": "Genetic Programming",
        "problem": "Regression: x^2 + x + 1",
        "history": history,
        "best_mse": best_mse,
        "best_expression": best_expr
    }
    with open("gp_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n✅ Results saved to gp_results.json")