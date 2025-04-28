import random
import numpy as np
import matplotlib.pyplot as plt

# Define bounds
bounds = [-5, 5]

# Objective function (we want to minimize this)
def objective_function(x):
    return x[0]**2 + x[1]**2  # Simple paraboloid

# Generate a random antibody (solution)
def random_antibody(bounds):
    return np.random.uniform(bounds[0], bounds[1], 2)

# Mutate an antibody (small change)
def mutate(antibody, mutation_rate, bounds):
    mutated = [
        antibody[0] + random.gauss(0, mutation_rate),
        antibody[1] + random.gauss(0, mutation_rate)
    ]
    # Clip to bounds
    mutated[0] = max(bounds[0], min(mutated[0], bounds[1]))
    mutated[1] = max(bounds[0], min(mutated[1], bounds[1]))
    return mutated

# Main Clonal Selection Algorithm
def clonal_selection(objective_function, bounds, n_population=20, n_generations=20, n_clones=5, mutation_rate=0.1):
    population = [random_antibody(bounds) for _ in range(n_population)]
    history = []
    best_scores = []

    print("=== Clonal Selection Algorithm Theoretical Output ===\n")

    for gen in range(n_generations):
        # Calculate fitness for all antibodies
        fitness = [objective_function(p) for p in population]

        # Sort population based on fitness
        sorted_population = [p for _, p in sorted(zip(fitness, population), key=lambda pair: pair[0])]
        sorted_fitness = sorted(fitness)

        best = sorted_population[0]
        best_fitness = sorted_fitness[0]
        avg_fitness = sum(fitness) / len(fitness)

        # Print theoretical output
        print(f"Generation {gen+1:02d}:")
        print(f" Best Antibody: {best}")
        print(f" Best Fitness : {best_fitness:.4f}")
        print(f" Avg Fitness  : {avg_fitness:.4f}\n")

        history.append(sorted_population)
        best_scores.append(best_fitness)

        # Clone and mutate
        new_population = []
        for i in range(n_population):
            for _ in range(n_clones):
                clone = mutate(sorted_population[i], mutation_rate, bounds)
                new_population.append(clone)

        # Select best new population
        new_fitness = [objective_function(p) for p in new_population]
        sorted_new_population = [p for _, p in sorted(zip(new_fitness, new_population), key=lambda pair: pair[0])]
        population = sorted_new_population[:n_population]

    return history, best_scores

# Run the algorithm
history, best_scores = clonal_selection(objective_function, bounds, n_population=15, n_generations=20, n_clones=5, mutation_rate=0.2)

# Plot convergence graph
def plot_convergence(best_scores):
    plt.figure(figsize=(8, 5))
    plt.plot(best_scores, marker='o', linestyle='-', color='blue', label='Best Fitness')
    plt.title("Clonal Selection Algorithm - Convergence", fontsize=14)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Best Fitness", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_convergence(best_scores)
