import numpy as np
import matplotlib.pyplot as plt

# Define bounds
bounds = [-5, 5]

# Objective function (we want to minimize this)
def objective_function(x):
    return x[0]**2 + x[1]**2  # Simple paraboloid

# Generate a random antibody (solution)
def random_antibody(bounds):
    # Uniform distribution means that every number between -5 and 5 has an equal probability of being selected.
    # size = 2 means length of array is 2
    # from range bound[0] to bound[1]
    return np.random.uniform(bounds[0], bounds[1], 2)

# Mutate an antibody (small change)
def mutate(antibody, mutation_rate, bounds):
    #  add some random noice to antibody to mutate it , around 0 , standard deviation = mutation_rate and size = 2 length of array
    mutation = antibody + np.random.normal(0, mutation_rate, size=2)
    # clip does not allow values outside the bounds 
    #  if any value less then bound[0] then given value will be bound[0] same for bound[1]
    return np.clip(mutation, bounds[0], bounds[1])

# Main Clonal Selection Algorithm with theoretical outputs and convergence data
def clonal_selection(objective_function, bounds, n_population=20, n_generations=20, n_clones=5, mutation_rate=0.1):
    population = [random_antibody(bounds) for _ in range(n_population)]
    history = []
    best_scores = []



    print("=== Clonal Selection Algorithm Theoretical Output ===\n")

    for gen in range(n_generations):
        fitness = np.array([objective_function(p) for p in population])
        sorted_indices = np.argsort(fitness)
        population = [population[i] for i in sorted_indices]
        best = population[0]
        best_fitness = fitness[sorted_indices[0]]
        avg_fitness = np.mean(fitness)

        # Print theoretical output
        print(f"Generation {gen+1:02d}:")
        print(f" Best Antibody: {best}")
        print(f" Best Fitness : {best_fitness:.4f}")
        print(f" Avg Fitness  : {avg_fitness:.4f}\n")

        history.append(population)
        best_scores.append(best_fitness)

        # Clone and mutate
        new_population = []
        for i in range(n_population):
            for _ in range(n_clones):
                clone = mutate(population[i], mutation_rate, bounds)
                new_population.append(clone)

        new_fitness = np.array([objective_function(p) for p in new_population])
        best_indices = np.argsort(new_fitness)[:n_population]
        population = [new_population[i] for i in best_indices]

    return history, best_scores


# Run the algorithm and plot results
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
