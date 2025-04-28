import random

# --- Problem Constants ---
IND_SIZE = 10  # Number of genes (dimensions)
LOW, UP = -5.12, 5.12  # Range of gene values

# --- Fitness Function: Sphere function ---
def evalSphere(individual):
    return sum(x**2 for x in individual)  # Sphere function

# --- Crossover Function (Blend Crossover) ---
def cxBlend(ind1, ind2, alpha=0.5):
    size = len(ind1)
    child1 = []
    child2 = []
    for i in range(size):
        if random.random() < 0.5:
            child1.append(ind1[i])
            child2.append(ind2[i])
        else:
            diff = ind2[i] - ind1[i]
            child1.append(ind1[i] + alpha * diff)
            child2.append(ind2[i] - alpha * diff)
    return child1, child2

# --- Mutation Function (Gaussian Mutation) ---
def mutate(individual, mu=0, sigma=1, indpb=0.2):
    for i in range(len(individual)):
        if random.random() < indpb:
            # mean (mu) and standard deviation (sigma).
            individual[i] += random.gauss(mu, sigma)  # Gaussian mutation
    return individual

# --- Selection Function (Tournament Selection) ---
def tournament_selection(population, tournsize=3):
    #  for len of population time 
    # select randomly 3 records from population 
    # calculate fitness of each record using evalSphere function 
    # and select best one
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournsize)
        tournament.sort(key=lambda ind: evalSphere(ind))  # Sort by fitness (Sphere function)
        selected.append(tournament[0])  # Select best individual from tournament
    return selected

# --- Main Evolutionary Algorithm ---
def main():
    POP_SIZE = 100  # Population size
    NGEN = 60  # Number of generations
    CXPB = 0.5  # Crossover probability
    MUTPB = 0.2  # Mutation probability

    # Initialize population with random individuals
    # 100 x 10 matrix of random values between -5.12 to 5.12
    population = [[random.uniform(LOW, UP) for _ in range(IND_SIZE)] for _ in range(POP_SIZE)]

    
    # Track the best individual
    best_individual = None
    best_fitness = float('inf') # infinity

    # Evolve the population for NGEN generations
    for gen in range(NGEN):
        # Evaluate fitness for each individual in the population
        fitness_values = [evalSphere(ind) for ind in population]
        
        # Select the best individual
        min_fitness = min(fitness_values)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_individual = population[fitness_values.index(min_fitness)]
        
        print(f"Generation {gen+1} - Best Fitness: {best_fitness}")

        # Selection: Use tournament selection to choose parents
        selected_population = tournament_selection(population, tournsize=3)

        # Create the next generation
        next_population = []
        for i in range(0, POP_SIZE, 2):
            # Crossover
            if random.random() < CXPB:
                parent1 = selected_population[i]
                parent2 = selected_population[i + 1]
                child1, child2 = cxBlend(parent1, parent2)
                next_population.extend([child1, child2])
            else:
                next_population.extend([selected_population[i], selected_population[i + 1]])

        # Mutation
        for ind in next_population:
            if random.random() < MUTPB:
                mutate(ind)

        # Replace the old population with the new population
        population = next_population

    # Print final result
    print("\nBest Individual:", best_individual)
    print("Fitness Value:", best_fitness)

if __name__ == "__main__":
    main()
