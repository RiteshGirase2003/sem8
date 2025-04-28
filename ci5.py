import numpy as np
import random
import matplotlib.pyplot as plt

# --- Problem Setup: TSP ---
# Coordinates of cities
cities = np.array([
    [0, 0],
    [1, 5],
    [5, 2],
    [7, 8],
    [8, 3],
    [6, 3],
    [2, 1],
    [6, 7],
])

n_cities = len(cities)

# Distance matrix - from each city to every other city
# using Euclidean distance
distances = np.zeros((n_cities, n_cities)) # 8 x 8
for i in range(n_cities):
    for j in range(n_cities):
        if i != j:
            distances[i][j] = ((cities[i][0] - cities[j][0])**2 + (cities[i][1] - cities[j][1])**2) ** 0.5

# --- ACO Parameters ---
n_ants = 20
n_best = 5
n_iterations = 10
decay = 0.5  # pheromone evaporation rate
alpha = 1    # pheromone importance
beta = 2     # distance importance

# Initialize pheromone levels
pheromone = np.ones((n_cities, n_cities)) / n_cities # 8 x 8 matrix of 1/n_cities which is 1/8 = 0.125


# --- Helper Functions ---
def route_length(route):
    total = 0
    #  total distance needed to travel from city 0 → 6 → 4 → 5 → 2 → 7 → 3 → 1 → 0 (back to start) 
    for i in range(n_cities):
        from_city = route[i % n_cities]
        to_city = route[(i + 1) % n_cities]
        total += distances[from_city][to_city]
    return total


def choose_next_city(pheromone, visibility, visited, current_city):
    pheromone_power = pheromone[current_city]** alpha
    visibility_power = visibility[current_city]** beta

    probs = pheromone_power * visibility_power
    probs[list(visited)] = 0
    probs /= np.sum(probs) # divide each element by sume of all element
    return np.argmax(probs)

# --- Main ACO Loop ---
best_route = None
best_length = float('inf') # infinity
# visibility is 8 x 8 matrix 
visibility = 1 / (distances + 1e-10)  # avoid division by zero


for iteration in range(n_iterations):
    all_routes = []
    all_lengths = []

    for ant in range(n_ants):
        # randomly select starting city
        route = [random.randint(0, n_cities - 1)]

        visited = set(route)


        # contruct route by finding next cities
        for _ in range(n_cities - 1):
            next_city = choose_next_city(pheromone, visibility, visited, route[-1])
            route.append(next_city)
            visited.add(next_city)
        
        # their will 20 routes bcz their are 20 ants
        all_routes.append(route)
        all_lengths.append(route_length(route))
    
    # Update pheromones
    # argsort returns indices that would sort the array
    sorted_indices = np.argsort(all_lengths)
    # pheromone evaporation
    pheromone *= (1 - decay)
    
    for i in sorted_indices[:n_best]:
        route = all_routes[i]
        length = all_lengths[i]
        for j in range(n_cities):
            from_city = route[j % n_cities]
            to_city = route[(j + 1) % n_cities]
            # pheromone deposit on the path taken by the best ants
            pheromone[from_city][to_city] += 1.0 / length

    # Keep track of best
    if min(all_lengths) < best_length:
        best_length = min(all_lengths)
        best_route = all_routes[np.argmin(all_lengths)]

    print(f"Iteration {iteration+1}: Best length so far = {best_length:.4f}")

# --- Output Results ---
print("\nBest Route:", best_route)
print("Best Length:", best_length)

# --- Plot ---
best_route_coords = cities[best_route + [best_route[0]]]  # close the loop
# plt.figure(figsize=(8,6))
# plt.plot(best_route_coords[:,0], best_route_coords[:,1], 'o-', label='Path')
# for i, (x, y) in enumerate(cities):
#     plt.text(x, y, str(i), fontsize=12, ha='right')
# plt.title("Best Path found by Ant Colony Optimization")
# plt.xlabel("X Coordinate")
# plt.ylabel("Y Coordinate")
# plt.grid(True)
# plt.legend()
# plt.show()
