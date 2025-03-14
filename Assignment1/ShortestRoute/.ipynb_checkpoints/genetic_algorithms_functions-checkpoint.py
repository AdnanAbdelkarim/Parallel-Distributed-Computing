import numpy as np


def calculate_fitness(route, distance_matrix):
    """
    calculate_fitness function: total distance traveled by the car.

    Parameters:
        - route (list): A list representing the order of nodes visited in the route.
        - distance_matrix (numpy.ndarray): A matrix of the distances between nodes.

    Returns:
        - float: The negative total distance traveled (negative because we want to minimize distance).
                 Returns a large negative penalty if the route is infeasible.
    """
    total_distance = 0
    penalty = 1e6  # Large penalty for infeasible routes

    # Iterate over consecutive nodes in the route
    for i in range(len(route) - 1):
        node1 = route[i]
        node2 = route[i + 1]
        distance = distance_matrix[node1][node2]

        # Check for infeasible route
        if distance == 100000:
            return penalty

        total_distance += distance

    # Add return to depot (node 0)
    distance_back = distance_matrix[route[-1]][0]
    if distance_back == 100000:
        return penalty

    total_distance += distance_back

    # Return negative because GA maximizes fitness
    return total_distance



def select_in_tournament(population,
                         scores,
                         number_tournaments=4,
                         tournament_size=3):
    """
    Tournament selection for genetic algorithm.

    Parameters:
        - population (list): The current population of routes.
        - scores (np.array): The calculate_fitness scores corresponding to each individual in the population.
        - number_tournaments (int): The number of tournaments to run in the population.
        - tournament_size (int): The number of individuals to compete in each tournament.

    Returns:
        - list: A list of selected individuals for crossover.
    """
    selected = []

    for _ in range(number_tournaments):
        # Randomly select tournament participants by indices
        idx = np.random.choice(len(population), size=tournament_size, replace=False)

        # Find the participant with the lowest distance (best fitness)
        best_idx = idx[np.argmin(scores[idx])]

        # Add the best individual to the selected list
        selected.append(population[best_idx])

    return selected



def order_crossover(parent1, parent2):
    """
    Order Crossover (OX) for permutations.

    Args:
        parent1 (list): First parent.
        parent2 (list): Second parent.

    Returns:
        tuple: Two children as lists.
    """
    size = len(parent1)

    # Choose two crossover points randomly
    start, end = sorted(np.random.choice(range(size), 2, replace=False))

    # Initialize offspring with None
    child1 = [None] * size
    child2 = [None] * size

    # Copy crossover segment from parents to children
    child1[start:end + 1] = parent1[start:end + 1]
    child2[start:end + 1] = parent2[start:end + 1]

    # Fill the remaining positions from the other parent
    def fill_child(child, parent):
        current_pos = (end + 1) % size
        parent_pos = (end + 1) % size

        while None in child:
            if parent[parent_pos] not in child:
                child[current_pos] = parent[parent_pos]
                current_pos = (current_pos + 1) % size
            parent_pos = (parent_pos + 1) % size

        return child

    child1 = fill_child(child1, parent2)
    child2 = fill_child(child2, parent1)

    return child1, child2

def mutate(route,
           mutation_rate = 0.1):
    """
    Mutation operator: swap two nodes in the route.

    Parameters:
        - route (list): The route to mutate.
        - mutation_rate (float): The chance to mutate an individual.
    Returns:
        - list: The mutated route.
    """
    if np.random.rand() < mutation_rate:
        i, j = np.random.choice(len(route), 2, replace=False)
        route[i], route[j] = route[j], route[i]
    return route

def generate_unique_population(population_size, num_nodes):
    """
    Generate a unique population of individuals for a genetic algorithm.

    Each individual in the population represents a route in a graph, where the first node is fixed (0) and the 
    remaining nodes are a permutation of the other nodes in the graph. This function ensures that all individuals
    in the population are unique.

    Parameters:
        - population_size (int): The desired size of the population.
        - num_nodes (int): The number of nodes in the graph, including the starting node.

    Returns:
        - list of lists: A list of unique individuals, where each individual is represented as a list of node indices.
    """
    population = set()
    while len(population) < population_size:
        individual = [0] + list(np.random.permutation(np.arange(1, num_nodes)))
        population.add(tuple(individual))
    return [list(ind) for ind in population]
