import numpy as np
import random

def calculate_fitness(route, distance_matrix):
    """
    Calculates the negative total distance traveled by the car.

    Parameters:
        - route (list): A list representing the order of nodes visited in the route.
        - distance_matrix (numpy.ndarray): A matrix of the distances between nodes.

    Returns:
        - float: The negative total distance traveled (negative because we want to minimize distance).
                 Returns a large negative penalty if the route is infeasible.
    """
    total_distance = 0
    penalty = -1000000

    # Loop through each node and its next in the route
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        dist = distance_matrix[from_node][to_node]

        if dist >= 100000:
            # Invalid edge
            return penalty
        total_distance += dist

    # Close the loop to the starting node
    from_node = route[-1]
    to_node = route[0]
    dist = distance_matrix[from_node][to_node]
    if dist >= 100000:
        return penalty

    total_distance += dist

    return -total_distance







import numpy as np

def select_in_tournament(population, scores, number_tournaments=4, tournament_size=3):
    """
    Tournament selection for genetic algorithm.
    """
    selected = []

    pop_size = len(population)
    if pop_size == 0:
        print("[WARNING] Population is empty during selection.")
        return selected

    for _ in range(number_tournaments):
        if pop_size < tournament_size:
            print("[WARNING] Tournament size larger than population size.")
            break
        
        idx = np.random.choice(pop_size, size=tournament_size, replace=False)
        best_idx = idx[np.argmax(scores[idx])]
        selected.append(population[best_idx])

    if len(selected) == 0:
        print("[WARNING] No parents selected. Returning empty list.")
    
    return selected


def order_crossover(parent1, parent2):
    """
    Order crossover (OX) for permutations.
    
    Produces two offspring from two parents.

    Parameters:
        - parent1 (list): The first parent route.
        - parent2 (list): The second parent route.

    Returns:
        - tuple: Two offspring routes.
    """
    size = len(parent1)

    # Step 1: Select random crossover points
    start, end = sorted(np.random.choice(range(size), 2, replace=False))

    # Step 2: Create children templates
    child1 = [None] * size
    child2 = [None] * size

    # Step 3: Copy the segment from each parent to child
    child1[start:end+1] = parent1[start:end+1]
    child2[start:end+1] = parent2[start:end+1]

    # Step 4: Fill the remaining positions from the other parent (in order)
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
    



