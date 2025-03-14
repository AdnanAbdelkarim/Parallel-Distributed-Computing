from mpi4py import MPI
import numpy as np
from ShortestRoute.genetic_algorithms_functions import (
    calculate_fitness,
    select_in_tournament,
    order_crossover,
    mutate,
    generate_unique_population
)

def validate_route(route, num_cities):
    """
    Validates that a route includes all cities from 0 to num_cities-1 exactly once.
    
    Args:
        route (list): A proposed route (list of nodes).
        num_cities (int): Total number of cities (nodes).

    Returns:
        bool: True if the route is valid, False otherwise.
    """
    return (
        len(route) == num_cities
        and len(set(route)) == num_cities
        and max(route) < num_cities
        and min(route) >= 0
    )


def parallel_genetic_algorithm(
    distance_matrix,
    pop_size=100,
    generations=500,
    crossover_rate=0.8,   # Not used in current logic, reserved for future improvements
    mutation_rate=0.02,
    elitism_size=2,
    patience=50,          # Early stopping patience
    mutation_decay=0.99   # Mutation rate decay
):
    """
    Parallel Genetic Algorithm with adaptive mutation rate and early stopping.

    Args:
        distance_matrix (np.ndarray): Square matrix of distances between cities.
        pop_size (int): Population size.
        generations (int): Number of generations to evolve.
        crossover_rate (float): Reserved for future use.
        mutation_rate (float): Initial mutation probability.
        elitism_size (int): Number of top individuals to retain each generation.
        patience (int): Early stopping threshold (generations with no improvement).
        mutation_decay (float): Factor to decrease mutation rate over generations.

    Returns:
        tuple: (best_route, best_distance)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Validate distance matrix
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError(f"[Rank {rank}] Distance matrix must be square. Got shape: {distance_matrix.shape}")

    num_cities = distance_matrix.shape[0]

    # Initialize population on root process
    if rank == 0:
        print(f"[Rank {rank}] Distance matrix shape: {distance_matrix.shape}")
        print(f"[Rank {rank}] Number of cities: {num_cities}")
        population = generate_unique_population(pop_size, num_cities)
    else:
        population = None

    best_fitness_overall = -np.inf
    no_improvement_counter = 0

    for generation in range(generations):
        # Split population chunks across processes
        if rank == 0:
            chunks = np.array_split(population, size)
        else:
            chunks = None

        # Distribute chunks and evaluate fitness locally
        local_chunk = comm.scatter(chunks, root=0)
        local_fitness = np.array([calculate_fitness(route, distance_matrix) for route in local_chunk])

        # Gather fitness evaluations at root
        all_fitnesses = comm.gather(local_fitness, root=0)

        if rank == 0:
            fitnesses = np.concatenate(all_fitnesses)

            # Track best fitness (max because fitness is negative distance)
            max_fitness = max(fitnesses)
            if max_fitness > best_fitness_overall:
                best_fitness_overall = max_fitness
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            # Early stopping if stagnation detected
            if no_improvement_counter >= patience:
                print(f"[Rank {rank}] Early stopping at generation {generation} "
                      f"with best fitness {-best_fitness_overall:.2f}")
                break

            # Decay mutation rate for exploration-exploitation balance
            mutation_rate = max(mutation_rate * mutation_decay, 0.001)

            # Elitism: Retain top individuals
            sorted_population = [
                x for _, x in sorted(zip(fitnesses, population), reverse=True)
            ]
            new_population = sorted_population[:elitism_size]

            # Parent selection for generating offspring
            parents = select_in_tournament(
                population, fitnesses,
                number_tournaments=(pop_size - elitism_size) // 2,
                tournament_size=3
            )

            # Generate offspring with crossover and mutation
            while len(new_population) < pop_size:
                parent1 = parents[np.random.randint(len(parents))]
                parent2 = parents[np.random.randint(len(parents))]

                child1_partial, child2_partial = order_crossover(parent1[1:], parent2[1:])

                # Add depot (node 0) back at the start
                child1 = [0] + child1_partial
                child2 = [0] + child2_partial

                # Mutate children
                child1 = mutate(child1, mutation_rate)
                child2 = mutate(child2, mutation_rate)

                # Validate and add children to population
                if validate_route(child1, num_cities):
                    new_population.append(child1)
                else:
                    print(f"[Rank {rank}] Invalid child1 detected: {child1}")

                if len(new_population) < pop_size and validate_route(child2, num_cities):
                    new_population.append(child2)
                elif len(new_population) < pop_size:
                    print(f"[Rank {rank}] Invalid child2 detected: {child2}")

            # Update population for next generation
            population = new_population[:pop_size]

            # Logging generation progress
            if generation % 50 == 0 or generation == generations - 1:
                print(f"[Rank {rank}] Generation {generation}: "
                      f"Best fitness = {-best_fitness_overall:.2f}, "
                      f"Mutation rate = {mutation_rate:.4f}")

        # Synchronize updated population across processes
        population = comm.bcast(population, root=0)

    # Final evaluation and result on root
    if rank == 0:
        final_fitnesses = np.array([calculate_fitness(route, distance_matrix) for route in population])
        best_index = np.argmax(final_fitnesses)
        best_route = population[best_index]
        best_distance = -final_fitnesses[best_index]

        print("\n✅ Parallel GA completed!")
        print(f"Best route: {best_route}")
        print(f"Best distance: {best_distance:.2f}")

        return best_route, best_distance
