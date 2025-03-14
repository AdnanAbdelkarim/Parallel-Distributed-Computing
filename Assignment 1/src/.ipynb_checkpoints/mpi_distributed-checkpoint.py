from mpi4py import MPI
import numpy as np
from ShortestRoute.genetic_algorithms_functions import (
    calculate_fitness,
    select_in_tournament,
    order_crossover,
    mutate,
    generate_unique_population
)

def validate_route(route, distance_matrix):
    """
    Validate if a route is feasible based on the distance matrix.
    A route is feasible if every consecutive pair of nodes has a valid path.
    """
    penalty_threshold = 100000
    for i in range(len(route) - 1):
        if distance_matrix[route[i], route[i + 1]] >= penalty_threshold:
            return False
    if distance_matrix[route[-1], route[0]] >= penalty_threshold:
        return False
    return True

def generate_feasible_population(pop_size, num_cities, distance_matrix, max_attempts=10000):
    """
    Generates a feasible initial population by ensuring no unreachable edges exist in routes.
    """
    population = []
    attempts = 0
    while len(population) < pop_size and attempts < max_attempts:
        candidate = np.random.permutation(num_cities).tolist()
        if validate_route(candidate, distance_matrix):
            population.append(candidate)
        attempts += 1
    if len(population) < pop_size:
        print(f"Warning: Only generated {len(population)} feasible individuals out of {pop_size} requested.")
    return population

def parallel_genetic_algorithm(distance_matrix, pop_size=100, generations=500,
                                crossover_rate=0.8, mutation_rate=0.02,
                                elitism_size=2, patience=50, mutation_decay=0.99):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError(f"[Rank {rank}] Distance matrix must be square. Got shape: {distance_matrix.shape}")

    num_cities = distance_matrix.shape[0]

    if rank == 0:
        print(f"[Rank {rank}] Distance matrix shape: {distance_matrix.shape}")
        print(f"[Rank {rank}] Distance matrix sample (first 5 rows):\n{distance_matrix[:5, :5]}")
        population = generate_feasible_population(pop_size, num_cities, distance_matrix)
    else:
        population = None

    best_fitness_overall = -np.inf
    no_improvement_counter = 0

    for generation in range(generations):
        if rank == 0:
            chunks = np.array_split(population, size)
        else:
            chunks = None

        local_chunk = comm.scatter(chunks, root=0)
        local_fitness = np.array([calculate_fitness(route, distance_matrix) for route in local_chunk])
        all_fitnesses = comm.gather(local_fitness, root=0)

        if rank == 0:
            fitnesses = np.concatenate(all_fitnesses)
            max_fitness = max(fitnesses)

            if max_fitness > best_fitness_overall:
                best_fitness_overall = max_fitness
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= patience:
                print(f"[Rank {rank}] Early stopping at generation {generation} with best fitness {best_fitness_overall:.5f}")
                break

            mutation_rate = max(mutation_rate * mutation_decay, 0.001)
            sorted_population = [x for _, x in sorted(zip(fitnesses, population), reverse=True)]
            new_population = sorted_population[:elitism_size]

            parents = select_in_tournament(population, fitnesses,
                                           number_tournaments=(pop_size - elitism_size)//2,
                                           tournament_size=3)

            attempts = 0
            max_attempts = 10000

            while len(new_population) < pop_size and attempts < max_attempts:
                parent1 = parents[np.random.randint(len(parents))]
                parent2 = parents[np.random.randint(len(parents))]
                child1_partial, child2_partial = order_crossover(parent1[1:], parent2[1:])
                child1 = [0] + list(child1_partial)
                child2 = [0] + list(child2_partial)

                child1 = mutate(child1, mutation_rate)
                child2 = mutate(child2, mutation_rate)

                if validate_route(child1, distance_matrix):
                    new_population.append(child1)
                if validate_route(child2, distance_matrix):
                    new_population.append(child2)

                attempts += 1

            if len(new_population) < pop_size:
                print(f"[Rank {rank}] Only generated {len(new_population)} individuals after {attempts} attempts. Filling with elite individuals.")
                while len(new_population) < pop_size:
                    new_population.append(sorted_population[0])

            population = new_population[:pop_size]

            if generation % 50 == 0 or generation == generations - 1:
                print(f"[Rank {rank}] Generation {generation}: Best fitness = {best_fitness_overall:.5f}, Mutation rate = {mutation_rate:.5f}")

        population = comm.bcast(population, root=0)

    if rank == 0:
        final_fitnesses = np.array([calculate_fitness(route, distance_matrix) for route in population])
        best_index = np.argmax(final_fitnesses)
        best_route = population[best_index]
        best_distance = -final_fitnesses[best_index]

        print("\n✅ Enhanced Parallel GA finished!")
        print(f"Best route: {best_route}")
        print(f"Best distance: {best_distance:.2f}")

        return best_route, best_distance
