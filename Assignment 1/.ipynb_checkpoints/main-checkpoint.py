import time
import multiprocessing
import numpy as np
from mpi4py import MPI
from src.tasks import generate_numbers
from src.sequential import sequential_square
from src.multiprocessing_tests import (
    multiprocessing_per_process,
    pool_map,
    pool_apply,
    pool_apply_async,
    concurrent_futures_executor
)
from src.connection_pool import ConnectionPool, access_database
from src.mpi_distributed import parallel_genetic_algorithm
from ShortestRoute.genetic_algorithms_functions import (
    calculate_fitness,
    select_in_tournament,
    order_crossover,
    mutate,
    generate_unique_population
)

# -------------------- PART 1 Functions -------------------- #

def run_and_time(func, numbers, label):
    start = time.time()
    func(numbers)
    end = time.time()
    print(f"{label} took {end - start:.4f} seconds")

def multiprocessing_tests():
    n1 = 10**6
    numbers_1m = generate_numbers(n1)

    print(f"\nRunning square number tests with {n1} numbers...\n")
    run_and_time(sequential_square, numbers_1m, "Sequential for loop")
    run_and_time(multiprocessing_per_process, numbers_1m[:1000], "Multiprocessing: process per number (limited to 1000)")
    run_and_time(pool_map, numbers_1m, "Multiprocessing Pool: map()")
    run_and_time(pool_apply, numbers_1m, "Multiprocessing Pool: apply()")
    run_and_time(pool_apply_async, numbers_1m, "Multiprocessing Pool: apply_async()")
    run_and_time(concurrent_futures_executor, numbers_1m, "concurrent.futures ProcessPoolExecutor")

def semaphore_demo():
    max_connections = 3
    total_processes = 10

    with multiprocessing.Manager() as manager:
        shared_connections = manager.list([f"Connection-{i+1}" for i in range(max_connections)])
        shared_semaphore = multiprocessing.Semaphore(max_connections)
        pool = ConnectionPool(max_connections, shared_connections, shared_semaphore)

        processes = []
        print(f"\n--- Semaphore Demo ---\n")
        print(f"Total processes: {total_processes}")
        print(f"Max simultaneous connections: {max_connections}\n")

        for i in range(total_processes):
            process = multiprocessing.Process(target=access_database, args=(pool, i + 1))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        print("\nAll processes have finished accessing the database.\n")

# -------------------- PART 2 Functions -------------------- #

def run_sequential_ga_trial():
    print("\n=== PART 2: Sequential Genetic Algorithm Trial ===\n")
    distance_matrix = np.genfromtxt('ShortestRoute/city_distances.csv', delimiter=',', skip_header=1)

    num_nodes = distance_matrix.shape[0]
    population_size = 1000
    num_generations = 200
    stagnation_limit = 5
    mutation_rate = 0.1

    population = generate_unique_population(population_size, num_nodes)
    best_fitness = float('inf')
    stagnation_counter = 0

    start_time = time.time()

    for generation in range(num_generations):
        fitness_values = np.array([calculate_fitness(route, distance_matrix) for route in population])
        current_best_fitness = np.min(fitness_values)

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter >= stagnation_limit:
            print(f"Regenerating population at generation {generation} due to stagnation")
            best_individual = population[np.argmin(fitness_values)]
            population = generate_unique_population(population_size - 1, num_nodes)
            population.append(best_individual)
            stagnation_counter = 0
            continue

        selected = select_in_tournament(population, fitness_values)
        offspring = []
        for i in range(0, len(selected), 2):
            p1 = selected[i]
            p2 = selected[min(i + 1, len(selected) - 1)]
            route1, route2 = order_crossover(p1[1:], p2[1:])
            offspring.append([0] + list(route1))
            offspring.append([0] + list(route2))

        mutated_offspring = [mutate(route, mutation_rate) for route in offspring]

        for i, idx in enumerate(np.argsort(fitness_values)[::-1][:len(mutated_offspring)]):
            population[idx] = mutated_offspring[i]

        if generation % 10 == 0 or generation == num_generations - 1:
            print(f"Generation {generation}: Best fitness = {best_fitness}")

    end_time = time.time()

    fitness_values = np.array([calculate_fitness(route, distance_matrix) for route in population])
    best_idx = np.argmin(fitness_values)
    best_solution = population[best_idx]

    print("\n--- Sequential GA Results ---")
    print("Best route:", best_solution)
    print("Total distance:", fitness_values[best_idx])
    print(f"Execution time: {end_time - start_time:.2f} seconds\n")

def run_parallel_ga():
    """
    Runs the parallel genetic algorithm on city_distances.csv.
    """
    print("\n=== PART 2: Parallel Genetic Algorithm ===\n")

    # Load the distance matrix
    distance_matrix = np.genfromtxt('ShortestRoute/city_distances.csv', delimiter=',', skip_header=1)

    # Basic validation
    print(f"[Rank 0] Distance matrix shape: {distance_matrix.shape}")
    
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError(f"[Rank 0] Distance matrix is not square: {distance_matrix.shape}")

    # OPTIONAL: Print matrix sample to check its content
    print(f"[Rank 0] Distance matrix sample (first 5 rows):\n{distance_matrix[:5,:5]}")

    # Start parallel genetic algorithm
    start_time = MPI.Wtime()

    best_route, best_distance = parallel_genetic_algorithm(
        distance_matrix,
        pop_size=200,           # You can tweak the population size
        generations=500,        # Number of generations
        crossover_rate=0.8,     # Crossover rate (currently not directly used but passed)
        mutation_rate=0.05,     # Mutation rate
        elitism_size=4,         # Number of elite individuals
        patience=50             # Early stopping after 50 generations without improvement
    )

    end_time = MPI.Wtime()

    # Report results (only on rank 0)
    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        print("\n--- Parallel GA Results ---")
        print("Best route:", best_route)
        print("Total distance:", best_distance)
        print(f"Execution time: {end_time - start_time:.2f} seconds\n")


def run_parallel_ga_large_scale():
    print("\n=== PART 2: Parallel Genetic Algorithm on Large Scale ===\n")
    distance_matrix = np.genfromtxt('ShortestRoute/city_distances_extended.csv', delimiter=',', skip_header=1)

    start_time = MPI.Wtime()

    best_route, best_distance = parallel_genetic_algorithm(
        distance_matrix,
        pop_size=500,
        generations=1000,
        crossover_rate=0.8,
        mutation_rate=0.05,
        elitism_size=10,
        patience=100
    )

    end_time = MPI.Wtime()
    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        print("\n--- Parallel GA Large Scale Results ---")
        print("Best route:", best_route)
        print("Total distance:", best_distance)
        print(f"Execution time: {end_time - start_time:.2f} seconds\n")

# -------------------- Main Menu -------------------- #

def main():
    print("Select option to run:\n")
    print("1 - Part 1: Multiprocessing Square Tests")
    print("2 - Part 1: Semaphore Connection Pool Demo")
    print("3 - Part 2: Sequential GA Trial")
    print("4 - Part 2: Parallel GA")
    print("5 - Part 2: Parallel GA Large Scale")

    choice = input("\nEnter choice (1/2/3/4/5): ").strip()

    if choice == '1':
        multiprocessing_tests()

    elif choice == '2':
        semaphore_demo()

    elif choice == '3':
        run_sequential_ga_trial()

    elif choice == '4':
        run_parallel_ga()

    elif choice == '5':
        run_parallel_ga_large_scale()

    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
