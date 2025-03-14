import time
import numpy as np
import pandas as pd
from mpi4py import MPI

# Import GA functions
from ShortestRoute.genetic_algorithms_functions import (
    calculate_fitness,
    select_in_tournament,
    order_crossover,
    mutate,
    generate_unique_population
)


# Import Parallel GA implementation
from src.mpi_distributed import parallel_genetic_algorithm


# ------------------------------------
# PART 2: SEQUENTIAL GA TRIAL (Part 5)
# ------------------------------------
def run_sequential_ga_trial():
    """
    Runs the sequential genetic algorithm trial on city_distances.csv.
    """
    print("\n=== PART 2: Sequential Genetic Algorithm Trial ===\n")

    distance_matrix = np.genfromtxt('ShortestRoute/city_distances.csv', delimiter=',', skip_header=1)

    num_nodes = distance_matrix.shape[0]
    population_size = 1000
    num_generations = 200
    stagnation_limit = 5
    mutation_rate = 0.1

    # Initialize population
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

            # order_crossover returns two offspring
            route1, route2 = order_crossover(p1[1:], p2[1:])

            # Convert to lists if needed and prepend depot node 0
            child1 = [0] + list(route1)
            child2 = [0] + list(route2)

            offspring.append(child1)
            offspring.append(child2)

        # Mutate each offspring route
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


# ------------------------------------
# PART 2: PARALLEL GA (Part 6)
# ------------------------------------
def run_parallel_ga():
    """
    Runs the parallel genetic algorithm on city_distances.csv.
    """
    print("\n=== PART 2: Parallel Genetic Algorithm ===\n")

    distance_matrix = np.genfromtxt('ShortestRoute/city_distances_extended.csv', delimiter=',', skip_header=1)

    start_time = MPI.Wtime()

    best_route, best_distance = parallel_genetic_algorithm(
        distance_matrix,
        pop_size=200,
        generations=500,
        crossover_rate=0.8,
        mutation_rate=0.05,
        elitism_size=4,
        patience=50
    )

    end_time = MPI.Wtime()

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        print("\n--- Parallel GA Results ---")
        print("Best route:", best_route)
        print("Total distance:", best_distance)
        print(f"Execution time: {end_time - start_time:.2f} seconds\n")


# ------------------------------------
# PART 2: PARALLEL GA LARGE SCALE (Parts 7 & 8)
# ------------------------------------
def run_parallel_ga_large_scale():
    """
    Runs the parallel genetic algorithm on city_distances_extended.csv.
    """
    print("\n=== PART 2: Parallel Genetic Algorithm on Large Scale ===\n")

    distance_matrix = np.genfromtxt('ShortestRoute/city_distances_extended.csv', delimiter=',',skip_header=1 )



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


# ------------------------------------
# MAIN FUNCTION
# ------------------------------------
def main():
    """
    Main entry point to run Assignment Part 2.
    """
    print("Select option to run:\n")
    print("1 - Sequential GA Trial (Part 5)")
    print("2 - Parallel GA (Part 6)")
    print("3 - Parallel GA Large Scale Problem (Parts 7 & 8)")

    choice = input("Enter choice (1/2/3): ").strip()

    if choice == '1':
        run_sequential_ga_trial()

    elif choice == '2':
        run_parallel_ga()

    elif choice == '3':
        run_parallel_ga_large_scale()

    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
