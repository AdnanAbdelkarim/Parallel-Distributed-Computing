import time
from mpi4py import MPI
import numpy as np
import pandas as pd
from genetic_algorithms_functions import calculate_fitness, select_in_tournament, order_crossover, mutate, generate_unique_population

start_time = time.time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    distance_matrix = pd.read_csv('city_distances_extended.csv').to_numpy()
else:
    distance_matrix = None

distance_matrix = comm.bcast(distance_matrix, root=0)

num_nodes = distance_matrix.shape[0]
population_size = 10000
num_tournaments = 4
mutation_rate = 0.1
num_generations = 200
infeasible_penalty = 1e6
stagnation_limit = 5

if rank == 0:
    np.random.seed(42)
    population = generate_unique_population(population_size, num_nodes)
else:
    population = None

best_fitness = int(1e6)
stagnation_counter = 0

for generation in range(num_generations):
    if rank == 0:
        chunks = np.array_split(population, size)
    else:
        chunks = None
    local_chunk = comm.scatter(chunks, root=0)
    local_fitness = np.array([calculate_fitness(route, distance_matrix) for route in local_chunk])
    fitness_values = comm.gather(local_fitness, root=0)
    if rank == 0:
        fitness_values = np.concatenate(fitness_values)
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
        selected = select_in_tournament(population, fitness_values, num_tournaments)
        offspring = []
        for i in range(0, len(selected), 2):
            parent1, parent2 = selected[i], selected[i + 1]
            route1 = order_crossover(parent1[1:], parent2[1:])
            offspring.append([0] + route1)
        mutated_offspring = [mutate(route, mutation_rate) for route in offspring]
        for i, idx in enumerate(np.argsort(fitness_values)[::-1][:len(mutated_offspring)]):
            population[idx] = mutated_offspring[i]
        unique_population = set(tuple(ind) for ind in population)
        while len(unique_population) < population_size:
            individual = [0] + list(np.random.permutation(np.arange(1, num_nodes)))
            unique_population.add(tuple(individual))
        population = [list(individual) for individual in unique_population]
        print(f"Generation {generation}: Best fitness = {current_best_fitness}")
if rank == 0:
    fitness_values = np.array([calculate_fitness(route, distance_matrix) for route in population])
    best_idx = np.argmin(fitness_values)
    best_solution = population[best_idx]
    print("Best Solution:", best_solution)
    print("Total Distance:", calculate_fitness(best_solution, distance_matrix))
end_time = time.time()
print(f"Parallel time: {end_time - start_time} seconds")
