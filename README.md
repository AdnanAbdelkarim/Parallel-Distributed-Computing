
# DSAI 3202: Parallel and Distributed Computing Assignment1

Done By: Adnan Abdelkarim - 60100667

## Overview
This repository contains the Python programs for the assignments of the DSAI 3202 course, focusing on parallel and distributed computing techniques using multiprocessing and genetic algorithms.

## Assignment 1 - Part 1: Multiprocessing
### Objectives:
Develop Python programs that leverage Python multiprocessing capabilities to perform various tasks.

### Tools and Concepts:
- Python programming language.
- `multiprocessing` and `concurrent.futures` packages.

### Files:
- `Square.py`: Contains functions to test sequential and multiprocessing performance of squaring numbers.

### Execution:
Run `Square.py` to perform the tests. It includes sequential, multiprocessing for loop, multiprocessing pool, and `ProcessPoolExecutor` tests.

## Assignment 1 - Part 2: Genetic Algorithms
### Objectives:
Develop Python programs that run genetic algorithms in a distributed fashion using MPI4PY or Celery.

### Tools and Concepts:
- Python programming language.
- `MPI4PY` package.

### Files:
- `genetic_algorithms_functions.py`: Contains functions for genetic algorithm operations such as fitness calculation, selection, crossover, and mutation.
- `genetic_algorithm_trial.py`: Script to run the genetic algorithm trial.
- `genetic_algorithm_parallel.py`: Script to run the genetic algorithm in parallel across multiple machines.

### Execution:
1. Run `genetic_algorithm_trial.py` for a sequential execution of the genetic algorithm.
2. Run `genetic_algorithm_parallel.py` to execute the algorithm in parallel using MPI.

## Setup Instructions
1. Ensure Python 3.x is installed on your system.
2. Install required packages:
   ```bash
   pip install numpy pandas mpi4py
   ```


