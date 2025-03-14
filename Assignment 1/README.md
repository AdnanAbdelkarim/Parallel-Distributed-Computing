
# DSAI 3202 – Parallel and Distributed Computing  
## **Assignment 1 – Multiprocessing and Distributed Genetic Algorithms**  
**Author:** Adnan Abdelkarim
**Course:** DSAI 3202  
**Instructor:** DR. Oussama Djedidi 

---

## 📁 Project Structure

```
Assignment1/
├── main.py                  # Combined entry point for Parts 1 & 2
├── src/                     # Source code for Part 1 and Part 2 helpers
│   ├── tasks.py
│   ├── sequential.py
│   ├── multiprocessing_tests.py
│   ├── connection_pool.py
│   └── mpi_distributed.py   # Parallel Genetic Algorithm (MPI)
├── ShortestRoute/           # Part 2: Genetic Algorithm data and logic
│   ├── city_distances.csv
│   ├── city_distances_extended.csv
│   ├── genetic_algorithm_trial.py
│   └── genetic_algorithms_functions.py
├── README.md                # This file
```

---

## ✅ Objectives

### Part 1: Multiprocessing (40 pts)
- Develop Python programs that leverage Python's multiprocessing capabilities.
- Implement various concurrency techniques:
  - Sequential processing
  - Multiprocessing (individual processes and pools)
  - concurrent.futures ProcessPoolExecutor
- Implement process synchronization using semaphores to manage access to a connection pool.

### Part 2: Distributed Genetic Algorithms (60 pts)
- Use **MPI4PY** to implement distributed genetic algorithms.
- Optimize delivery vehicle routes over a city graph:
  - Single vehicle version
  - Large-scale problem with 100+ nodes.
- Parallelize and enhance the algorithm across multiple machines.

---

## ⚙️ Tools and Concepts
| Tool / Library     | Usage                                  |
|--------------------|----------------------------------------|
| Python 3.x         | Core programming                      |
| multiprocessing    | Process-based parallelism (Part 1)    |
| concurrent.futures | Executor-based concurrency (Part 1)   |
| mpi4py             | Distributed computing (Part 2)        |
| pandas / numpy     | Data handling, matrix operations      |
| random             | Randomized operations in the GA       |

---

## 🚀 How to Run the Project

### ✅ 1. Activate Conda Environment  
```bash
conda activate parallel
```

### ✅ 2. Run the Combined `main.py`  
```bash
# For sequential and multiprocessing tasks:
python main.py
```

### ✅ 3. Run Parallel Genetic Algorithms with MPI  
```bash
# Example using 4 processes:
mpirun -np 4 python main.py
```

---

## 🧭 Menu Options (When Running `main.py`)

```
1 - Part 1: Multiprocessing Tests (Square Numbers)
2 - Part 1: Semaphore Demo (Connection Pool)
3 - Part 2: Sequential GA Trial (Part 5)
4 - Part 2: Parallel GA (Part 6)
5 - Part 2: Parallel GA Large Scale Problem (Parts 7 & 8)
```

---

## 📚 Explanation of the Program

### Part 1 - Multiprocessing (40 pts)
- **Square Program**
  - Runs sequential, per-process, and pool-based calculations on lists of size `10^6` and `10^7`.
  - Tests `map()`, `apply()`, `apply_async()`, and `concurrent.futures`.
- **Process Synchronization with Semaphores**
  - Implements a `ConnectionPool` class using `multiprocessing.Semaphore`.
  - Demonstrates safe access control with multiple processes.

---

### Part 2 - Distributed Genetic Algorithm (60 pts)
- **Fleet Management (Single Car Version):**
  - Uses a genetic algorithm to minimize total distance in a delivery route.
  - Sequential implementation (`run_sequential_ga_trial()`).
- **Parallel Genetic Algorithm with MPI:**
  - Distributes population evaluations across multiple processes.
  - Early stopping and mutation decay implemented.
- **Large-Scale Problem:**
  - Uses `city_distances_extended.csv` for 100 nodes and 4000 edges.
  - Ensures feasibility and timely execution.

---

## 📈 Performance Metrics (Summary)

| Task                                    | Time (Example) |
|-----------------------------------------|----------------|
| Part 1 - Sequential 10^6                | 0.06 sec       |
| Part 1 - Multiprocessing Pool (map)     | 0.09 sec       |
| Part 1 - concurrent.futures 10^6        | 106 sec        |
| Part 2 - Sequential GA (20 nodes)       | ~XX min        |
| Part 2 - Parallel GA (100 nodes, 4 CPUs)| ~XX min        |

---

## 📊 Observations and Conclusions

### Part 1
- **Multiprocessing Pool (map)** outperforms other techniques for mid-size data.
- **Per-process spawning** is expensive and impractical beyond ~1000 processes.
- **Semaphores** effectively limit concurrent access and prevent race conditions.

### Part 2
- **MPI parallelization** accelerates fitness evaluation in large populations.
- **Mutation rate decay and early stopping** prevent wasted computation cycles.
- **Sequential GA** becomes impractical as problem size increases.

---

## 🔧 Enhancements (Parts 6 & 7)
- Distributed execution over multiple machines via `mpirun`.
- Adaptive mutation and elitism improve convergence speed.
- Patience-based early stopping prevents unnecessary computation.

---

## 🏙️ Large Scale Problem (Part 8)
- Successfully executed on `city_distances_extended.csv` (100 nodes).
- Proposed multi-car solution (explained):
  - Partition nodes into clusters per vehicle.
  - Run parallel GAs per vehicle, then globally optimize.

---

## 📂 Deliverables  
✅ Fully documented Python code  
✅ README.md (this file)  
✅ Answers to assignment questions  
✅ Performance results and analysis  

---
