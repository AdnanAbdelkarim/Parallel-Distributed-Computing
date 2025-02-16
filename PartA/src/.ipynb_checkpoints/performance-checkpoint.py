import time

# Function to measure execution time of a function
def measure_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

# Function to calculate speedup
def calculate_speedup(sequential_time, parallel_time):
    return sequential_time / parallel_time

# Function to calculate efficiency
def calculate_efficiency(speedup, num_processors):
    return speedup / num_processors

# Function to calculate Amdahl's Law Speedup
def amdahls_law_speedup(num_processors, sequential_time):
    return 1 / (sequential_time / num_processors + (1 - sequential_time))

# Function to calculate Gustafson's Law Speedup
def gustafsons_law_speedup(num_processors, parallel_time):
    return num_processors - (1 - parallel_time) * num_processors
