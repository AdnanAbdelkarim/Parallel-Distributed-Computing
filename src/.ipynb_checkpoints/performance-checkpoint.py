import time
from src.sequential import run_sequential
from src.threading import run_threads
from src.processing import run_process

def compute_performance():
    # Run sequential and measure time
    print("Running Sequential Execution...")
    start_time = time.time()
    run_sequential()
    end_time = time.time()
    T_seq = end_time - start_time

    # Run threading and measure time
    print("\nRunning Threading Execution...")
    start_time = time.time()
    run_threads()
    end_time = time.time()
    T_thread = end_time - start_time

    # Run multiprocessing and measure time
    print("\nRunning Processing Execution...")
    start_time = time.time()
    run_process()
    end_time = time.time()
    T_process = end_time - start_time

    # Number of threads/processes used
    p_threads = 4  # Two for letters + Two for numbers
    p_process = 4  # Two for letters + Two for numbers

    # Compute speedups
    S_threads = T_seq / T_thread
    S_process = T_seq / T_process

    # Compute efficiency
    E_threads = S_threads / p_threads
    E_process = S_process / p_process

    # Amdahl’s Law Speedups
    f = 0.9  # Assuming 90% of the task is parallelizable
    S_A_threads = 1 / ((1 - f) + (f / p_threads))
    S_A_process = 1 / ((1 - f) + (f / p_process))

    # Gustafson’s Law Speedups
    S_G_threads = p_threads + (1 - p_threads) * f
    S_G_process = p_process + (1 - p_process) * f

    # Print results
    print("\nPerformance Analysis:")
    print(f"Sequential Execution Time: {T_seq:.6f} seconds")
    print(f"Threading Execution Time: {T_thread:.6f} seconds")
    print(f"Processing Execution Time: {T_process:.6f} seconds")

    print(f"\nSpeedup (Threads): {S_threads:.2f}")
    print(f"Speedup (Processing): {S_process:.2f}")

    print(f"Efficiency (Threads): {E_threads:.2f}")
    print(f"Efficiency (Processing): {E_process:.2f}")

    print(f"Amdahl’s Speedup (Threads): {S_A_threads:.2f}")
    print(f"Amdahl’s Speedup (Processing): {S_A_process:.2f}")

    print(f"Gustafson’s Speedup (Threads): {S_G_threads:.2f}")
    print(f"Gustafson’s Speedup (Processing): {S_G_process:.2f}")
