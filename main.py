import time
import threading
import multiprocessing
from src.sequential import sequential_sum as seq_sum_partA
from src.threading import threaded_sum as thread_sum_partA
from src.processing import process_sum as proc_sum_partA

def main():
    n = 10**8  # Adjust for the first set of tests
    num_threads = 4
    num_processes = 4
    
    # Part A: Sequential, Threaded, Processed Summation
    start_time = time.time()
    seq_result = seq_sum_partA(n)
    seq_time = time.time() - start_time
    print(f"Part A - Sequential Sum from 1 to n: {seq_result}, Time: {seq_time:.4f} sec")

    start_time = time.time()
    thread_result = thread_sum_partA(n)
    thread_time = time.time() - start_time
    print(f"Part A - Threaded Sum: {thread_result}, Time: {thread_time:.4f} sec")
    
    start_time = time.time()
    process_result = proc_sum_partA(n)
    process_time = time.time() - start_time
    print(f"Part A - Process Sum: {process_result}, Time: {process_time:.4f} sec")

    print("Part A - Speedup (Threads):", seq_time / thread_time)
    print("Part A - Speedup (Processes):", seq_time / process_time)



if __name__ == "__main__":
    main()