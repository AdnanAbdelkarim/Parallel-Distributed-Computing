import time
import threading
import multiprocessing
from PartA.src.sequential import sequential_sum as seq_sum_partA
from PartA.src.threading import threaded_sum as thread_sum_partA
from PartA.src.processing import process_sum as proc_sum_partA
from PartB.src.sequential import sequential_sum as seq_sum_partB
from PartB.src.thread import parallel_sum as thread_sum_partB
from PartB.src.processing import multiprocessing_sum as proc_sum_partB

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

    # Part B: Sequential, Threading, Multiprocessing Summation
    n_partB = 10**6  # Adjust for Part B test
    # Sequential Execution (Part B)
    start_time = time.time()
    seq_resultB = seq_sum_partB(n_partB)
    seq_timeB = time.time() - start_time
    print(f"Part B - Sequential Sum: {seq_resultB}, Time: {seq_timeB:.4f} sec")
    
    # Threading Execution (Part B)
    thread_resultB, thread_timeB = thread_sum_partB(n_partB, num_threads)
    print(f"Part B - Threading Sum: {thread_resultB}, Time: {thread_timeB:.4f} sec")
    
    # Multiprocessing Execution (Part B)
    process_resultB, process_timeB = proc_sum_partB(n_partB, num_processes)
    print(f"Part B - Multiprocessing Sum: {process_resultB}, Time: {process_timeB:.4f} sec")

if __name__ == "__main__":
    main()
