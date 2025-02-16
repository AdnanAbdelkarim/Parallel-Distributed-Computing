import time
import threading
import multiprocessing
from PartB.src.sequential import sequential_sum
from PartB.src.thread import parallel_sum
from PartB.src.processing import multiprocessing_sum

def main():
    n = 10**6  # Adjust the number for testing
    num_threads = 4
    num_processes = 4
    
    # Sequential Execution
    start_time = time.time()
    seq_result = sequential_sum(n)
    seq_time = time.time() - start_time
    print(f"Sequential Sum: {seq_result}, Execution Time: {seq_time:.4f} sec")
    
    # Threading Execution
    thread_result, thread_time = parallel_sum(n, num_threads)
    print(f"Threading Sum: {thread_result}, Execution Time: {thread_time:.4f} sec")
    
    # Multiprocessing Execution
    process_result, process_time = multiprocessing_sum(n, num_processes)
    print(f"Multiprocessing Sum: {process_result}, Execution Time: {process_time:.4f} sec")
    
if __name__ == "__main__":
    main()