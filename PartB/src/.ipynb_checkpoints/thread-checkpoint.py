import threading
import time

def worker(thread_id, start, end, result):
    partial_sum = sum(range(start, end + 1))  # Calculate sum of the assigned range
    result[thread_id] = partial_sum  # Store the result in the result list
    print(f"Thread {thread_id} finished summing from {start} to {end} with sum {partial_sum}")

def parallel_sum(n, num_threads):
    chunk_size = n // num_threads
    result = [0] * num_threads
    threads = []
    
    start_time = time.time()
    
    for i in range(num_threads):
        start = i * chunk_size + 1
        end = (i + 1) * chunk_size if i != num_threads - 1 else n
        thread = threading.Thread(target=worker, args=(i, start, end, result))
        threads.append(thread)
        thread.start()
        
    for thread in threads:
        thread.join()
    
    total_sum = sum(result)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return total_sum, execution_time