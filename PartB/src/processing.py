import multiprocessing
import time

def worker(process_id, start, end, result):
    partial_sum = sum(range(start, end + 1))
    result[process_id] = partial_sum
    print(f"Process {process_id} finished summing from {start} to {end} with sum {partial_sum}")

def multiprocessing_sum(n, num_processes):
    chunk_size = n // num_processes
    manager = multiprocessing.Manager()
    result = manager.list([0] * num_processes)
    processes = []
    
    start_time = time.time()
    
    for i in range(num_processes):
        start = i * chunk_size + 1
        end = (i + 1) * chunk_size if i != num_processes - 1 else n
        process = multiprocessing.Process(target=worker, args=(i, start, end, result))
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()
    
    total_sum = sum(result)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return total_sum, execution_time