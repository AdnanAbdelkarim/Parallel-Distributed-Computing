import multiprocessing

"""
Calculates the sum of numbers from 1 to n using multiple processes.
    
Parameters:
n (int): The upper limit of the range to sum.
num_processes (int): The number of processes to use.
    
Returns:
int: The sum of numbers from 1 to n computed using processes.
"""

def process_sum(n, num_processes=4):
    def partial_sum(start, end, queue):
        queue.put(sum(range(start, end)))
    
    processes = []
    chunk_size = n // num_processes
    queue = multiprocessing.Queue()
    
    for i in range(num_processes):
        start = i * chunk_size + 1
        end = (i + 1) * chunk_size + 1 if i != num_processes - 1 else n + 1
        process = multiprocessing.Process(target=partial_sum, args=(start, end, queue))
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()
    
    total_sum = sum(queue.get() for _ in range(num_processes))
    return total_sum