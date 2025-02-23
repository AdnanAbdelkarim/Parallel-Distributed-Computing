import threading

"""
Calculates the sum of numbers from 1 to n using multiple threads.
    
Parameters:
n (int): The upper limit of the range to sum.
num_threads (int): The number of threads to use.
    
Returns:
int: The sum of numbers from 1 to n computed using threads.
"""

def threaded_sum(n, num_threads=4):
    def partial_sum(start, end, result_list, index):
        result_list[index] = sum(range(start, end))
    
    threads = []
    chunk_size = n // num_threads
    results = [0] * num_threads
    
    for i in range(num_threads):
        start = i * chunk_size + 1
        end = (i + 1) * chunk_size + 1 if i != num_threads - 1 else n + 1
        thread = threading.Thread(target=partial_sum, args=(start, end, results, i))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    return sum(results)