# src/multiprocessing_tests.py

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from src.tasks import square

def multiprocessing_per_process(numbers):
    """
    Creates a process for each number (limited to small lists for practical reasons).
    
    Parameters:
    numbers (list): List of numbers.
    
    Returns:
    list: List of squared numbers.
    """

    def worker(num, queue):
        queue.put(square(num))

    processes = []
    queue = multiprocessing.Queue()

    for num in numbers:
        p = multiprocessing.Process(target=worker, args=(num, queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    result = [queue.get() for _ in processes]
    return result

def pool_map(numbers):
    """
    Uses multiprocessing.Pool with map().
    
    Parameters:
    numbers (list): List of numbers.
    
    Returns:
    list: List of squared numbers.
    """
    with multiprocessing.Pool() as pool:
        result = pool.map(square, numbers)
    return result

def pool_apply(numbers):
    """
    Uses multiprocessing.Pool with apply() (synchronous).
    
    Parameters:
    numbers (list): List of numbers.
    
    Returns:
    list: List of squared numbers.
    """
    with multiprocessing.Pool() as pool:
        results = []
        for num in numbers:
            res = pool.apply(square, (num,))
            results.append(res)
    return results

def pool_apply_async(numbers):
    """
    Uses multiprocessing.Pool with apply_async() (asynchronous).
    
    Parameters:
    numbers (list): List of numbers.
    
    Returns:
    list: List of squared numbers.
    """
    with multiprocessing.Pool() as pool:
        async_results = [pool.apply_async(square, (num,)) for num in numbers]
        results = [res.get() for res in async_results]
    return results

def concurrent_futures_executor(numbers):
    """
    Uses concurrent.futures ProcessPoolExecutor.
    
    Parameters:
    numbers (list): List of numbers.
    
    Returns:
    list: List of squared numbers.
    """
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(square, numbers))
    return results
