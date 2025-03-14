import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from src.tasks import square

def multiprocessing_per_process(numbers):
    processes = []
    for num in numbers:
        p = multiprocessing.Process(target=square, args=(num,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

def pool_map(numbers):
    with multiprocessing.Pool() as pool:
        pool.map(square, numbers)

def pool_apply(numbers):
    with multiprocessing.Pool() as pool:
        for num in numbers:
            pool.apply(square, args=(num,))

def pool_apply_async(numbers):
    with multiprocessing.Pool() as pool:
        results = [pool.apply_async(square, args=(num,)) for num in numbers]
        [r.get() for r in results]

def concurrent_futures_executor(numbers):
    with ProcessPoolExecutor() as executor:
        executor.map(square, numbers)