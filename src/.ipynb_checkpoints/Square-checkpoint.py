import multiprocessing
import random
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from multiprocessing import Process, Queue

def square(x):
    return x ** 2
def SequentialTest(y):
    start_time = time.time()
    results_seq = [square(x) for x in y]
    end_time = time.time()
    print(f"Sequential processing time: {end_time - start_time} seconds")
def ProcessForEachNumTest(y):
    print("CAUTION THIS WILL TAKE A LONG TIME")
    def square_worker(number, queue):
        queue.put(square(number))
    start_time = time.time()
    queue = Queue()
    processes = [Process(target=square_worker, args=(x, queue)) for x in y]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    results_multi = [queue.get() for _ in y]
    end_time = time.time()
    print(f"Multiprocessing (one process per number) time: {end_time - start_time} seconds")
def MapTest(y):
    start_time = time.time()
    with Pool() as pool:
        results_map = pool.map(square, y)
    end_time = time.time()
    print(f"Multiprocessing pool (map) time: {end_time - start_time} seconds")
def MapAsyncTest(y):
    start_time = time.time()
    with Pool(processes=4) as pool:
        async_result = pool.map_async(square, y)
        results = async_result.get()
    end_time = time.time()
    print(f"Asynchronous processing time: {end_time - start_time} seconds")
def ApplyTest(y):
    start_time = time.time()
    with Pool() as pool:
        results_apply = [pool.apply(square, (x,)) for x in y]
    end_time = time.time()
    print(f"Multiprocessing pool (apply) time: {end_time - start_time} seconds")
def ProcessPoolExecutorTest(y):
    start_time = time.time()
    with ProcessPoolExecutor() as executor:
        results_futures = list(executor.map(square, y))
    end_time = time.time()
    print(f"ProcessPoolExecutor time: {end_time - start_time} seconds")
numbers = [random.randint(1, 100) for _ in range(10 ** 6)]
numbers_large = [random.randint(1, 100) for _ in range(10 ** 7)]

print("For 10^6")
SequentialTest(numbers)
ProcessForEachNumTest(numbers)
MapTest(numbers)
MapAsyncTest(numbers)
ApplyTest(numbers)
ProcessPoolExecutorTest(numbers)
print("For 10^7")
SequentialTest(numbers_large)
ProcessForEachNumTest(numbers_large)
MapTest(numbers_large)
MapAsyncTest(numbers_large)
ApplyTest(numbers_large)
ProcessPoolExecutorTest(numbers_large)

class ConnectionPool:
    def __init__(self, max_connections):
        self.connections = list(range(max_connections))
        self.semaphore = multiprocessing.Semaphore(max_connections)

    def get_connection(self):
        self.semaphore.acquire()
        connection = self.connections.pop()
        print(f"Process {multiprocessing.current_process().name} acquired connection: {connection}")
        return connection

    def release_connection(self, connection):
        self.connections.append(connection)
        print(f"Process {multiprocessing.current_process().name} released connection: {connection}")
        self.semaphore.release()
def access_database(pool):
    connection = pool.get_connection()
    time.sleep(random.uniform(0.1, 0.5))
    pool.release_connection(connection)
if __name__ == "__main__":
    pool = ConnectionPool(max_connections=3)
    processes = []
    for i in range(10):
        p = multiprocessing.Process(target=access_database, args=(pool,), name=f"Process-{i + 1}")
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print("All processes have completed.")
    print("Assignment 1 Part 1 Ends Here")