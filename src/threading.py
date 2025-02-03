import threading
import time
from src.tasks import join_random_letters, add_random_numbers

# Time the Threading Case with 2 threads per function
def run_threads(num_letters=1000000, num_numbers=1000000):
    start_time = time.time()

    # Two threads for letters
    thread1 = threading.Thread(target=join_random_letters, args=(num_letters,))
    thread2 = threading.Thread(target=join_random_letters, args=(num_letters,))

    # Two threads for numbers
    thread3 = threading.Thread(target=add_random_numbers, args=(num_numbers,))
    thread4 = threading.Thread(target=add_random_numbers, args=(num_numbers,))

    # Start all threads
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()

    # Wait for all threads to finish
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()

    end_time = time.time()
    threading_time = end_time - start_time
    print(f"Threading Execution Time: {threading_time:.6f} seconds")





