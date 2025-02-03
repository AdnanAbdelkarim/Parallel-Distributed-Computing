import multiprocessing
import time
from src.tasks import join_random_letters, add_random_numbers  # Import functions directly

def run_process(num_letters=1000000, num_numbers=1000000):
    start_time = time.time()
    
    # Create processes (1 for letters, 1 for numbers)
    process1 = multiprocessing.Process(target=join_random_letters, args=(num_letters,))
    process2 = multiprocessing.Process(target=add_random_numbers, args=(num_numbers,))
    
    # Start processes
    process1.start()
    process2.start()
    
    # Wait for processes to finish
    process1.join()
    process2.join()
    
    end_time = time.time()
    print(f"Processing Execution Time: {end_time - start_time:.6f} seconds")
    
    # Advanced: Two processes per function
    start_time = time.time()
    
    process3 = multiprocessing.Process(target=join_random_letters, args=(num_letters,))
    process4 = multiprocessing.Process(target=join_random_letters, args=(num_letters,))
    process5 = multiprocessing.Process(target=add_random_numbers, args=(num_numbers,))
    process6 = multiprocessing.Process(target=add_random_numbers, args=(num_numbers,))
    
    # Start all processes
    process3.start()
    process4.start()
    process5.start()
    process6.start()
    
    # Wait for all processes to finish
    process3.join()
    process4.join()
    process5.join()
    process6.join()
    
    end_time = time.time()
    print(f"Advanced Processing Execution Time: {end_time - start_time:.6f} seconds")
