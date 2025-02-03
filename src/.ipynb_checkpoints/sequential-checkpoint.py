import time
from src.tasks import join_random_letters, add_random_numbers

def run_sequential(start_letters=0, end_letters=1000000, start_numbers=0, end_numbers=1000000):
    start_time = time.time()
    
    # Generate letters and numbers using the specified ranges
    letters = join_random_letters(num=(end_letters - start_letters))
    numbers = add_random_numbers(num=(end_numbers - start_numbers))
    
    end_time = time.time()
    sequential_time = end_time - start_time
    
    print(f"Sequential Execution Time: {sequential_time:.6f} seconds")