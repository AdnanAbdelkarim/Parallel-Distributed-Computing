import random

# Function to sort a list of random numbers
def sort_numbers(num=1000):
    numbers = [random.randint(1, 100) for _ in range(num)]
    return sorted(numbers)

