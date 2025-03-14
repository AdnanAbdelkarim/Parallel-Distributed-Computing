# src/tasks.py

import random

def square(n):
    """
    Computes the square of an integer.
    
    Parameters:
    n (int): The number to square.
    
    Returns:
    int: The squared result.
    """
    return n * n

def generate_numbers(count):
    """
    Generates a list of random integers between 1 and 100.
    
    Parameters:
    count (int): Number of random integers to generate.
    
    Returns:
    list: List of random integers.
    """
    return [random.randint(1, 100) for _ in range(count)]
