# src/sequential.py

from src.tasks import square

def sequential_square(numbers):
    """
    Squares numbers sequentially using a simple for loop.
    
    Parameters:
    numbers (list): List of numbers.
    
    Returns:
    list: List of squared numbers.
    """
    result = []
    for num in numbers:
        result.append(square(num))
    return result
