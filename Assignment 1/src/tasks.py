import random

def square(number):
    """
    Returns the square of a number.
    """
    return number * number

def generate_numbers(count):
    """
    Generates a list of random numbers.
    """
    return [random.randint(1, 100) for _ in range(count)]