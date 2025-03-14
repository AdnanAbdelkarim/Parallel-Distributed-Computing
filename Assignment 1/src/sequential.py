from src.tasks import square

def sequential_square(numbers):
    """
    Computes squares of numbers sequentially.
    """
    return [square(num) for num in numbers]