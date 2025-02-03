import random
import string
# Function to join 1000 random letters
def join_random_letters(num = 1000):
    letters = [random.choice(string.ascii_letters) for _ in range(num)]
    return ''.join(letters)

# Function to add 1000 random numbers
def add_random_numbers(num = 1000):
    numbers = [random.randint(1, 100) for _ in range(num)]
    return sum(numbers)