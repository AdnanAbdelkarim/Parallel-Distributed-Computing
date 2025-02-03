from src.sequential import run_sequential
from src.threading import run_threads
from src.processing import run_process
from src.performance import compute_performance
print("Running Sequential Execution...")
run_sequential()

print("----------------------------------------")
print("Running Threading Execution...")
run_threads()

print("----------------------------------------")
print("Running Processing Execution...")
run_process()

print("----------------------------------------")
print("Performance Metrics:")
compute_performance()
