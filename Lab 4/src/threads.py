import threading
import time
from src.tasks import simulate_sensor, process_temperatures, update_display

def start_threads(num_sensors):
    """Start sensor, processing, and display update threads."""
    threads = []

    # Start sensor threads
    for sensor_id in range(num_sensors):
        thread = threading.Thread(target=simulate_sensor, args=(sensor_id,), daemon=True)
        threads.append(thread)
        thread.start()

    # Start processing thread
    process_thread = threading.Thread(target=process_temperatures, daemon=True)
    process_thread.start()

    # Start display update threads (1s for latest, 5s for averages)
    latest_temp_thread = threading.Thread(target=latest_temp_loop, daemon=True)
    avg_temp_thread = threading.Thread(target=avg_temp_loop, daemon=True)

    latest_temp_thread.start()
    avg_temp_thread.start()

    threads.extend([process_thread, latest_temp_thread, avg_temp_thread])
    
    return threads

def latest_temp_loop():
    """Update latest temperatures every 1 second."""
    while True:
        update_display(latest_only=True)  # Only updates latest temperatures
        time.sleep(1)

def avg_temp_loop():
    """Update average temperatures every 5 seconds."""
    while True:
        update_display(latest_only=False)  # Updates both latest and averages
        time.sleep(5)
