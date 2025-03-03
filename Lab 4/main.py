import time
from src.threads import start_threads
from src.tasks import initialize_display

if __name__ == "__main__":
    num_sensors = 5  # Adjust number of sensors as needed
    initialize_display(num_sensors)
    
    start_threads(num_sensors)

    try:
        while True:
            time.sleep(1)  # Keep the main thread running
    except KeyboardInterrupt:
        print("\nStopping temperature monitoring...")
