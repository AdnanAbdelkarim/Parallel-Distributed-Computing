import time
import random
import sys
from threading import RLock, Condition

# Global dictionaries
latest_temperatures = {}
temperature_averages = {}
lock = RLock()
condition = Condition(lock)

def simulate_sensor(sensor_id):
    """Simulate temperature readings from a sensor."""
    while True:
        temperature = random.randint(15, 40)
        with lock:
            latest_temperatures[sensor_id] = temperature
        with condition:
            condition.notify_all()  # Notify processing thread
        time.sleep(1)


def process_temperatures():
    """Continuously calculate the average temperature."""
    temp_data = {}
    count_data = {}
    
    while True:
        with condition:
            condition.wait()  # Wait for sensor updates before recalculating averages
            
            for sensor_id, temperature in latest_temperatures.items():
                temp_data[sensor_id] = temp_data.get(sensor_id, 0) + temperature
                count_data[sensor_id] = count_data.get(sensor_id, 0) + 1
                temperature_averages[sensor_id] = temp_data[sensor_id] / count_data[sensor_id]


def initialize_display(num_sensors):
    """Print the initial layout for displaying temperatures."""
    print("Current temperatures:")
    print("Latest Temperatures:", end=" ")
    for i in range(num_sensors):
        print(f"Sensor {i}: --°C", end=" ")
    print()
    for i in range(num_sensors):
        print(f"Sensor {i} Average: --°C")
    print()

import sys
import time
from threading import RLock

# Global dictionaries
latest_temperatures = {}
temperature_averages = {}
lock = RLock()

def update_display(latest_only=True):
    """Refresh latest temperatures every 1s and averages every 5s without erasing the console."""
    sys.stdout.write("\033[H\033[J")  # Clear console
    sys.stdout.flush()
    
    # Display latest temperatures
    print("\nLatest Temperatures:")
    with lock:
        for i in range(len(latest_temperatures)):
            temp = latest_temperatures.get(i, "--")
            print(f"Sensor {i}: {temp}°C")

    # Only update averages when `latest_only` is False (every 5s)
    #if not latest_only:
    print("\nAverage Temperatures:")
    with lock:
        for i in range(len(temperature_averages)):
            avg_temp = temperature_averages.get(i, "--")
            print(f"Sensor {i} Average: {avg_temp:.2f}°C")
