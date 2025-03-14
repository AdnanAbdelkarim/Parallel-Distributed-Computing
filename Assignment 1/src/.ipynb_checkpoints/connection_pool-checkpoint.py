import time
import random

class ConnectionPool:
    def __init__(self, max_connections, shared_connections=None, shared_semaphore=None):
        self.max_connections = max_connections
        self.connections = shared_connections or [f"Connection-{i+1}" for i in range(max_connections)]
        self.semaphore = shared_semaphore or multiprocessing.Semaphore(max_connections)

    def get_connection(self):
        self.semaphore.acquire()
        if self.connections:
            return self.connections.pop()
        return None

    def release_connection(self, connection):
        self.connections.append(connection)
        self.semaphore.release()

def access_database(pool, process_id):
    print(f"Process-{process_id} is waiting for a connection...")
    connection = pool.get_connection()
    if connection:
        print(f"Process-{process_id} acquired {connection}")
        time.sleep(random.uniform(0.5, 2))
        print(f"Process-{process_id} released {connection}")
        pool.release_connection(connection)
    else:
        print(f"Process-{process_id} failed to acquire a connection")