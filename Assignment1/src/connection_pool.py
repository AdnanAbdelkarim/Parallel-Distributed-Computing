import time
import random
class ConnectionPool:
    """
    Simulates a pool of database connections using a semaphore to limit access.
    """
    def __init__(self, max_connections, connections, semaphore):
        """
        Initialize the ConnectionPool.

        Parameters:
        max_connections (int): Maximum number of simultaneous connections.
        connections (list): Shared list of connections.
        semaphore (multiprocessing.Semaphore): Shared semaphore.
        """
        self.max_connections = max_connections
        self.connections = connections
        self.semaphore = semaphore

    def get_connection(self):
        """
        Acquire a connection from the pool.

        Returns:
        str: The acquired connection.
        """
        self.semaphore.acquire()
        connection = self.connections.pop()
        return connection

    def release_connection(self, connection):
        """
        Release a connection back to the pool.

        Parameters:
        connection (str): The connection to release.
        """
        self.connections.append(connection)
        self.semaphore.release()

def access_database(pool, process_id):
    """
    Simulates a process performing a database operation.

    Parameters:
    pool (ConnectionPool): The shared connection pool.
    process_id (int): Unique identifier for the process.
    """
    print(f"Process-{process_id} is waiting for a connection...")

    connection = pool.get_connection()
    print(f"Process-{process_id} acquired {connection}")

    time.sleep(random.uniform(1, 3))

    pool.release_connection(connection)
    print(f"Process-{process_id} released {connection}")
