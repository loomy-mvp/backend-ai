from psycopg2 import pool
from dotenv import load_dotenv
import os
import time

# Load environment variables from .env
load_dotenv()
DB_USER = os.getenv("DB_USER_POOL")
DB_PASSWORD = os.getenv("DB_PASSWORD_POOL")
DB_HOST = os.getenv("DB_HOST_POOL")
DB_PORT = os.getenv("DB_PORT_POOL")
DB_NAME = os.getenv("DB_NAME_POOL")


class DBUtils:
    _connection_pool = None
    
    @classmethod
    def initialize_pool(cls, minconn=1, maxconn=10, max_retries=3, retry_delay=2):
        """
        Initialize connection pool with retry logic.
        
        Args:
            minconn: Minimum number of connections in the pool
            maxconn: Maximum number of connections in the pool
            max_retries: Maximum number of connection attempts
            retry_delay: Delay in seconds between retries
        """
        if cls._connection_pool is not None:
            print("[DBUtils] Connection pool already initialized")
            return
        
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[DBUtils] Attempting to connect to database (attempt {attempt}/{max_retries})...")
                cls._connection_pool = pool.SimpleConnectionPool(
                    minconn, maxconn,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    host=DB_HOST,
                    port=DB_PORT,
                    dbname=DB_NAME,
                    # Add connection timeout to fail faster
                    connect_timeout=10
                )
                print(f"[DBUtils] Successfully connected to database on attempt {attempt}")
                return
            except Exception as e:
                last_error = e
                print(f"[DBUtils] Connection attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    print(f"[DBUtils] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        
        # If all retries failed, raise the last error
        raise Exception(f"Failed to initialize database pool after {max_retries} attempts: {last_error}")
    
    @classmethod
    def get_connection(cls):
        """Get a connection from the pool"""
        if cls._connection_pool is None:
            raise Exception("Database connection pool not initialized. Call initialize_pool() first.")
        return cls._connection_pool.getconn()
    
    @classmethod
    def return_connection(cls, connection):
        """Return connection back to pool"""
        if cls._connection_pool is None:
            raise Exception("Database connection pool not initialized.")
        cls._connection_pool.putconn(connection)
    
    @classmethod
    def execute_query(cls, query, params=None) -> list:
        """Execute query and return results"""
        connection = cls.get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                # connection.commit() only needed for INSERT/UPDATE/DELETE
                return cursor.fetchall()
        finally:
            cls.return_connection(connection)
    
    @classmethod
    def close_pool(cls):
        """Close all connections in the pool"""
        if cls._connection_pool is not None:
            cls._connection_pool.closeall()
            cls._connection_pool = None
            print("[DBUtils] Connection pool closed")