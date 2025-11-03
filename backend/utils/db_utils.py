from psycopg2 import pool
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")


class DBUtils:
    _connection_pool = None
    
    @classmethod
    def initialize_pool(cls, minconn=1, maxconn=10):
        """Initialize connection pool once at app startup"""
        if cls._connection_pool is None:
            cls._connection_pool = pool.SimpleConnectionPool(
                minconn, maxconn,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME
            )
    
    @classmethod
    def get_connection(cls):
        """Get a connection from the pool"""
        return cls._connection_pool.getconn()
    
    @classmethod
    def return_connection(cls, connection):
        """Return connection back to pool"""
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