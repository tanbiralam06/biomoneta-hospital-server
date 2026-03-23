import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5433")),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "database": os.getenv("DB_NAME", "biomoneta_air"),
}

pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Return the existing connection pool or create a new one."""
    global pool
    if pool is None:
        pool = await asyncpg.create_pool(**DATABASE_CONFIG)
    return pool


async def close_pool():
    """Close the connection pool gracefully."""
    global pool
    if pool:
        await pool.close()
        pool = None
