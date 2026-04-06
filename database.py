import os
import asyncpg
import asyncio
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


async def create_pool_with_retry() -> asyncpg.Pool:
    """Create DB pool with retry logic (critical for Docker startup)."""
    retries = 10
    delay = 2

    for attempt in range(retries):
        try:
            print(f"🔄 Connecting to DB (attempt {attempt + 1})...")
            pool = await asyncpg.create_pool(
                **DATABASE_CONFIG,
                min_size=1,
                max_size=10,
                command_timeout=60,
            )
            print("✅ Database pool created")
            return pool

        except Exception as e:
            print(f"❌ DB connection failed: {e}")
            if attempt < retries - 1:
                print(f"⏳ Retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                raise Exception("🚨 Could not connect to DB after multiple attempts")


async def get_pool() -> asyncpg.Pool:
    """Get or recreate pool safely."""
    global pool

    if pool is None:
        pool = await create_pool_with_retry()

    return pool


async def close_pool():
    """Close the connection pool gracefully."""
    global pool
    if pool:
        await pool.close()
        pool = None