import os
import asyncpg
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Configuration from environment with robust defaults
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5433")),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "database": os.getenv("DB_NAME", "biomoneta_air"),
}

pool: asyncpg.Pool | None = None


async def create_pool_with_retry() -> asyncpg.Pool:
    """Create DB pool with exponential backoff and connectivity validation."""
    retries = 15
    base_delay = 2
    
    # Identify target host (masked password)
    target = f"{DATABASE_CONFIG['user']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
    print(f"🔄 Initializing DB pool for: {target}")

    for attempt in range(retries):
        try:
            # Create a pool with improved multi-user connection handling
            new_pool = await asyncpg.create_pool(
                **DATABASE_CONFIG,
                min_size=2,          # Keep some connections warm
                max_size=10,
                max_inactive_connection_lifetime=300, # Clean stale connections after 5m
                command_timeout=60,
            )
            
            # Connection fingerprinting: verify we hit the RIGHT database
            async with new_pool.acquire() as conn:
                version = await conn.fetchval("SELECT version();")
                print(f"✅ DB Connected. Version: {version}")
                if "TimescaleDB" not in version:
                    print("⚠️  Alert: This instance does not report TimescaleDB hypertable engine.")

            print("✅ Database pool created and validated")
            return new_pool

        except asyncpg.exceptions.InvalidPasswordError:
            print(f"❌ AUTH ERROR: Invalid password for user '{DATABASE_CONFIG['user']}' at {target}")
            # We still retry because sometimes Docker DNS resolves to the wrong container briefly on startup
        except Exception as e:
            print(f"❌ Connection Attempt {attempt + 1} failed: {str(e)}")

        # Exponential backoff: 2s, 4s, 8s... up to 30s
        delay = min(base_delay * (2 ** attempt), 30)
        if attempt < retries - 1:
            print(f"⏳ Retrying in {delay}s...")
            await asyncio.sleep(delay)
        else:
            raise Exception(f"🚨 Failed to establish a healthy DB connection to {target} after {retries} attempts.")


async def get_pool() -> asyncpg.Pool:
    """Get active pool or recreate if it was closed or corrupted."""
    global pool

    # If pool exists but is in a closed or unusable state, reset it
    if pool is not None:
        # Check internal _closed flag if available
        if getattr(pool, "_closed", False):
            print("🔄 Detected closed database pool, resetting...")
            pool = None

    if pool is None:
        pool = await create_pool_with_retry()

    return pool


async def close_pool():
    """Close the connection pool gracefully."""
    global pool
    if pool:
        print("🛑 Gracefully closing database pool...")
        await pool.close()
        pool = None