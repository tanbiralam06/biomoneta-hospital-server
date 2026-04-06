import os
import joblib
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from database import get_pool, close_pool

# ---------------------------------------------------------------------------
# Model loading at startup
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "zebox_best_model.pkl")
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model and DB pool on startup; clean up on shutdown."""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Zebox model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️  ML Model load failed: {e}. Some features may be disabled.")

    # Critical: Wait for DB to be truly ready before starting
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("SELECT 1")
        print("✅ Database pool created and verified")
    except Exception as e:
        print(f"🚨 CRITICAL: Database startup failed: {e}")
        # We don't crash the server here to allow health checks to report failure
        
    yield
    await close_pool()
    print("🛑 Shutdown complete")


app = FastAPI(title="Biomoneta Backend", lifespan=lifespan)

# Allow the Next.js frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check for Docker/VPS monitoring."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unhealthy: {str(e)}")


# ---------------------------------------------------------------------------
# Sensor Ingest  — handles GET (Microcontroller) and POST (Testing Tools)
# ---------------------------------------------------------------------------
@app.get("/api/sensors/ingest")
@app.post("/api/sensors/ingest")
async def ingest_sensor_data(
    room_id: str = "room_001",
    device_type: str = "IN",
    para_i: float = 0,
    para_ii: float = 0,
    para_iii: float = 0,
    para_v: float = 0,
    para_vi: float = 0,
    para_vii: float = 0,
    para_viii: float = 0,
    para_ix: float = 0,
    para_x: float = 0,
):
    max_retries = 2
    for attempt in range(max_retries):
        try:
            print(f"📥 [{attempt+1}] Receiving {device_type} data for {room_id}...")
            pool = await get_pool()

            # 1. Store the raw reading
            insert_query = """
                INSERT INTO readings (
                    time, room_id, device_type, co2, temperature, humidity,
                    pm1_0, pm2_5, pm4_0, pm10_0, voc_index, nox_index
                ) VALUES (
                    NOW(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
                )
            """
            await pool.execute(
                insert_query,
                room_id, device_type,
                para_i, para_ii, para_iii,
                para_v, para_vi, para_vii, para_viii,
                para_ix, para_x,
            )

            # 2. Attempt IN/OUT correlation for bacteria prediction
            opposite_type = "OUT" if device_type == "IN" else "IN"
            correlation_query = """
                SELECT co2, temperature, humidity, pm1_0, pm4_0, voc_index, nox_index
                FROM readings
                WHERE room_id = $1
                  AND device_type = $2
                  AND time > NOW() - INTERVAL '5 minutes'
                ORDER BY time DESC
                LIMIT 1
            """
            opposite_row = await pool.fetchrow(correlation_query, room_id, opposite_type)

            bacteria_count = None
            if opposite_row is not None:
                # Correlation logic (same as before)
                if device_type == "IN":
                    in_data = {
                        "co2": para_i, "temp": para_ii, "hum": para_iii,
                        "pm1": para_v, "pm4": para_vii,
                        "voc": para_ix, "nox": para_x,
                    }
                    out_data = {"pm1": opposite_row["pm1_0"], "pm4": opposite_row["pm4_0"]}
                else:
                    in_data = {
                        "co2": opposite_row["co2"], "temp": opposite_row["temperature"],
                        "hum": opposite_row["humidity"],
                        "pm1": opposite_row["pm1_0"], "pm4": opposite_row["pm4_0"],
                        "voc": opposite_row["voc_index"], "nox": opposite_row["nox_index"],
                    }
                    out_data = {"pm1": para_v, "pm4": para_vii}

                bacteria_mass = (in_data["pm4"] - in_data["pm1"]) - (out_data["pm4"] - out_data["pm1"])
                features = np.array([[
                    in_data["co2"], in_data["temp"], in_data["hum"],
                    bacteria_mass, in_data["voc"], in_data["nox"],
                ]])

                if model:
                    bacteria_count = float(model.predict(features)[0])
                    update_query = """
                        UPDATE readings
                        SET bacteria_count = $1
                        WHERE ctid = (
                            SELECT ctid FROM readings
                            WHERE room_id = $2 AND device_type = 'IN'
                              AND time > NOW() - INTERVAL '5 minutes'
                            ORDER BY time DESC
                            LIMIT 1
                        )
                    """
                    await pool.execute(update_query, bacteria_count, room_id)

            return {
                "success": True,
                "message": "Data logged successfully",
                "bacteria_count": bacteria_count,
            }

        except (asyncpg.exceptions.PostgresError, OSError) as e:
            print(f"⚠️  Database connection issue during ingestion (attempt {attempt+1}): {str(e)}")
            # On first failure, try to reset the pool
            if attempt == 0:
                await close_pool()
                await asyncio.sleep(1)
            else:
                raise HTTPException(status_code=503, detail="Database currently unavailable")
        except Exception as e:
            print(f"❌ Unexpected Error: {str(e)}")
            return {"success": False, "message": f"Server Error: {str(e)}"}



# ---------------------------------------------------------------------------
# GET /api/rooms/{room_id}/history  — dashboard data
# ---------------------------------------------------------------------------
@app.get("/api/rooms/{room_id}/history")
async def get_room_history(room_id: str):
    """Fetch last 24 hours of data aggregated by 5-minute intervals."""
    pool = await get_pool()

    query = """
        SELECT
            time_bucket('5 minutes', time) AT TIME ZONE 'Asia/Kolkata' AS bucket,
            AVG(co2) AS co2,
            AVG(temperature) AS temperature,
            AVG(humidity) AS humidity,
            AVG(pm1_0) AS pm1_0,
            AVG(pm2_5) AS pm2_5,
            AVG(pm4_0) AS pm4_0,
            AVG(pm10_0) AS pm10_0,
            AVG(voc_index) AS voc_index,
            AVG(nox_index) AS nox_index,
            AVG(bacteria_count) AS bacteria_count
        FROM readings
        WHERE room_id = $1
          AND device_type = 'IN'
          AND time > NOW() - INTERVAL '24 hours'
        GROUP BY bucket
        ORDER BY bucket ASC
    """

    rows = await pool.fetch(query, room_id)

    formatted = [
        {
            "time": row["bucket"].strftime("%-I:%M %p"),
            "value": round(row["bacteria_count"] or 0, 2) if row["bacteria_count"] is not None else round(row["pm2_5"] or 0),
            "bacteria_count": round(row["bacteria_count"] or 0, 2) if row["bacteria_count"] is not None else None,
            "fullData": {
                "co2": row["co2"],
                "temperature": row["temperature"],
                "humidity": row["humidity"],
                "pm1_0": row["pm1_0"],
                "pm2_5": row["pm2_5"],
                "pm4_0": row["pm4_0"],
                "pm10_0": row["pm10_0"],
                "voc_index": row["voc_index"],
                "nox_index": row["nox_index"],
                "bacteria_count": row["bacteria_count"],
            },
        }
        for row in rows
    ]

    return formatted
