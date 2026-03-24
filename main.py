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
    model = joblib.load(MODEL_PATH)
    print(f"✅ Zebox model loaded from {MODEL_PATH}")
    await get_pool()
    print("✅ Database pool created")
    yield
    await close_pool()
    print("🛑 Shutdown complete")


app = FastAPI(title="Biomoneta Backend", lifespan=lifespan)

# Allow the Next.js frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    try:
        print(f"📥 Received {device_type} data for {room_id}:")
        print(f"   CO2={para_i}, T={para_ii}, H={para_iii}")
        print(f"   PM1={para_v}, PM2.5={para_vi}, PM4={para_vii}, PM10={para_viii}")
        print(f"   VOC={para_ix}, NOx={para_x}")
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
            # Determine which reading is IN and which is OUT
            if device_type == "IN":
                in_data = {
                    "co2": para_i, "temp": para_ii, "hum": para_iii,
                    "pm1": para_v, "pm4": para_vii,
                    "voc": para_ix, "nox": para_x,
                }
                out_data = {
                    "pm1": opposite_row["pm1_0"], "pm4": opposite_row["pm4_0"],
                }
            else:
                in_data = {
                    "co2": opposite_row["co2"], "temp": opposite_row["temperature"],
                    "hum": opposite_row["humidity"],
                    "pm1": opposite_row["pm1_0"], "pm4": opposite_row["pm4_0"],
                    "voc": opposite_row["voc_index"], "nox": opposite_row["nox_index"],
                }
                out_data = {"pm1": para_v, "pm4": para_vii}

            # Calculate bacteria_mass = (pm4_in - pm1_in) - (pm4_out - pm1_out)
            bacteria_mass = (in_data["pm4"] - in_data["pm1"]) - (out_data["pm4"] - out_data["pm1"])

            # Model expects: [CO2, Temperature, Humidity, Bacteria_Mass, VOC, NOx]
            features = np.array([[
                in_data["co2"], in_data["temp"], in_data["hum"],
                bacteria_mass,
                in_data["voc"], in_data["nox"],
            ]])

            if model:
                bacteria_count = float(model.predict(features)[0])

                # Update the most recent IN reading with the prediction
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
    except Exception as e:
        print(f"❌ Error during ingestion: {str(e)}")
        return {
            "success": False,
            "message": f"Server Error: {str(e)}"
        }


# ---------------------------------------------------------------------------
# GET /api/rooms/{room_id}/history  — dashboard data
# ---------------------------------------------------------------------------
@app.get("/api/rooms/{room_id}/history")
async def get_room_history(room_id: str):
    """Fetch last 24 hours of data aggregated by 5-minute intervals."""
    pool = await get_pool()

    query = """
        SELECT
            time_bucket('5 minutes', time) AS bucket,
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
            "value": round(row["pm2_5"] or 0),
            "bacteria_count": round(row["bacteria_count"] or 0, 2) if row["bacteria_count"] else None,
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
