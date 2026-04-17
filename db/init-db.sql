-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Hospitals table
CREATE TABLE IF NOT EXISTS hospitals (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    location TEXT
);

-- Rooms table
CREATE TABLE IF NOT EXISTS rooms (
    id TEXT PRIMARY KEY,
    hospital_id TEXT REFERENCES hospitals(id),
    name TEXT NOT NULL
);

-- Sensor readings table (Time-series)
CREATE TABLE IF NOT EXISTS readings (
    time TIMESTAMPTZ NOT NULL,
    room_id TEXT REFERENCES rooms(id),
    device_type TEXT,                -- 'IN' or 'OUT'
    
    co2 DOUBLE PRECISION,           -- para_i
    temperature DOUBLE PRECISION,   -- para_ii
    humidity DOUBLE PRECISION,      -- para_iii
    
    pm1_0 DOUBLE PRECISION,         -- para_v
    pm2_5 DOUBLE PRECISION,         -- para_vi
    pm4_0 DOUBLE PRECISION,         -- para_vii
    pm10_0 DOUBLE PRECISION,        -- para_viii
    
    voc_index DOUBLE PRECISION,     -- para_ix
    nox_index DOUBLE PRECISION,     -- para_x
    tvoc_ppb DOUBLE PRECISION,      -- new sensor
    tvoc_ppm DOUBLE PRECISION,      -- new sensor

    bacteria_count DOUBLE PRECISION -- predicted CFU/m³ from ML model
);

-- Convert readings table to a hypertable
SELECT create_hypertable('readings', 'time', if_not_exists => TRUE);

-- Create index for faster querying by room and time
CREATE INDEX IF NOT EXISTS idx_readings_room_time ON readings (room_id, time DESC);

-- Seed initial single hospital and room
INSERT INTO hospitals (id, name) VALUES 
('hosp_001', 'Primary Hospital')
ON CONFLICT (id) DO NOTHING;

INSERT INTO rooms (id, hospital_id, name) VALUES 
('room_001', 'hosp_001', 'Room 001'),
('room_002', 'hosp_001', 'Room 002')
ON CONFLICT (id) DO NOTHING;
