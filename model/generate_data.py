"""
Generates synthetic but realistic traffic volume data for multiple road segments
in a simulated city, over 90 days, at hourly resolution.

Patterns modeled:
- Morning rush (8-10 AM) and evening rush (5-8 PM) on weekdays
- Lower, flatter traffic on weekends
- Random incidents (accidents/roadwork) that spike congestion for a few hours
- Road-specific baseline traffic (highways busier than residential roads)
- Gaussian noise for realism
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

ROADS = [
    {"id": "R1", "name": "MG Road (Highway)", "base_volume": 800, "capacity": 1200},
    {"id": "R2", "name": "Station Road (Arterial)", "base_volume": 500, "capacity": 800},
    {"id": "R3", "name": "College Chowk (Residential)", "base_volume": 200, "capacity": 400},
    {"id": "R4", "name": "Ring Road (Highway)", "base_volume": 900, "capacity": 1400},
    {"id": "R5", "name": "Market Square (Arterial)", "base_volume": 450, "capacity": 700},
]

DAYS = 90
START_DATE = datetime(2026, 4, 1)


def hourly_multiplier(hour, is_weekend):
    """Traffic volume multiplier based on hour of day."""
    if is_weekend:
        # Flatter curve, mild afternoon peak
        base = 0.4 + 0.3 * np.exp(-((hour - 14) ** 2) / 30)
    else:
        # Morning peak ~9AM, evening peak ~18:30
        morning = 0.9 * np.exp(-((hour - 9) ** 2) / 4)
        evening = 1.0 * np.exp(-((hour - 18.5) ** 2) / 5)
        base = 0.25 + morning + evening
    return base


def generate():
    rows = []
    # Pre-schedule random incidents: (road_id, start_datetime, duration_hours, severity)
    incidents = []
    for _ in range(35):
        road = np.random.choice(ROADS)["id"]
        day_offset = np.random.randint(0, DAYS)
        hour = np.random.randint(6, 22)
        start = START_DATE + timedelta(days=day_offset, hours=hour)
        duration = np.random.randint(1, 4)
        severity = np.random.uniform(1.3, 2.2)
        incidents.append((road, start, duration, severity))

    for day in range(DAYS):
        date = START_DATE + timedelta(days=day)
        is_weekend = date.weekday() >= 5
        for hour in range(24):
            ts = date + timedelta(hours=hour)
            for road in ROADS:
                mult = hourly_multiplier(hour, is_weekend)
                volume = road["base_volume"] * mult
                volume *= np.random.normal(1.0, 0.08)  # noise

                # Apply any active incident
                incident_active = 0
                for r_id, start, dur, sev in incidents:
                    if r_id == road["id"] and start <= ts < start + timedelta(hours=dur):
                        volume *= sev
                        incident_active = 1
                        break

                volume = max(20, volume)
                congestion_pct = min(100, (volume / road["capacity"]) * 100)

                rows.append({
                    "timestamp": ts,
                    "road_id": road["id"],
                    "road_name": road["name"],
                    "hour": hour,
                    "day_of_week": date.weekday(),
                    "is_weekend": int(is_weekend),
                    "volume": round(volume, 1),
                    "capacity": road["capacity"],
                    "congestion_pct": round(congestion_pct, 1),
                    "incident": incident_active,
                })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "/home/claude/traffic-prediction-dashboard/data/traffic_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df.head(10))
    print("\nRoads:", [r["name"] for r in ROADS])
