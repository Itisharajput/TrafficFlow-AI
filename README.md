# 🚦 TrafficFlow AI

An AI-powered traffic congestion prediction system built using Python, Scikit-learn, and React.

## 🌟 Features

- 🚗 Predicts traffic congestion 1–24 hours ahead
- 📊 Machine Learning model using Random Forest
- 📈 Achieved R² = 0.937 and MAE = 2.77
-  📊 Visualizes traffic prediction results
- 📁 Synthetic dataset with realistic traffic patterns

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Git & GitHub
# TrafficFlow AI — City Traffic Congestion Prediction Dashboard

Predicts road congestion 1–24 hours ahead for multiple city road segments using historical traffic patterns, so commuters can avoid jams and traffic police can plan deployment.

## Problem

Traffic congestion in Indian cities is largely predictable — rush hours, weekday vs weekend patterns, and recurring bottlenecks — but that predictability isn't used to help commuters or traffic police plan ahead. This project builds a forecasting pipeline and dashboard that turns historical traffic data into short-term predictions.

## What it does

- Forecasts congestion % for 5 road segments (highways, arterials, residential roads) using a Random Forest regression model
- Evaluates on a held-out 14-day time-based split (no data leakage)
- Achieves **R² = 0.937**, **MAE = 2.77 percentage points** on unseen data
- Surfaces which factors drive predictions (feature importance: 24h-ago congestion, recent trend, hour of day)
- Interactive dashboard: live-style "signal board" per road, 72h actual-vs-predicted chart, 24h forecast, and model transparency panel

## Tech stack

- **Data**: synthetic traffic generator modeling realistic rush-hour, weekday/weekend, and incident patterns (Python, NumPy, Pandas) — swappable for real sensor/GPS data
- **ML**: scikit-learn `RandomForestRegressor`, lag + rolling-window feature engineering, time-based train/test split
  

## Project structure

```
traffic-prediction-dashboard/
├── data/
│   ├── generate_data.py      # synthetic dataset generator
│   └── traffic_data.csv      # generated dataset (90 days, hourly, 5 roads)
├── model/
│   ├── train_model.py        # feature engineering + RandomForest training + eval
│   └── dashboard_data.json   # model output consumed by the dashboard
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
python data/generate_data.py     # generates data/traffic_data.csv
python model/train_model.py      # trains model, prints MAE/R², writes dashboard_data.json
```

## Swapping in real data

Replace `data/traffic_data.csv` with real traffic sensor / GPS / Google Maps Roads API data using the same columns (`timestamp, road_id, road_name, hour, day_of_week, is_weekend, volume, capacity, congestion_pct, incident`), then re-run `train_model.py`.

## Model approach

For each road segment, at each hour, the model predicts next-hour congestion using:
- Time features: hour of day, day of week, weekend flag
- Lag features: congestion 1h ago, 2h ago, same hour yesterday (24h ago)
- Rolling 3-hour average
- Incident flag

The dominant signal is same-time-yesterday congestion (46% importance) — confirming that Indian city traffic is highly repetitive day-to-day, which is exactly what makes short-term forecasting tractable with a relatively simple model.

## 🚀 Future Improvements

- Build a React dashboard for interactive visualization
- Integrate real traffic datasets
- Connect with Google Maps APIs
- Add live traffic prediction
- Add accident detection using computer vision (YOLO + OpenCV)
- Deploy the application online

## Author

Itisha Rajput — B.Sc. Computer Science, Pradhan Mantri College of Excellence, Badwani (DAVV Indore)
Built as a companion project to [AnnadataAI](https://github.com/Itisharajput/AnnadataAI)
