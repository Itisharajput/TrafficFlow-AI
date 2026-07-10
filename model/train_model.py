"""
Trains a traffic congestion forecasting model.

Task: predict congestion_pct for each road, N hours ahead, using:
- time features (hour, day_of_week, is_weekend)
- lag features (congestion 1h ago, 2h ago, 24h ago / same time yesterday)
- rolling average (previous 3 hours)

Model: RandomForestRegressor (scikit-learn) - no internet/install needed,
handles nonlinearity and interactions well for this tabular time-series task.

Outputs:
- Trained model performance (MAE, R2)
- dashboard_data.json: everything the React dashboard needs to render
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "/home/claude/traffic-prediction-dashboard/data/traffic_data.csv"
OUT_PATH = "/home/claude/traffic-prediction-dashboard/model/dashboard_data.json"

FORECAST_HORIZON = 1  # predict 1 hour ahead (chain forward for multi-step at inference)


def build_features(df):
    df = df.sort_values(["road_id", "timestamp"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    feats = []
    for road_id, g in df.groupby("road_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        g["lag_1h"] = g["congestion_pct"].shift(1)
        g["lag_2h"] = g["congestion_pct"].shift(2)
        g["lag_24h"] = g["congestion_pct"].shift(24)
        g["roll_3h"] = g["congestion_pct"].shift(1).rolling(3).mean()
        g["target"] = g["congestion_pct"].shift(-FORECAST_HORIZON)
        feats.append(g)

    out = pd.concat(feats).dropna().reset_index(drop=True)
    return out


def main():
    df = pd.read_csv(DATA_PATH)
    feat_df = build_features(df)

    feature_cols = ["hour", "day_of_week", "is_weekend", "lag_1h", "lag_2h", "lag_24h", "roll_3h", "incident"]
    road_dummies = pd.get_dummies(feat_df["road_id"], prefix="road")
    X = pd.concat([feat_df[feature_cols], road_dummies], axis=1)
    y = feat_df["target"]

    # Time-based split: last 14 days = test set (no shuffling - respects time order)
    split_ts = feat_df["timestamp"].max() - pd.Timedelta(days=14)
    train_mask = feat_df["timestamp"] < split_ts
    test_mask = ~train_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    model = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Test MAE: {mae:.2f} percentage points")
    print(f"Test R^2: {r2:.3f}")

    feature_importance = sorted(
        zip(X.columns, model.feature_importances_), key=lambda x: -x[1]
    )
    print("\nTop features:")
    for name, imp in feature_importance[:6]:
        print(f"  {name}: {imp:.3f}")

    # ---- Build dashboard payload ----
    feat_df = feat_df.copy()
    feat_df["predicted"] = np.nan
    feat_df.loc[test_mask, "predicted"] = preds

    roads_meta = df[["road_id", "road_name", "capacity"]].drop_duplicates().to_dict("records")

    # Last 7 days of actual vs predicted, per road, for the chart
    recent_cutoff = feat_df["timestamp"].max() - pd.Timedelta(days=7)
    recent = feat_df[(feat_df["timestamp"] >= recent_cutoff) & test_mask]

    timeseries = {}
    for road_id, g in recent.groupby("road_id"):
        g = g.sort_values("timestamp")
        timeseries[road_id] = [
            {
                "ts": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
                "actual": round(row["congestion_pct"], 1),
                "predicted": round(row["predicted"], 1) if not np.isnan(row["predicted"]) else None,
                "incident": int(row["incident"]),
            }
            for _, row in g.iterrows()
        ]

    # Next 24h forecast per road (using last known real row + chained lag prediction)
    forecasts = {}
    for road_id, g in df.groupby("road_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        last_rows = g.tail(25).reset_index(drop=True)  # need history for lags
        history = list(last_rows["congestion_pct"])
        last_ts = pd.to_datetime(last_rows["timestamp"].iloc[-1])
        last_incident = 0

        future = []
        for step in range(1, 25):
            future_ts = last_ts + pd.Timedelta(hours=step)
            hour = future_ts.hour
            dow = future_ts.weekday()
            is_weekend = int(dow >= 5)
            lag_1h = history[-1]
            lag_2h = history[-2]
            lag_24h = history[-24] if len(history) >= 24 else history[0]
            roll_3h = np.mean(history[-3:])

            row = {c: 0 for c in X.columns}
            row.update({
                "hour": hour, "day_of_week": dow, "is_weekend": is_weekend,
                "lag_1h": lag_1h, "lag_2h": lag_2h, "lag_24h": lag_24h,
                "roll_3h": roll_3h, "incident": last_incident,
                f"road_{road_id}": 1
            })
            x_row = pd.DataFrame([row])[X.columns]
            pred = float(model.predict(x_row)[0])
            pred = max(0, min(100, pred))
            history.append(pred)

            future.append({"ts": future_ts.strftime("%Y-%m-%d %H:%M"), "predicted": round(pred, 1)})

        forecasts[road_id] = future

    payload = {
        "metrics": {"mae": round(mae, 2), "r2": round(r2, 3)},
        "roads": roads_meta,
        "timeseries": timeseries,
        "forecasts": forecasts,
        "feature_importance": [{"name": n, "importance": round(float(i), 3)} for n, i in feature_importance[:6]],
    }

    with open(OUT_PATH, "w") as f:
        json.dump(payload, f)
    print(f"\nDashboard payload written -> {OUT_PATH}")


if __name__ == "__main__":
    main()
