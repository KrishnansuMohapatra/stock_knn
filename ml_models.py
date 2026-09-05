"""
Next-session close models.

Training target is Close[t+1]. Features are values known at the close of day t.
A persistence baseline (tomorrow ≈ today) is always reported so skill is honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from config import MIN_TRAIN_ROWS


def _metrics(y_true, y_pred, close_t):
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    actual_dir = np.sign(np.asarray(y_true) - np.asarray(close_t))
    pred_dir = np.sign(np.asarray(y_pred) - np.asarray(close_t))
    mask = actual_dir != 0
    if mask.sum() == 0:
        da = float("nan")
    else:
        da = float((actual_dir[mask] == pred_dir[mask]).mean() * 100)
    return rmse, mape, da


def train_models(data: pd.DataFrame, features: list[str], k: int = 5) -> dict:
    missing = [f for f in features if f not in data.columns]
    if missing:
        return {"error": f"Missing features: {', '.join(missing)}"}

    if "Target_Close" not in data.columns:
        return {"error": "Target_Close missing — call add_ml_features first."}

    cols = list(dict.fromkeys(list(features) + ["Target_Close", "Close"]))
    df = data.loc[:, cols].copy()
    df = df.dropna()
    n = len(df)
    if n < MIN_TRAIN_ROWS:
        return {
            "error": (
                f"Not enough complete rows to train ({n} < {MIN_TRAIN_ROWS}). "
                "Widen the date range."
            )
        }

    X = df[features]
    y = df["Target_Close"]
    close_t = df["Close"]

    split = int(n * 0.8)
    min_train = max(k, 20)
    if split < min_train or (n - split) < 5:
        return {
            "error": (
                f"Train/test split too small (n={n}, train={split}). "
                "Widen the date range."
            )
        }

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    close_test = close_t.iloc[split:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    k_eff = max(1, min(int(k), len(X_train)))
    models = {
        "KNN": KNeighborsRegressor(n_neighbors=k_eff),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=80, random_state=42, n_jobs=1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=80, random_state=42
        ),
    }

    predictions: dict[str, np.ndarray] = {}
    rows = []
    for name, model in models.items():
        model.fit(X_tr, y_train)
        pred = model.predict(X_te)
        predictions[name] = pred
        rmse, mape, da = _metrics(y_test, pred, close_test)
        rows.append((name, rmse, mape, da))

    persist = close_test.to_numpy(dtype=float)
    predictions["Persistence"] = persist
    rmse, mape, _da = _metrics(y_test, persist, close_test)
    # Persistence always predicts 0 change, so directional accuracy is not meaningful.
    rows.append(("Persistence", rmse, mape, float("nan")))

    perf_df = pd.DataFrame(
        {
            "Model": [r[0] for r in rows],
            "RMSE": [round(r[1], 4) for r in rows],
            "MAPE %": [round(r[2], 4) for r in rows],
            "Dir. Acc %": [round(r[3], 2) if r[3] == r[3] else None for r in rows],
        }
    ).set_index("Model")

    return {
        "scaler": scaler,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "close_test": close_test,
        "models": models,
        "predictions": predictions,
        "perf_df": perf_df,
        "k_eff": k_eff,
        "beats_persistence": (
            perf_df.loc[perf_df.index != "Persistence", "RMSE"].min()
            < perf_df.loc["Persistence", "RMSE"]
        ),
    }


def predict_next_day(results: dict, data: pd.DataFrame, features: list[str], price: float, ticker: str):
    """Forecast the *next* session close using the last completed bar."""
    from utils import money_symbol

    if not results or results.get("error"):
        return {}, float("nan"), money_symbol(ticker)

    latest = data[features].iloc[[-1]]
    if latest.isna().any(axis=1).iloc[0]:
        valid = data[features].dropna()
        if valid.empty:
            return {}, float("nan"), money_symbol(ticker)
        latest = valid.iloc[[-1]]

    latest_scaled = results["scaler"].transform(latest)
    preds = {}
    for name, model in results["models"].items():
        preds[name] = float(model.predict(latest_scaled)[0])
    preds["Persistence"] = float(price)
    ml_only = [preds[n] for n in results["models"]]
    pred_avg = float(np.mean(ml_only))
    return preds, pred_avg, money_symbol(ticker)
