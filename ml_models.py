"""
Machine learning model training, evaluation and prediction.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def _metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mse, rmse, mape


def train_models(data, features, k=5):
    data_ml = data.dropna()
    X = data_ml[features]
    y = data_ml["Close"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    models = {
        "KNN": KNeighborsRegressor(n_neighbors=k),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    predictions = {}
    metrics_list = []
    for name, model in models.items():
        model.fit(X_tr, y_train)
        pred = model.predict(X_te)
        predictions[name] = pred
        metrics_list.append(_metrics(y_test, pred))

    perf_df = pd.DataFrame({
        "Model": list(models.keys()),
        "RMSE": [round(m[1], 4) for m in metrics_list],
        "MAPE %": [round(m[2], 4) for m in metrics_list],
        "MSE": [round(m[0], 4) for m in metrics_list],
    }).set_index("Model")

    return {
        "scaler": scaler, "X": X, "y": y,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "models": models, "predictions": predictions, "perf_df": perf_df,
    }


def predict_next_day(results, price, ticker):
    latest = results["X"].tail(1)
    latest_scaled = results["scaler"].transform(latest)
    preds = {}
    for name, model in results["models"].items():
        preds[name] = float(model.predict(latest_scaled)[0])
    pred_avg = np.mean(list(preds.values()))
    cur_sym = "₹" if ".NS" in ticker else "$"
    return preds, pred_avg, cur_sym
