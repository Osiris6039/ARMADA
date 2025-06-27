
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import pickle
from datetime import timedelta, datetime

def load_model():
    if os.path.exists("models/forecast_model.pkl"):
        with open("models/forecast_model.pkl", "rb") as f:
            return pickle.load(f)
    else:
        return xgb.XGBRegressor()

def retrain_model(data):
    data = data.copy()
    if len(data) < 10:
        return
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["is_weekend"] = data["day_of_week"] >= 5
    X = data[["day_of_week", "month", "is_weekend", "customers", "add_on_sales"]]
    y = data["sales"]
    model = xgb.XGBRegressor()
    model.fit(X, y)
    with open("models/forecast_model.pkl", "wb") as f:
        pickle.dump(model, f)

def predict_next_10_days(data, model):
    last_date = data["date"].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 11)]
    predictions = []
    for date in future_dates:
        dow = date.weekday()
        month = date.month
        is_weekend = dow >= 5
        # Estimate using average of similar past days
        past = data[data["date"].dt.weekday == dow]
        avg_customers = past["customers"].tail(5).mean()
        avg_addons = past["add_on_sales"].tail(5).mean()
        X_pred = pd.DataFrame([[dow, month, is_weekend, avg_customers, avg_addons]], 
                              columns=["day_of_week", "month", "is_weekend", "customers", "add_on_sales"])
        y_pred = model.predict(X_pred)[0]
        predictions.append({"date": date, "forecast_sales": round(y_pred), "forecast_customers": round(avg_customers)})
    return pd.DataFrame(predictions)
