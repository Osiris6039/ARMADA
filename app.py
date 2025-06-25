# app.py - Deep Learning Forecasting App using PyTorch + Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib

# File setup
data_file = "data.csv"
event_file = "events.csv"
scaler_file = "scaler.pkl"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Initialize files if missing
if not os.path.exists(data_file):
    pd.DataFrame(columns=["Date", "Sales", "Customers", "Weather", "AddOnSales"]).to_csv(data_file, index=False)
if not os.path.exists(event_file):
    pd.DataFrame(columns=["EventDate", "EventName", "LastYearSales", "LastYearCustomers"]).to_csv(event_file, index=False)

# PyTorch model
class ForecastNet(nn.Module):
    def __init__(self, input_size):
        super(ForecastNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Output: Sales, Customers
        )

    def forward(self, x):
        return self.fc(x)

# Auth
st.set_page_config(page_title="PyTorch Forecast", layout="wide")
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "forecast123":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Daily Entry", "Event Entry", "Forecast"])
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.experimental_rerun()

# Load data
data = pd.read_csv(data_file, parse_dates=["Date"])
events = pd.read_csv(event_file)
today = pd.Timestamp.today().normalize()

if page == "Daily Entry":
    st.header("📥 Daily Entry")
    with st.form("input_form", clear_on_submit=True):
        date = st.date_input("Date")
        sales = st.number_input("Sales", 0)
        customers = st.number_input("Customers", 0)
        weather = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy"])
        addon = st.number_input("Add-On Sales", 0)
        if st.form_submit_button("Submit"):
            if not data[data["Date"] == pd.to_datetime(date)].empty:
                st.warning("Duplicate date detected. Entry skipped.")
            else:
                new = pd.DataFrame([{
                    "Date": date, "Sales": sales, "Customers": customers, "Weather": weather, "AddOnSales": addon
                }])
                data = pd.concat([data, new], ignore_index=True)
                data.to_csv(data_file, index=False)
                st.success("Saved")
    st.subheader("📊 Last 10 Days")
    st.dataframe(data[data["Date"] >= (today - pd.Timedelta(days=10))].sort_values("Date", ascending=False))

elif page == "Event Entry":
    st.header("📅 Future Event")
    with st.form("event_form", clear_on_submit=True):
        edate = st.date_input("Event Date")
        ename = st.text_input("Event Name")
        esales = st.number_input("Last Year Sales", 0)
        ecustomers = st.number_input("Last Year Customers", 0)
        if st.form_submit_button("Save Event"):
            new = pd.DataFrame([{
                "EventDate": edate.strftime('%Y-%m-%d'),
                "EventName": ename,
                "LastYearSales": esales,
                "LastYearCustomers": ecustomers
            }])
            events = pd.concat([events, new], ignore_index=True)
            events.to_csv(event_file, index=False)
            st.success("Event saved")

else:
    st.header("🔮 Forecast Next 10 Days")

    def prepare_data(df):
        df["DayOfYear"] = df["Date"].dt.dayofyear
        df["Weekday"] = df["Date"].dt.weekday
        df["Month"] = df["Date"].dt.month
        df["AddOnFlag"] = df["AddOnSales"].apply(lambda x: 1 if x > 0 else 0)
        df = pd.get_dummies(df, columns=["Weather"])
        X = df[["DayOfYear", "Weekday", "Month", "AddOnFlag"] + [c for c in df.columns if "Weather_" in c]]
        y = df[["Sales", "Customers"]]
        return X, y

    def train_model(X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_file)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32)

        model = ForecastNet(X.shape[1])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        losses = []
        for epoch in range(300):
            model.train()
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model_path = os.path.join(model_dir, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
        torch.save(model.state_dict(), model_path)
        return model, X.columns, losses, model_path

    def forecast_next_days(model, columns, days=10):
        model.eval()
        records = []
        scaler = joblib.load(scaler_file)
        for i in range(days):
            fdate = today + timedelta(days=i)
            weather = random.choice(["Sunny", "Rainy", "Cloudy"])
            row = {
                "DayOfYear": fdate.dayofyear,
                "Weekday": fdate.weekday(),
                "Month": fdate.month,
                "AddOnFlag": 0,
                "Weather_Sunny": 1 if weather == "Sunny" else 0,
                "Weather_Rainy": 1 if weather == "Rainy" else 0,
                "Weather_Cloudy": 1 if weather == "Cloudy" else 0
            }
            if not events[events["EventDate"] == fdate.strftime('%Y-%m-%d')].empty:
                row["AddOnFlag"] = 1
            df = pd.DataFrame([row])[columns]
            df_scaled = scaler.transform(df)
            tensor = torch.tensor(df_scaled, dtype=torch.float32)
            output = model(tensor)
            sales, customers = output[0].detach().numpy()
            records.append((fdate.strftime('%Y-%m-%d'), round(sales), round(customers)))
        return pd.DataFrame(records, columns=["Date", "Forecasted Sales", "Forecasted Customers"])

    if st.button("Run Forecast"):
        if len(data) < 10:
            st.warning("Need at least 10 days of data.")
        else:
            X, y = prepare_data(data.copy())
            model, feature_cols, losses, model_path = train_model(X, y)
            forecast = forecast_next_days(model, feature_cols)
            st.success(f"Model trained and saved to {model_path}")
            st.line_chart(losses)
            st.dataframe(forecast)
            st.download_button("📥 Download Forecast", forecast.to_csv(index=False), "forecast.csv")
            st.line_chart(forecast.set_index("Date"))
            st.info(f"📌 Last model: {model_path}")
