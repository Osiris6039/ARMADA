
import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import bcrypt
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ==============================
# Utility and Authentication
# ==============================

USERS = {
    "admin": bcrypt.hashpw("forecast123".encode(), bcrypt.gensalt())
}

def authenticate(username, password):
    if username in USERS:
        return bcrypt.checkpw(password.encode(), USERS[username])
    return False

# ==============================
# Model Definition
# ==============================

class SalesCustomerPredictor(nn.Module):
    def __init__(self, input_size):
        super(SalesCustomerPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)

# ==============================
# App Layout
# ==============================

def main():
    st.title("🔐 AI Sales & Customer Forecasting")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.experimental_rerun()
            else:
                st.error("❌ Incorrect credentials")
        return

    section = st.sidebar.radio("Navigation", ["Daily Input", "Event Entry", "Forecast"])

    if section == "Daily Input":
        daily_input()
    elif section == "Event Entry":
        event_entry()
    elif section == "Forecast":
        forecast()

# ==============================
# Daily Input Section
# ==============================

def daily_input():
    st.header("📥 Input Daily Data")
    date = st.date_input("Date")
    sales = st.number_input("Sales", min_value=0)
    customers = st.number_input("Customers", min_value=0)
    weather = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy"])
    addon = st.checkbox("Add-On Sales (e.g., Party, Promo)")

    df = load_data("data.csv")
    if st.button("Save Entry"):
        if date in pd.to_datetime(df["Date"]).dt.date.values:
            st.warning("⚠️ Duplicate date entry")
        else:
            new = {
                "Date": date.strftime("%Y-%m-%d"),
                "Sales": sales,
                "Customers": customers,
                "Weather": weather,
                "AddOnFlag": int(addon)
            }
            df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
            df.to_csv("data.csv", index=False)
            st.success("✅ Entry saved")

# ==============================
# Event Entry Section
# ==============================

def event_entry():
    st.header("📅 Add Future Events")
    date = st.date_input("Event Date", min_value=datetime.today())
    name = st.text_input("Event Name")
    last_sales = st.number_input("Last Year Sales", min_value=0)
    last_customers = st.number_input("Last Year Customers", min_value=0)

    df = load_data("events.csv")
    if st.button("Save Event"):
        new = {
            "Date": date.strftime("%Y-%m-%d"),
            "Event": name,
            "LastYearSales": last_sales,
            "LastYearCustomers": last_customers
        }
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        df.to_csv("events.csv", index=False)
        st.success("✅ Event saved")

# ==============================
# Forecast Section
# ==============================

def forecast():
    st.header("🔮 10-Day Forecast")
    if not os.path.exists("data.csv"):
        st.warning("No data to train on.")
        return

    df = pd.read_csv("data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Weekday"] = df["Date"].dt.weekday
    df["Month"] = df["Date"].dt.month
    df = pd.get_dummies(df, columns=["Weather"])

    for col in ["Weather_Sunny", "Weather_Rainy", "Weather_Cloudy"]:
        if col not in df:
            df[col] = 0

    features = ["DayOfYear", "Weekday", "Month", "AddOnFlag",
                "Weather_Sunny", "Weather_Rainy", "Weather_Cloudy"]
    X = df[features].values.astype(np.float32)
    y = df[["Sales", "Customers"]].values.astype(np.float32)

    model = SalesCustomerPredictor(input_size=X.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for epoch in range(300):
        optimizer.zero_grad()
        output = model(torch.tensor(X))
        loss = loss_fn(output, torch.tensor(y))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    st.line_chart(losses)
    torch.save(model.state_dict(), "model.pt")

    forecast_dates = [datetime.today() + timedelta(days=i) for i in range(1, 11)]
    forecast_data = pd.DataFrame({
        "Date": forecast_dates,
        "DayOfYear": [d.timetuple().tm_yday for d in forecast_dates],
        "Weekday": [d.weekday() for d in forecast_dates],
        "Month": [d.month for d in forecast_dates],
        "AddOnFlag": [0]*10,
        "Weather": np.random.choice(["Sunny", "Rainy", "Cloudy"], 10)
    })

    forecast_data = pd.get_dummies(forecast_data, columns=["Weather"])
    for col in ["Weather_Sunny", "Weather_Rainy", "Weather_Cloudy"]:
        if col not in forecast_data:
            forecast_data[col] = 0

    forecast_data = forecast_data[features]
    with torch.no_grad():
        prediction = model(torch.tensor(forecast_data.values.astype(np.float32))).numpy()

    result = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in forecast_dates],
        "Forecasted Sales": prediction[:, 0].round(2),
        "Forecasted Customers": prediction[:, 1].round(2)
    })
    st.dataframe(result)
    st.download_button("Download Forecast", result.to_csv(index=False), "forecast.csv", "text/csv")

# ==============================
# Helper Functions
# ==============================

def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

# ==============================
# Run App
# ==============================

if __name__ == "__main__":
    main()
