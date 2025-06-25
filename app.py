# app.py - Advanced AI Forecasting with Deep Learning

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Files
data_file = "data.csv"
event_file = "events.csv"
model_file = "model.h5"

# Initialize data files
if not os.path.exists(data_file):
    pd.DataFrame(columns=["Date", "Sales", "Customers", "Weather", "AddOnSales"]).to_csv(data_file, index=False)

if not os.path.exists(event_file):
    pd.DataFrame(columns=["EventDate", "EventName", "LastYearSales", "LastYearCustomers"]).to_csv(event_file, index=False)

# --- Auth ---
def authenticate(username, password):
    return username == "admin" and password == "forecast123"

st.set_page_config(page_title="Deep Learning AI Forecast", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid login")
    st.stop()

st.title("🤖 Deep Learning AI Forecasting App")

# --- Load Data ---
data = pd.read_csv(data_file)
events = pd.read_csv(event_file)
data["Date"] = pd.to_datetime(data["Date"])
today = pd.Timestamp.today().normalize()

# --- Input Section ---
st.header("📥 Add New Daily Entry")
with st.form("input_form", clear_on_submit=True):
    date = st.date_input("Date")
    sales = st.number_input("Sales", 0)
    customers = st.number_input("Customers", 0)
    weather = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy"])
    addon = st.number_input("Add-On Sales", 0)
    if st.form_submit_button("Submit Entry"):
        new_row = pd.DataFrame([{"Date": date, "Sales": sales, "Customers": customers, "Weather": weather, "AddOnSales": addon}])
        data = pd.concat([data, new_row], ignore_index=True)
        data.to_csv(data_file, index=False)
        st.success("Entry saved!")

# --- Event Input ---
st.header("📅 Input Future Event")
with st.form("event_form", clear_on_submit=True):
    edate = st.date_input("Event Date")
    ename = st.text_input("Event Name")
    esales = st.number_input("Last Year Sales", 0)
    ecustomers = st.number_input("Last Year Customers", 0)
    if st.form_submit_button("Submit Event"):
        new_event = pd.DataFrame([{
            "EventDate": edate.strftime('%Y-%m-%d'),
            "EventName": ename,
            "LastYearSales": esales,
            "LastYearCustomers": ecustomers
        }])
        events = pd.concat([events, new_event], ignore_index=True)
        events.to_csv(event_file, index=False)
        st.success("Event saved!")

# --- Show Table ---
st.subheader("📋 Recent 10 Days")
recent = data[data["Date"] >= (today - pd.Timedelta(days=10))]
st.dataframe(recent.sort_values("Date", ascending=False).reset_index(drop=True))

# --- Deep Learning Forecasting ---
def prepare_features(data):
    df = data.copy()
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["AddOnFlag"] = df["AddOnSales"].apply(lambda x: 1 if x > 0 else 0)
    df = df[["DayOfYear", "Weather", "AddOnFlag", "Sales", "Customers"]]
    X = df[["DayOfYear", "Weather", "AddOnFlag"]]
    y_sales = df["Sales"]
    y_cust = df["Customers"]
    X = pd.get_dummies(X)
    return X, y_sales, y_cust

def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_save_models(data):
    X, y_sales, y_cust = prepare_features(data)
    sm = build_model(X.shape[1])
    cm = build_model(X.shape[1])
    es = EarlyStopping(monitor='loss', patience=5, verbose=0)
    sm.fit(X, y_sales, epochs=100, verbose=0, callbacks=[es])
    cm.fit(X, y_cust, epochs=100, verbose=0, callbacks=[es])
    sm.save("model_sales.h5")
    cm.save("model_customers.h5")
    return sm, cm, X.columns

def forecast_next_days(model_sales, model_customers, columns, events, days=10):
    results = []
    for i in range(days):
        date = today + pd.Timedelta(days=i)
        row = {"DayOfYear": date.dayofyear, "Weather": "Sunny", "AddOnFlag": 0}
        if not events[events["EventDate"] == date.strftime('%Y-%m-%d')].empty:
            row["AddOnFlag"] = 1
        df = pd.DataFrame([row])
        df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
        sale = model_sales.predict(df)[0][0]
        cust = model_customers.predict(df)[0][0]
        results.append((date.strftime('%Y-%m-%d'), round(sale), round(cust)))
    return pd.DataFrame(results, columns=["Date", "Forecasted Sales", "Forecasted Customers"])

st.header("🔮 Deep Learning Forecast (10 Days)")
if st.button("Run Forecast"):
    if len(data) < 10:
        st.warning("At least 10 entries required.")
    else:
        if os.path.exists("model_sales.h5") and os.path.exists("model_customers.h5"):
            sm = load_model("model_sales.h5")
            cm = load_model("model_customers.h5")
            _, _, col = prepare_features(data)
        else:
            sm, cm, col = train_and_save_models(data)
        forecast_df = forecast_next_days(sm, cm, col, events)
        st.dataframe(forecast_df)
        st.download_button("📥 Download Forecast CSV", forecast_df.to_csv(index=False), "forecast.csv", "text/csv")
        st.line_chart(forecast_df.set_index("Date"))
