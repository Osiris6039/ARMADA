# app.py
# AI Sales and Customer Forecast Web App

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
import os

# --- Authentication ---
def authenticate(username, password):
    return username == "admin" and password == "forecast123"

# --- Data Files ---
data_file = "data.csv"
event_file = "events.csv"

# --- Initial Setup ---
if not os.path.exists(data_file):
    pd.DataFrame(columns=["Date", "Sales", "Customers", "Weather", "AddOnSales"]).to_csv(data_file, index=False)

if not os.path.exists(event_file):
    pd.DataFrame(columns=["EventDate", "EventName", "LastYearSales", "LastYearCustomers"]).to_csv(event_file, index=False)

# --- Forecasting Function ---
def train_forecaster(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values("Date", inplace=True)
    df['Day'] = df['Date'].dt.dayofyear
    df['AddOnFlag'] = df['AddOnSales'].fillna(0).apply(lambda x: 1 if x > 0 else 0)
    features = ['Day', 'Weather', 'AddOnFlag']
    X = pd.get_dummies(df[features])
    sales_model = GradientBoostingRegressor().fit(X, df['Sales'])
    cust_model = GradientBoostingRegressor().fit(X, df['Customers'])
    return sales_model, cust_model, X.columns

def make_forecast(sales_model, cust_model, columns, df, events, days=10):
    today = datetime.today()
    forecast_dates = [today + timedelta(days=i) for i in range(days)]
    forecasts = []

    for d in forecast_dates:
        day = d.timetuple().tm_yday
        row = {"Day": day, "AddOnFlag": 0, "Weather": "Sunny"}
        event_boost = events[events["EventDate"] == d.strftime('%Y-%m-%d')]
        if not event_boost.empty:
            row["AddOnFlag"] = 1
        df_row = pd.DataFrame([row])
        df_row = pd.get_dummies(df_row).reindex(columns=columns, fill_value=0)
        sale = sales_model.predict(df_row)[0]
        cust = cust_model.predict(df_row)[0]
        if not event_boost.empty:
            sale += event_boost['LastYearSales'].values[0] * 0.15
            cust += event_boost['LastYearCustomers'].values[0] * 0.10
        forecasts.append((d.strftime('%Y-%m-%d'), round(sale), round(cust)))
    return forecasts

# --- Streamlit UI ---
st.set_page_config(page_title="AI Forecast App", layout="wide")

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
            st.error("Invalid credentials")
    st.stop()

st.title("📊 AI Sales & Customer Forecasting App")

# --- Load Data ---
data = pd.read_csv(data_file)
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
event_data = pd.read_csv(event_file)

# --- Daily Data Input ---
st.header("📥 Input Daily Data")
with st.form("daily_form", clear_on_submit=True):
    date = st.date_input("Date")
    sales = st.number_input("Sales", 0)
    customers = st.number_input("Customers", 0)
    weather = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy"])
    addon = st.number_input("Add-On Sales", 0)
    if st.form_submit_button("Submit Entry"):
        new_row = pd.DataFrame([{"Date": date, "Sales": sales, "Customers": customers, "Weather": weather, "AddOnSales": addon}])
        data = pd.concat([data, new_row], ignore_index=True)
        data["Date"] = pd.to_datetime(data["Date"])
        data.sort_values("Date", inplace=True)
        data.to_csv(data_file, index=False)
        st.success("Entry added!")

# --- Event Input ---
st.header("📅 Input Future Event")
with st.form("event_form", clear_on_submit=True):
    edate = st.date_input("Event Date")
    ename = st.text_input("Event Name")
    esales = st.number_input("Last Year's Sales", 0)
    ecustomers = st.number_input("Last Year's Customers", 0)
    if st.form_submit_button("Submit Event"):
        new_event = pd.DataFrame([{
            "EventDate": edate.strftime('%Y-%m-%d'),
            "EventName": ename,
            "LastYearSales": esales,
            "LastYearCustomers": ecustomers
        }])
        event_data = pd.concat([event_data, new_event], ignore_index=True)
        event_data.to_csv(event_file, index=False)
        st.success("Event added!")


# --- Show Data Records ---
st.subheader("📋 Daily Records")

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter to show last 10 days
today = pd.Timestamp.today().normalize()
last_10_days = data[data['Date'] >= (today - pd.Timedelta(days=10))]

# Show as table
if not last_10_days.empty:
    st.dataframe(last_10_days.sort_values("Date", ascending=False).reset_index(drop=True))
else:
    st.info("No entries in the last 10 days.")

# Option to show all data monthly
if st.button("📂 View All Historical Data"):
    selected_month = st.selectbox("Select Month", sorted(data['Date'].dt.to_period('M').astype(str).unique()), index=len(data['Date'].dt.to_period('M').unique()) - 1)
    selected_data = data[data['Date'].dt.to_period('M').astype(str) == selected_month]
    for idx, row in selected_data.iterrows():
        with st.expander(f"{row['Date'].strftime('%Y-%m-%d')}"):
            editable = st.columns(5)
            new_sales = editable[0].number_input("Sales", value=int(row['Sales']), key=f"edit_sales_{idx}")
            new_customers = editable[1].number_input("Customers", value=int(row['Customers']), key=f"edit_customers_{idx}")
            new_weather = editable[2].selectbox("Weather", ["Sunny", "Rainy", "Cloudy"], index=["Sunny", "Rainy", "Cloudy"].index(row["Weather"]), key=f"edit_weather_{idx}")
            new_addon = editable[3].number_input("AddOnSales", value=int(row["AddOnSales"]), key=f"edit_addon_{idx}")

            if editable[4].button("Update", key=f"update_{idx}"):
                data.at[idx, "Sales"] = new_sales
                data.at[idx, "Customers"] = new_customers
                data.at[idx, "Weather"] = new_weather
                data.at[idx, "AddOnSales"] = new_addon
                data.to_csv(data_file, index=False)
                st.success("Record updated.")
                st.rerun()

            if st.button(f"Delete {row['Date'].strftime('%Y-%m-%d')}", key=f"del_full_{idx}"):
                data = data.drop(index=idx)
                data.to_csv(data_file, index=False)
                st.rerun()


# --- Forecast Button ---
st.header("🔮 Forecast 10 Days Ahead")
if st.button("Run Forecast"):
    if len(data) < 5:
        st.warning("Need at least 5 data entries to generate forecast.")
    else:
        sm, cm, col = train_forecaster(data)
        forecast = make_forecast(sm, cm, col, data, event_data)
        forecast_df = pd.DataFrame(forecast, columns=["Date", "Forecasted Sales", "Forecasted Customers"])
        st.write(forecast_df)
        st.download_button("📥 Download Forecast CSV", forecast_df.to_csv(index=False), "forecast.csv", "text/csv")
        st.line_chart(forecast_df.set_index("Date"))
