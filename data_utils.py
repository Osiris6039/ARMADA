
import pandas as pd
from datetime import datetime, timedelta

def load_data():
    try:
        df = pd.read_csv("data/data.csv", parse_dates=["date"])
        df.sort_values("date", inplace=True)
        return df
    except:
        return pd.DataFrame(columns=["date", "sales", "customers", "add_on_sales", "weather"])

def save_data(row):
    df = load_data()
    df = df.append(row, ignore_index=True)
    df.to_csv("data/data.csv", index=False)

def get_last_7_days(df):
    return df.tail(7)

def display_accuracy_graph(df):
    import matplotlib.pyplot as plt
    import streamlit as st

    if "forecast" in df.columns:
        actual = df["sales"].tail(10)
        forecast = df["forecast"].tail(10)
        plt.plot(actual.values, label="Actual")
        plt.plot(forecast.values, label="Forecast")
        plt.legend()
        st.pyplot(plt)
    else:
        st.info("No forecast data available for comparison.")
