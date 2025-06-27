
import streamlit as st
import pandas as pd
from utils.data_utils import save_data, load_data

def render(data):
    st.subheader("📝 Input Daily Sales Data")
    with st.form("input_form"):
        date = st.date_input("Date")
        sales = st.number_input("Total Sales", min_value=0.0)
        customers = st.number_input("Number of Customers", min_value=0)
        add_on_sales = st.number_input("Add-On Sales", min_value=0.0)
        weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Stormy"])
        submitted = st.form_submit_button("Submit")
        if submitted:
            new_row = {"date": date, "sales": sales, "customers": customers, "add_on_sales": add_on_sales, "weather": weather}
            save_data(new_row)
            st.success("Data saved!")

    st.subheader("📅 Last 7 Entries")
    st.dataframe(data.tail(7))
