
import streamlit as st
import pandas as pd

def render():
    st.subheader("📆 Event History & Future Events")
    try:
        event_data = pd.read_csv("data/event_data.csv")
    except:
        event_data = pd.DataFrame(columns=["date", "event_name", "last_year_sales", "last_year_customers"])
    with st.form("event_form"):
        date = st.date_input("Event Date")
        name = st.text_input("Event Name")
        ly_sales = st.number_input("Last Year Sales", min_value=0.0)
        ly_customers = st.number_input("Last Year Customers", min_value=0)
        submitted = st.form_submit_button("Save Event")
        if submitted:
            new_event = {"date": date, "event_name": name, "last_year_sales": ly_sales, "last_year_customers": ly_customers}
            event_data = event_data.append(new_event, ignore_index=True)
            event_data.to_csv("data/event_data.csv", index=False)
            st.success("Event saved!")
    st.dataframe(event_data)
