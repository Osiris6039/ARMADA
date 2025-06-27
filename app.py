import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import joblib # For saving/loading models
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import logging

# Suppress Prophet logs to keep Streamlit output clean
logging.getLogger('prophet').setLevel(logging.WARNING)

# --- Configuration Constants ---
DATA_DIR = "data"
MODELS_DIR = "models"
SALES_DATA_PATH = os.path.join(DATA_DIR, "sales_data.csv")
EVENTS_DATA_PATH = os.path.join(DATA_DIR, "events_data.csv")

# Paths for saved machine learning models
SALES_RF_MODEL_PATH = os.path.join(MODELS_DIR, "sales_rf_model.pkl")
CUSTOMERS_RF_MODEL_PATH = os.path.join(MODELS_DIR, "customers_rf_model.pkl")
SALES_PROPHET_MODEL_PATH = os.path.join(MODELS_DIR, "sales_prophet_model.pkl")
CUSTOMERS_PROPHET_MODEL_PATH = os.path.join(MODELS_DIR, "customers_prophet_model.pkl")

# Ensure necessary directories exist at the start of the application
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Data Loading and Saving Functions ---
@st.cache_data(show_spinner=False) # Cache data loading to improve performance
def load_sales_data():
    """
    Loads sales data from 'sales_data.csv'.
    If the file doesn't exist or is empty, an empty DataFrame is created and saved.
    Ensures 'Date' column is datetime, and data is sorted and deduplicated by date.
    """
    if not os.path.exists(SALES_DATA_PATH) or os.path.getsize(SALES_DATA_PATH) == 0:
        # Create an empty DataFrame with the expected columns and datetime type for 'Date'
        df = pd.DataFrame(columns=['Date', 'Sales', 'Customers', 'Add_on_Sales', 'Weather'])
        df['Date'] = pd.to_datetime(df['Date']) # Convert to datetime (will be NaT for empty initially)
        df.to_csv(SALES_DATA_PATH, index=False)
    else:
        # Load existing data
        df = pd.read_csv(SALES_DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # Coerce invalid dates to NaT
        df = df.dropna(subset=['Date']) # Drop rows where Date conversion failed

    # Ensure loaded data is always sorted by Date and unique (keeping the latest entry for any duplicate date)
    return df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)

def save_sales_data(df):
    """
    Saves the given DataFrame to 'sales_data.csv'.
    Data is deduplicated and sorted before saving to maintain file consistency.
    Clears Streamlit's cache for 'load_sales_data' to force a fresh reload next time.
    """
    df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last').to_csv(SALES_DATA_PATH, index=False)
    st.cache_data.clear() # Clear cache for load_sales_data()

@st.cache_data(show_spinner=False) # Cache data loading for events
def load_events_data():
    """
    Loads event data from 'events_data.csv'.
    If the file doesn't exist or is empty, an empty DataFrame is created and saved.
    Ensures 'Event_Date' column is datetime, and data is sorted and deduplicated by event date.
    """
    if not os.path.exists(EVENTS_DATA_PATH) or os.path.getsize(EVENTS_DATA_PATH) == 0:
        # Create an empty DataFrame with expected columns
        df = pd.DataFrame(columns=['Event_Date', 'Event_Name', 'Impact'])
        df['Event_Date'] = pd.to_datetime(df['Event_Date'])
        df.to_csv(EVENTS_DATA_PATH, index=False)
    else:
        # Load existing data
        df = pd.read_csv(EVENTS_DATA_PATH)
        df['Event_Date'] = pd.to_datetime(df['Event_Date'], errors='coerce') # Coerce invalid dates
        df = df.dropna(subset=['Event_Date']) # Drop rows where Event_Date conversion failed

    # Ensure loaded data is always sorted by Event_Date and unique
    return df.sort_values('Event_Date').drop_duplicates(subset=['Event_Date'], keep='last').reset_index(drop=True)

def save_events_data(df):
    """
    Saves the given DataFrame to 'events_data.csv'.
    Data is deduplicated and sorted before saving.
    Clears Streamlit's cache for 'load_events_data' to force a fresh reload.
    """
    df.sort_values('Event_Date').drop_duplicates(subset=['Event_Date'], keep='last').to_csv(EVENTS_DATA_PATH, index=False)
    st.cache_data.clear() # Clear cache for load_events_data()

# --- Preprocessing for RandomForestRegressor ---
def preprocess_rf_data(df_sales, df_events):
    """
    Preprocesses sales and events data to create features for RandomForestRegressor.
    Includes time-based features, weather one-hot encoding, and event impact.
    Handles empty input DataFrames.
    """
    if df_sales.empty:
        # Return empty dataframes/series if sales data is empty, as no features can be created
        return pd.DataFrame(), pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.DataFrame()

    df = df_sales.copy() # Work on a copy to prevent SettingWithCopyWarning
    df['Date'] = pd.to_datetime(df['Date']) # Ensure Date is datetime type
    df = df.sort_values('Date').reset_index(drop=True)

    # Time-based features extraction
    df['day_of_week'] = df['Date'].dt.dayofweek # Monday=0, Sunday=6
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int) # Week of year as integer
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int) # 1 for weekend, 0 for weekday

    # Weather one-hot encoding
    all_weather_conditions = ['Sunny', 'Cloudy', 'Rainy', 'Snowy'] # Standardize weather conditions
    for cond in all_weather_conditions:
        col_name = f'weather_{cond}'
        df[col_name] = (df['Weather'] == cond).astype(int) # Create a binary column for each weather type

    # Merge with events data to incorporate event impact
    df['is_event'] = 0 # Initialize 'is_event' flag
    df['event_impact_score'] = 0.0 # Initialize numerical event impact score

    if not df_events.empty:
        df_events_copy = df_events.copy()
        df_events_copy['Event_Date'] = pd.to_datetime(df_events_copy['Event_Date'])
        impact_map = {'Low': 0.1, 'Medium': 0.5, 'High': 1.0} # Map impact levels to numerical scores
        df_events_copy['Impact_Score'] = df_events_copy['Impact'].map(impact_map).fillna(0)

        # Perform left merge to bring event data into the sales dataframe
        merged = pd.merge(df[['Date']], df_events_copy[['Event_Date', 'Impact_Score']],
                          left_on='Date', right_on='Event_Date', how='left')
        
        df['is_event'] = merged['Event_Date'].notna().astype(int) # Set flag if an event occurred
        df['event_impact_score'] = merged['Impact_Score'].fillna(0) # Use impact score, default to 0 if no event

    # Lag features: sales and customers from previous days/weeks
    # These are crucial for time series prediction and will be updated iteratively during forecasting
    df['Sales_Lag1'] = df['Sales'].shift(1)
    df['Customers_Lag1'] = df['Customers'].shift(1)
    df['Sales_Lag7'] = df['Sales'].shift(7) # Sales from the same day last week
    df['Customers_Lag7'] = df['Customers'].shift(7)

    # Fill NaN values created by shifting (e.g., first few days won't have lags)
    # Using 0 as a placeholder for missing prior data
    df = df.fillna(0)

    # Define the list of features to be used for model training
    feature_columns = [
        'day_of_week', 'day_of_year', 'month', 'year', 'week_of_year', 'is_weekend',
        'Sales_Lag1', 'Customers_Lag1', 'Sales_Lag7', 'Customers_Lag7',
        'is_event', 'event_impact_score'
    ]
    feature_columns.extend([f'weather_{cond}' for cond in all_weather_conditions])

    # Ensure all feature columns exist in the DataFrame, adding with 0 if missing
    # This prevents errors if a particular weather condition, for instance, isn't in the training data
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_columns] # Features DataFrame
    y_sales = df['Sales']   # Target Series for Sales
    y_customers = df['Customers'] # Target Series for Customers

    # Store feature columns and weather conditions in session state for later use in forecasting
    st.session_state['rf_feature_columns'] = feature_columns
    st.session_state['all_weather_conditions'] = all_weather_conditions

    return X, y_sales, y_customers, df # Return the processed DataFrame for potential further use (e.g., lag initialization)

# --- Preprocessing for Prophet ---
def preprocess_prophet_data(df_sales, df_events, target_column):
    """
    Preprocesses data for the Prophet model.
    Transforms data to 'ds' (datetime) and 'y' (target) format.
    Integrates 'Add_on_Sales' and weather conditions as extra regressors, and events as holidays.
    Handles empty input DataFrames.
    """
    if df_sales.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df_sales.copy() # Work on a copy
    df['ds'] = pd.to_datetime(df['Date']) # Prophet requires a 'ds' column for datetime
    df['y'] = df[target_column] # Prophet requires a 'y' column for the target variable

    # Add 'Add_on_Sales' as an external regressor if it's not the target itself
    if target_column != 'Add_on_Sales':
        df['Add_on_Sales'] = df_sales['Add_on_Sales']
    
    # Add weather as extra regressors using one-hot encoding
    # Ensure 'Weather' column exists and is not empty before creating dummies
    if 'Weather' in df.columns and not df['Weather'].empty:
        weather_dummies = pd.get_dummies(df['Weather'], prefix='weather')
        df = pd.concat([df, weather_dummies], axis=1)
    else:
        # If 'Weather' column is missing or empty, create dummy columns initialized to zeros
        for cond in ['Sunny', 'Cloudy', 'Rainy', 'Snowy']:
            df[f'weather_{cond}'] = 0

    # Prepare holidays DataFrame for Prophet from events data
    holidays_df = pd.DataFrame()
    if not df_events.empty:
        holidays_df = df_events.rename(columns={'Event_Date': 'ds', 'Event_Name': 'holiday'})
        holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
        holidays_df = holidays_df[['ds', 'holiday']].drop_duplicates(subset=['ds']) # Ensure unique holidays by date

    # Select final columns for Prophet model training
    prophet_df = df[['ds', 'y']].copy() # Create a copy for prophet_df
    if target_column != 'Add_on_Sales': # Avoid circularity
        prophet_df['Add_on_Sales'] = df['Add_on_Sales'] # Add add-on sales as regressor
    
    # Add weather regressors to the prophet_df
    all_weather_conditions = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
    for cond in all_weather_conditions:
        col_name = f'weather_{cond}'
        if col_name in df.columns:
            prophet_df[col_name] = df[col_name]
        else:
            prophet_df[col_name] = 0 # Ensure regressor column exists, even if all zeros

    return prophet_df, holidays_df

# --- AI Model Training Functions ---
def train_random_forest_models(X, y_sales, y_customers, n_estimators):
    """
    Trains RandomForestRegressor models for Sales and Customers, then saves them to disk.
    Requires at least 2 data points for meaningful training (due to lag features).
    """
    if X.empty or len(X) < 2:
        st.warning("Not enough data to train the RandomForest models. Need at least 2 sales records for meaningful features and training.")
        return None, None

    sales_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    customers_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)

    sales_model.fit(X, y_sales)
    joblib.dump(sales_model, SALES_RF_MODEL_PATH) # Save sales model

    customers_model.fit(X, y_customers)
    joblib.dump(customers_model, CUSTOMERS_RF_MODEL_PATH) # Save customers model

    return sales_model, customers_model

def train_prophet_models(prophet_sales_df, prophet_customers_df, holidays_df):
    """
    Trains Prophet models for Sales and Customers, then saves them to disk.
    Requires at least 2 data points for Prophet training.
    """
    if prophet_sales_df.empty or prophet_customers_df.empty:
        st.warning("Not enough data to train the Prophet models. Please add more sales records.")
        return None, None

    if len(prophet_sales_df) < 2 or len(prophet_customers_df) < 2:
        st.warning("Prophet requires at least 2 data points for training. Please add more sales records.")
        return None, None

    sales_prophet_model = Prophet(holidays=holidays_df, interval_width=0.95) # 95% confidence interval
    customers_prophet_model = Prophet(holidays=holidays_df, interval_width=0.95)

    # Add extra regressors to Prophet models if they exist in the dataframe
    if 'Add_on_Sales' in prophet_sales_df.columns:
        sales_prophet_model.add_regressor('Add_on_Sales')
        customers_prophet_model.add_regressor('Add_on_Sales')
    
    weather_cols = [col for col in prophet_sales_df.columns if col.startswith('weather_')]
    for col in weather_cols:
        sales_prophet_model.add_regressor(col)
        customers_prophet_model.add_regressor(col)

    sales_prophet_model.fit(prophet_sales_df)
    joblib.dump(sales_prophet_model, SALES_PROPHET_MODEL_PATH) # Save sales prophet model

    customers_prophet_model.fit(prophet_customers_df)
    joblib.dump(customers_prophet_model, CUSTOMERS_PROPHET_MODEL_PATH) # Save customers prophet model

    return sales_prophet_model, customers_prophet_model

@st.cache_resource(hash_funcs={pd.DataFrame: pd.util.hash_pandas_object, pd.Series: pd.util.hash_pandas_object})
def load_or_train_models(model_type, n_estimators_rf=100):
    """
    Loads pre-trained models from disk if they exist, otherwise trains them.
    Models are cached to avoid retraining on every Streamlit rerun if data hasn't changed.
    """
    sales_df_current = load_sales_data() # Get current sales data
    events_df_current = load_events_data() # Get current events data

    sales_model = None
    customers_model = None

    if model_type == "RandomForest":
        sales_model_path = SALES_RF_MODEL_PATH
        customers_model_path = CUSTOMERS_RF_MODEL_PATH
        
        # Only attempt to load/train if sufficient sales data is available for feature creation
        if not sales_df_current.empty and sales_df_current.shape[0] >= 2:
            X, y_sales, y_customers, _ = preprocess_rf_data(sales_df_current, events_df_current)
            if not X.empty and X.shape[0] >= 2: # Ensure features DataFrame is also not empty and has enough samples
                if os.path.exists(sales_model_path) and os.path.exists(customers_model_path):
                    try:
                        sales_model = joblib.load(sales_model_path)
                        customers_model = joblib.load(customers_model_path)
                        st.info("RandomForest models loaded from disk.")
                    except Exception as e:
                        st.error(f"Error loading RandomForest models: {e}. Attempting to retrain.")
                        sales_model, customers_model = train_random_forest_models(X, y_sales, y_customers, n_estimators_rf)
                else:
                    st.info("No RandomForest models found on disk. Training AI models now...")
                    sales_model, customers_model = train_random_forest_models(X, y_sales, y_customers, n_estimators_rf)
            else:
                st.info("Not enough valid preprocessed data for RandomForest training (requires at least 2 records for features/lags).")
        else:
            st.info("Insufficient sales data (minimum 2 records) to train RandomForest models.")


    elif model_type == "Prophet":
        sales_model_path = SALES_PROPHET_MODEL_PATH
        customers_model_path = CUSTOMERS_PROPHET_MODEL_PATH

        # Only attempt to load/train if sufficient sales data is available
        if not sales_df_current.empty and sales_df_current.shape[0] >= 2:
            prophet_sales_df, holidays_df = preprocess_prophet_data(sales_df_current, events_df_current, 'Sales')
            prophet_customers_df, _ = preprocess_prophet_data(sales_df_current, events_df_current, 'Customers')

            if not prophet_sales_df.empty and not prophet_customers_df.empty and len(prophet_sales_df) >= 2:
                if os.path.exists(sales_model_path) and os.path.exists(customers_model_path):
                    try:
                        sales_model = joblib.load(sales_model_path)
                        customers_model = joblib.load(customers_model_path)
                        st.info("Prophet models loaded from disk.")
                    except Exception as e:
                        st.error(f"Error loading Prophet models: {e}. Attempting to retrain.")
                        sales_model, customers_model = train_prophet_models(prophet_sales_df, prophet_customers_df, holidays_df)
                else:
                    st.info("No Prophet models found on disk. Training AI models now...")
                    sales_model, customers_model = train_prophet_models(prophet_sales_df, prophet_customers_df, holidays_df)
            else:
                st.info("Not enough valid preprocessed data for Prophet training (requires at least 2 records).")
        else:
            st.info("Insufficient sales data (minimum 2 records) to train Prophet models.")

    return sales_model, customers_model

# --- Forecast Generation Functions ---
def generate_rf_forecast(sales_df, events_df, sales_model, customers_model, future_weather_inputs, num_days=10):
    """
    Generates sales and customer forecasts for the next N days using RandomForest.
    Uses an iterative (chained) prediction approach for lagged features.
    """
    if sales_model is None or customers_model is None:
        st.warning("RandomForest models are not trained. Please ensure you have sufficient data and a model is selected and trained.")
        return pd.DataFrame()

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    forecast_dates = [today + timedelta(days=i) for i in range(1, num_days + 1)]

    # Prepare historical data for initializing lag features for the first forecast day
    historical_data_for_lags = sales_df.copy()
    historical_data_for_lags['Date'] = pd.to_datetime(historical_data_for_lags['Date'])
    historical_data_for_lags = historical_data_for_lags.sort_values('Date').reset_index(drop=True)

    forecast_results = []
    
    # Initialize current day's lag features from the *latest* actual historical data
    current_sales_lag1 = historical_data_for_lags['Sales'].iloc[-1] if not historical_data_for_lags.empty else 0
    current_customers_lag1 = historical_data_for_lags['Customers'].iloc[-1] if not historical_data_for_lags.empty else 0
    
    # Get last 7 days of sales/customers for Sales_Lag7/Customers_Lag7
    # Pad with zeros if less than 7 days of history available
    last_7_sales = historical_data_for_lags['Sales'].tail(7).tolist()
    last_7_customers = historical_data_for_lags['Customers'].tail(7).tolist()
    
    # Ensure these lists always have 7 elements, padding with 0s at the beginning if history is short
    last_7_sales = [0] * (7 - len(last_7_sales)) + last_7_sales
    last_7_customers = [0] * (7 - len(last_7_customers)) + last_7_customers

    # Iterate through each future day to generate a forecast
    for i in range(num_days):
        forecast_date = forecast_dates[i]
        
        # Get the weather input for the current forecast date from user-defined future weather
        current_weather_input = next((item['weather'] for item in future_weather_inputs if item['date'] == forecast_date.strftime('%Y-%m-%d')), 'Sunny')

        # Construct a DataFrame of features for the current forecast date
        current_features = pd.DataFrame([{
            'Date': forecast_date,
            'day_of_week': forecast_date.weekday(),
            'day_of_year': forecast_date.dayofyear,
            'month': forecast_date.month,
            'year': forecast_date.year,
            'week_of_year': forecast_date.isocalendar().week.astype(int),
            'is_weekend': int(forecast_date.weekday() in [5, 6]),
            'Sales_Lag1': current_sales_lag1,       # Use the lagged value from previous prediction or actual
            'Customers_Lag1': current_customers_lag1, # Use the lagged value from previous prediction or actual
            'Sales_Lag7': last_7_sales[i],           # Use the appropriate lag from the 7-day history
            'Customers_Lag7': last_7_customers[i],   # Use the appropriate lag from the 7-day history
            'is_event': 0,
            'event_impact_score': 0.0
        }])

        # Add weather one-hot encoded columns
        all_weather_conditions = st.session_state.get('all_weather_conditions', ['Sunny', 'Cloudy', 'Rainy', 'Snowy'])
        for cond in all_weather_conditions:
            col_name = f'weather_{cond}'
            current_features[col_name] = (current_weather_input == cond).astype(int)

        # Incorporate event impact for future dates if applicable
        if not events_df.empty:
            matching_event = events_df[events_df['Event_Date'] == forecast_date]
            if not matching_event.empty:
                current_features['is_event'] = 1
                impact_map = {'Low': 0.1, 'Medium': 0.5, 'High': 1.0}
                current_features['event_impact_score'] = impact_map.get(matching_event['Impact'].iloc[0], 0)

        # Ensure feature columns match the model's expected order and presence
        feature_cols = st.session_state.get('rf_feature_columns', [])
        for col in feature_cols:
            if col not in current_features.columns:
                current_features[col] = 0 # Add missing features with zero
        
        input_for_prediction = current_features[feature_cols]

        # Predict sales and customers using the trained RandomForest models
        predicted_sales = sales_model.predict(input_for_prediction)[0]
        predicted_customers = customers_model.predict(input_for_prediction)[0]

        # Calculate approximate 95% Confidence Intervals using quantiles of individual tree predictions
        sales_predictions_per_tree = np.array([tree.predict(input_for_prediction)[0] for tree in sales_model.estimators_])
        customers_predictions_per_tree = np.array([tree.predict(input_for_prediction)[0] for tree in customers_model.estimators_])
        
        sales_lower = np.percentile(sales_predictions_per_tree, 2.5)
        sales_upper = np.percentile(sales_predictions_per_tree, 97.5)
        customers_lower = np.percentile(customers_predictions_per_tree, 2.5)
        customers_upper = np.percentile(customers_predictions_per_tree, 97.5)

        # Store forecast results, ensuring values are non-negative and rounded
        forecast_results.append({
            'Date': forecast_date.strftime('%Y-%m-%d'),
            'Forecasted Sales': max(0, round(predicted_sales, 2)),
            'Sales Lower Bound (95%)': max(0, round(sales_lower, 2)),
            'Sales Upper Bound (95%)': max(0, round(sales_upper, 2)),
            'Forecasted Customers': max(0, round(predicted_customers)),
            'Customers Lower Bound (95%)': max(0, round(customers_lower)),
            'Customers Upper Bound (95%)': max(0, round(customers_upper)),
            'Weather': current_weather_input # Include the assumed weather for traceability
        })

        # Update lag values for the next iteration of forecasting (chained prediction)
        current_sales_lag1 = predicted_sales
        current_customers_lag1 = predicted_customers
        
        # Update the 7-day history lists by removing the oldest and adding the new prediction
        last_7_sales.pop(0)
        last_7_sales.append(predicted_sales)
        last_7_customers.pop(0)
        last_7_customers.append(predicted_customers)

    return pd.DataFrame(forecast_results)

def generate_prophet_forecast(sales_df, events_df, sales_model, customers_model, future_weather_inputs, num_days=10):
    """
    Generates sales and customer forecasts for the next N days using Prophet.
    """
    if sales_model is None or customers_model is None:
        st.warning("Prophet models are not trained. Please ensure you have sufficient data and a model is selected and trained.")
        return pd.DataFrame()

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    forecast_dates = [today + timedelta(days=i) for i in range(1, num_days + 1)]

    # Create a future DataFrame required by Prophet for predictions
    future_prophet_df = pd.DataFrame({'ds': forecast_dates})
    
    # Add 'Add_on_Sales' as an external regressor for future dates.
    # For simplicity, we use the historical average; in a real app, this might come from another forecast.
    avg_add_on_sales = sales_df['Add_on_Sales'].mean() if not sales_df.empty else 0
    future_prophet_df['Add_on_Sales'] = avg_add_on_sales

    # Add weather as extra regressors based on user inputs for future weather
    all_weather_conditions = st.session_state.get('all_weather_conditions', ['Sunny', 'Cloudy', 'Rainy', 'Snowy'])
    for cond in all_weather_conditions:
        col_name = f'weather_{cond}'
        future_prophet_df[col_name] = 0 # Initialize weather columns to zero

    # Populate the weather columns based on the user's future weather inputs
    for i, row in future_prophet_df.iterrows():
        current_date_str = row['ds'].strftime('%Y-%m-%d')
        matching_weather_input = next((item for item in future_weather_inputs if item['date'] == current_date_str), None)
        if matching_weather_input:
            chosen_weather = matching_weather_input['weather']
            col_name = f'weather_{chosen_weather}'
            if col_name in future_prophet_df.columns:
                future_prophet_df.loc[i, col_name] = 1 # Set the corresponding weather dummy to 1

    # Predict using the trained Prophet models
    forecast_sales = sales_model.predict(future_prophet_df)
    forecast_customers = customers_model.predict(future_prophet_df)

    # Combine Prophet's forecast output into a user-friendly DataFrame
    forecast_df = pd.DataFrame({
        'Date': forecast_sales['ds'].dt.strftime('%Y-%m-%d'),
        'Forecasted Sales': forecast_sales['yhat'].apply(lambda x: max(0, round(x, 2))),
        'Sales Lower Bound (95%)': forecast_sales['yhat_lower'].apply(lambda x: max(0, round(x, 2))),
        'Sales Upper Bound (95%)': forecast_sales['yhat_upper'].apply(lambda x: max(0, round(x, 2))),
        'Forecasted Customers': forecast_customers['yhat'].apply(lambda x: max(0, round(x))),
        'Customers Lower Bound (95%)': forecast_customers['yhat_lower'].apply(lambda x: max(0, round(x))),
        'Customers Upper Bound (95%)': forecast_customers['yhat_upper'].apply(lambda x: max(0, round(x))),
    })
    
    # Add the assumed weather back for display purposes
    forecast_df['Weather'] = [next((item['weather'] for item in future_weather_inputs if item['date'] == date_str), 'Sunny') for date_str in forecast_df['Date']]

    return forecast_df

# --- Streamlit UI Layout and Logic ---
st.set_page_config(layout="wide", page_title="AI Sales & Customer Forecast App")

st.title("🎯 AI Sales & Customer Forecast Analyst")
st.markdown("Your 200 IQ analyst for daily sales and customer volume forecasting!")

# --- Initialize Streamlit Session State ---
# This ensures variables persist across reruns and are initialized only once
if 'app_initialized' not in st.session_state:
    st.session_state['sales_data'] = load_sales_data()
    st.session_state['events_data'] = load_events_data()
    st.session_state['sales_model'] = None
    st.session_state['customers_model'] = None
    st.session_state['model_type'] = "RandomForest" # Default model selection
    st.session_state['rf_n_estimators'] = 100 # Default hyperparameter for RandomForest
    st.session_state['future_weather_inputs'] = [] # Stores user's input for future weather
    st.session_state['app_initialized'] = False # Flag to control initial setup logic

# --- Initial Sample Data Creation (Runs only once if data files are empty) ---
# This function is called only on the very first run or after data files are manually deleted.
if not st.session_state.get('app_initialized', False):
    def create_sample_data_if_empty_and_initialize_state():
        # Check if sales data file is empty
        sales_df_check = load_sales_data()
        events_df_check = load_events_data()

        if sales_df_check.empty:
            st.info("Creating sample sales data for a quick start...")
            # Generate 60 days of sample data ending yesterday
            start_date_for_sample = datetime.now() - timedelta(days=60)
            dates = pd.to_datetime(pd.date_range(start=start_date_for_sample, periods=60, freq='D'))
            np.random.seed(42) # For reproducibility of sample data
            sales = np.random.randint(500, 1500, size=len(dates)) + np.random.randn(len(dates)) * 50
            customers = np.random.randint(50, 200, size=len(dates)) + np.random.randn(len(dates)) * 10
            add_on_sales = np.random.randint(0, 100, size=len(dates))
            weather_choices = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
            weather = np.random.choice(weather_choices, size=len(dates), p=[0.5, 0.3, 0.15, 0.05])

            # Introduce some simple patterns for more realistic sample data
            for i, date in enumerate(dates):
                if date.weekday() >= 5: # Increase sales/customers on weekends
                    sales[i] = sales[i] * 1.2
                    customers[i] = customers[i] * 1.2
                if weather[i] == 'Rainy': # Decrease sales/customers on rainy days
                    sales[i] = sales[i] * 0.8
                    customers[i] = customers[i] * 0.8
                if weather[i] == 'Snowy': # Further decrease on snowy days
                    sales[i] = sales[i] * 0.7
                    customers[i] = customers[i] * 0.7

            sample_sales_df = pd.DataFrame({
                'Date': dates,
                'Sales': sales.round(2),
                'Customers': customers.round().astype(int),
                'Add_on_Sales': add_on_sales.round(2),
                'Weather': weather
            })
            save_sales_data(sample_sales_df) # Save to disk
            st.session_state.sales_data = load_sales_data() # Update session state from the saved (clean) data
            
        if events_df_check.empty:
            st.info("Creating sample event data...")
            sample_events_df = pd.DataFrame([
                {'Event_Date': pd.to_datetime('2024-06-20'), 'Event_Name': 'Annual Fair', 'Impact': 'High'},
                {'Event_Date': pd.to_datetime('2023-12-25'), 'Event_Name': 'Christmas Day', 'Impact': 'High'},
                {'Event_Date': pd.to_datetime('2024-03-15'), 'Event_Name': 'Spring Festival', 'Impact': 'Medium'},
                {'Event_Date': pd.to_datetime('2025-06-27'), 'Event_Name': 'Charter Day 2025 (Future)', 'Impact': 'High'},
                {'Event_Date': pd.to_datetime('2025-07-04'), 'Event_Name': 'Independence Day (Future)', 'Impact': 'Medium'},
                {'Event_Date': pd.to_datetime('2024-07-04'), 'Event_Name': 'Independence Day 2024', 'Impact': 'Medium'},
            ])
            save_events_data(sample_events_df) # Save to disk
            st.session_state.events_data = load_events_data() # Update session state from the saved (clean) data
            st.success("Sample data created!")
            
        st.session_state['app_initialized'] = True # Mark initialization as complete
        st.experimental_rerun() # Rerun the app to display updated state and models

    create_sample_data_if_empty_and_initialize_state()


# --- Sidebar for Model Settings and Event Logger ---
st.sidebar.header("🛠️ Model Settings")
model_type_selection = st.sidebar.selectbox(
    "Select AI Model:",
    ["RandomForest", "Prophet"],
    index=0 if st.session_state.model_type == "RandomForest" else 1,
    key='model_type_selector', # Unique key for this widget
    help="RandomForest is versatile. Prophet is good for time series with strong seasonality and holidays."
)
if model_type_selection != st.session_state.model_type:
    st.session_state.model_type = model_type_selection
    st.session_state.sales_model = None # Reset models when model type changes to force retraining
    st.session_state.customers_model = None
    st.experimental_rerun() # Rerun to apply model type change and trigger model loading/training

if st.session_state.model_type == "RandomForest":
    rf_n_estimators = st.sidebar.slider(
        "RandomForest n_estimators:",
        min_value=50, max_value=500, value=st.session_state.rf_n_estimators, step=50,
        key='rf_n_estimators_slider', # Unique key
        help="Number of trees in the forest. Higher values increase accuracy but also computation time."
    )
    if rf_n_estimators != st.session_state.rf_n_estimators:
        st.session_state.rf_n_estimators = rf_n_estimators
        st.session_state.sales_model = None # Reset model to re-train with new hyperparameter
        st.session_state.customers_model = None
        st.experimental_rerun()

st.sidebar.header("🗓️ Event Logger (Past & Future)")
with st.sidebar.form("event_input_form", clear_on_submit=True): # Clear form after submission
    st.subheader("Add Historical/Future Event")
    event_date = st.date_input("Event Date", datetime.now() - timedelta(days=365), key='sidebar_event_date_input')
    event_name = st.text_input("Event Name (e.g., Charter Day, Fiesta)", max_chars=100, key='sidebar_event_name_input')
    event_impact = st.selectbox("Impact", ['Low', 'Medium', 'High'], key='sidebar_event_impact_select')
    add_event_button = st.form_submit_button("Add Event")

    if add_event_button:
        new_event_df = pd.DataFrame([{
            'Event_Date': pd.to_datetime(event_date),
            'Event_Name': event_name,
            'Impact': event_impact
        }])
        st.session_state.events_data = pd.concat([st.session_state.events_data, new_event_df], ignore_index=True)
        save_events_data(st.session_state.events_data) # Save to disk
        st.session_state.events_data = load_events_data() # Reload clean data into session state
        st.sidebar.success(f"Event '{event_name}' added! AI will retrain.")
        st.session_state.sales_model = None # Force retraining of ML models
        st.session_state.customers_model = None
        st.experimental_rerun() # Rerun to reflect changes and retrain models


st.sidebar.subheader("Logged Events")
if not st.session_state.events_data.empty:
    display_events_df = st.session_state.events_data.sort_values('Event_Date', ascending=False).copy()
    display_events_df['Event_Date'] = display_events_df['Event_Date'].dt.strftime('%Y-%m-%d')
    st.sidebar.dataframe(display_events_df, use_container_width=True)

    # Delete Event functionality
    event_dates_to_delete = st.sidebar.multiselect(
        "Select events to delete:",
        st.session_state.events_data['Event_Date'].dt.strftime('%Y-%m-%d').tolist(),
        key='event_delete_multiselect'
    )
    if st.sidebar.button("Delete Selected Events", key='delete_event_btn'):
        if event_dates_to_delete:
            dates_to_delete_dt = [datetime.strptime(d, '%Y-%m-%d').date() for d in event_dates_to_delete]
            st.session_state.events_data = st.session_state.events_data[
                ~st.session_state.events_data['Event_Date'].dt.date.isin(dates_to_delete_dt)
            ].reset_index(drop=True)
            save_events_data(st.session_state.events_data) # Save changes to disk
            st.session_state.events_data = load_events_data() # Reload clean data
            st.sidebar.success("Selected events deleted! AI will retrain.")
            st.session_state.sales_model = None # Force retraining
            st.session_state.customers_model = None
            st.experimental_rerun()
        else:
            st.sidebar.warning("No events selected for deletion.")
else:
    st.sidebar.info("No events logged yet.")


# --- Model Loading and Training (runs on most reruns, but uses caching) ---
# Display spinner and messages for model loading/training
if st.session_state.sales_data.shape[0] > 1: # Only attempt if there's enough data for basic training
    with st.spinner(f"Loading/Training AI models ({st.session_state.model_type})... This happens after data changes."):
        try:
            st.session_state.sales_model, st.session_state.customers_model = load_or_train_models(
                st.session_state.model_type, st.session_state.rf_n_estimators
            )
        except Exception as e:
            st.error(f"An unexpected error occurred during model loading/training: {e}")
            st.warning("Please ensure you have enough sales data (at least 2 days) and correct dependencies.")
else:
    st.info("Add more sales records (at least 2 days) to enable AI model training and forecasting. Model training will start automatically once data is sufficient.")


# --- Main Application Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Daily Sales Input", "📈 10-Day Forecast", "📊 Forecast Accuracy Tracking"])

with tab1:
    st.header("Smart Data Input System")
    st.markdown("Enter daily sales and customer data. The AI will learn from these inputs.")

    with st.form("daily_input_form", clear_on_submit=True): # Clear form after submission
        st.subheader("Add New Daily Record")
        col1, col2, col3 = st.columns(3)
        with col1:
            # Default date for new input form
            input_date = st.date_input("Date", datetime.now(), key='daily_input_date_picker')
        with col2:
            sales = st.number_input("Sales", min_value=0.0, format="%.2f", key='daily_sales_input')
        with col3:
            customers = st.number_input("Number of Customers", min_value=0, step=1, key='daily_customers_input')

        col4, col5 = st.columns(2)
        with col4:
            add_on_sales = st.number_input("Add-on Sales (e.g., birthdays, bulk)", min_value=0.0, format="%.2f", key='daily_addon_sales_input')
        with col5:
            weather = st.selectbox("Weather", ['Sunny', 'Cloudy', 'Rainy', 'Snowy'], key='daily_weather_select')

        add_record_button = st.form_submit_button("Add Record")

        if add_record_button:
            # Convert input date to datetime for comparison
            input_date_dt = pd.to_datetime(input_date)
            # Check for duplicate date before adding
            if input_date_dt in st.session_state.sales_data['Date'].values:
                st.warning(f"Data for {input_date.strftime('%Y-%m-%d')} already exists. Please edit the existing record or choose a different date.")
            else:
                new_record = pd.DataFrame([{
                    'Date': input_date_dt,
                    'Sales': sales,
                    'Customers': customers,
                    'Add_on_Sales': add_on_sales,
                    'Weather': weather
                }])
                st.session_state.sales_data = pd.concat([st.session_state.sales_data, new_record], ignore_index=True)
                save_sales_data(st.session_state.sales_data) # Save to disk
                st.session_state.sales_data = load_sales_data() # Reload clean data into session state
                st.success("Record added successfully! AI will retrain automatically.")
                st.session_state.sales_model = None # Invalidate models to force retraining
                st.session_state.customers_model = None
                st.experimental_rerun() # Rerun to update UI and trigger model loading


    st.subheader("Last 7 Days of Inputs")
    if not st.session_state.sales_data.empty:
        # Display the latest 7 unique and sorted records
        display_data = st.session_state.sales_data.sort_values('Date', ascending=False).drop_duplicates(subset=['Date'], keep='first').head(7).copy()
        display_data['Date'] = display_data['Date'].dt.strftime('%Y-%m-%d')
        st.dataframe(display_data, use_container_width=True)
    else:
        st.info("No sales data entered yet. Add records above to see them here.")
        
    st.subheader("Edit/Delete Records")
    # Conditional block for Edit/Delete based on whether sales data exists
    if not st.session_state.sales_data.empty:
        # Prepare the list of unique and sorted dates for the selectbox options
        unique_dates_for_selectbox = sorted(st.session_state.sales_data['Date'].dt.strftime('%Y-%m-%d').unique().tolist(), reverse=True)
        
        # Streamlit selectbox to choose a record for editing/deleting
        selected_date_str = st.selectbox(
            "Select a record by Date for editing or deleting:",
            options=unique_dates_for_selectbox, # Use 'options' argument for clarity
            key='edit_delete_selector'
        )

        # Retrieve the selected row based on the chosen date string
        # This is safe because 'selected_date_str' will always be one of the dates present in 'st.session_state.sales_data'
        selected_row_df = st.session_state.sales_data[
            st.session_state.sales_data['Date'] == pd.to_datetime(selected_date_str)
        ]
        selected_row = selected_row_df.iloc[0] # Access the single matching row

        st.markdown(f"**Selected Record for {selected_date_str}:**")
        
        with st.form("edit_delete_form", clear_on_submit=False): # Form for editing/deleting
            # Populate form fields with data from the selected row
            edit_sales = st.number_input("Edit Sales", value=float(selected_row['Sales']), format="%.2f", key='edit_sales_input')
            edit_customers = st.number_input("Edit Customers", value=int(selected_row['Customers']), step=1, key='edit_customers_input')
            edit_add_on_sales = st.number_input("Edit Add-on Sales", value=float(selected_row['Add_on_Sales']), format="%.2f", key='edit_add_on_sales_input')
            
            weather_options = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
            # Safely determine the default index for the weather selectbox
            try:
                default_weather_index = weather_options.index(selected_row['Weather'])
            except ValueError:
                default_weather_index = 0 # Fallback to 'Sunny' if weather value is unexpected or not found
            edit_weather = st.selectbox("Edit Weather", weather_options, index=default_weather_index, key='edit_weather_select')

            col_edit_del_btns1, col_edit_del_btns2 = st.columns(2)
            with col_edit_del_btns1:
                update_button = st.form_submit_button("Update Record")
            with col_edit_del_btns2:
                delete_button = st.form_submit_button("Delete Record")

            if update_button:
                # Logic to update the record in session state data
                st.session_state.sales_data.loc[
                    st.session_state.sales_data['Date'] == pd.to_datetime(selected_date_str),
                    ['Sales', 'Customers', 'Add_on_Sales', 'Weather']
                ] = [edit_sales, edit_customers, edit_add_on_sales, edit_weather]
                save_sales_data(st.session_state.sales_data) # Save updated data to disk
                st.session_state.sales_data = load_sales_data() # Reload clean data into session state
                st.success("Record updated successfully! AI will retrain.")
                st.session_state.sales_model = None # Invalidate models to force retraining
                st.session_state.customers_model = None
                st.experimental_rerun() # Rerun to display updated data and trigger model retraining
            elif delete_button:
                # Logic to delete the record from session state data
                st.session_state.sales_data = st.session_state.sales_data[
                    st.session_state.sales_data['Date'] != pd.to_datetime(selected_date_str)
                ].reset_index(drop=True)
                save_sales_data(st.session_state.sales_data) # Save changes to disk
                st.session_state.sales_data = load_sales_data() # Reload clean data
                st.success("Record deleted successfully! AI will retrain.")
                st.session_state.sales_model = None # Invalidate models
                st.session_state.customers_model = None
                st.experimental_rerun() # Rerun to update UI and trigger model retraining
    else: # This 'else' now clearly pairs with 'if not st.session_state.sales_data.empty:' for this entire section
        st.info("No sales data to edit or delete yet. Please add records first via the 'Add New Daily Record' section.")


with tab2:
    st.header("10-Day Sales & Customer Forecast")
    st.markdown("View the AI's predictions for the next 10 days.")

    st.subheader("Future Weather Forecast (Next 10 Days)")
    st.markdown("Specify the expected weather for each forecast day. Default is 'Sunny'.")
    
    # Generate the next 10 forecast dates
    forecast_dates_for_weather = [(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(10)]
    weather_options = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']

    # Initialize future_weather_inputs in session state if not already populated or out of sync
    if not st.session_state.future_weather_inputs or \
       any(item['date'] not in forecast_dates_for_weather for item in st.session_state.future_weather_inputs) or \
       len(st.session_state.future_weather_inputs) != 10:
        st.session_state.future_weather_inputs = [{'date': d, 'weather': 'Sunny'} for d in forecast_dates_for_weather]

    # Display and allow editing of future weather for each day
    weather_inputs_edited = []
    for i, item in enumerate(st.session_state.future_weather_inputs):
        col_w1, col_w2 = st.columns([1, 2])
        with col_w1:
            st.write(f"**Day {i+1}: {item['date']}**")
        with col_w2:
            selected_weather = st.selectbox(
                "Weather",
                weather_options,
                index=weather_options.index(item['weather']),
                key=f"future_weather_select_{item['date']}" # Unique key for each selectbox
            )
            weather_inputs_edited.append({'date': item['date'], 'weather': selected_weather})
    
    st.session_state.future_weather_inputs = weather_inputs_edited # Update session state with edited weather

    if st.button("Generate 10-Day Forecast", key='generate_forecast_btn'):
        # Check for sufficient data before attempting forecast
        if st.session_state.sales_data.empty or st.session_state.sales_data.shape[0] < 2:
            st.warning("Please enter at least 2 days of sales data to generate a meaningful forecast.")
        elif st.session_state.sales_model is None or st.session_state.customers_model is None:
             st.warning("AI models are not yet trained or loaded. Please ensure you have sufficient data and select a model type in the sidebar.")
        else:
            with st.spinner(f"Generating forecast using {st.session_state.model_type}... This might take a moment as the AI thinks ahead!"):
                if st.session_state.model_type == "RandomForest":
                    forecast_df = generate_rf_forecast(
                        st.session_state.sales_data,
                        st.session_state.events_data,
                        st.session_state.sales_model,
                        st.session_state.customers_model,
                        st.session_state.future_weather_inputs
                    )
                elif st.session_state.model_type == "Prophet":
                    forecast_df = generate_prophet_forecast(
                        st.session_state.sales_data,
                        st.session_state.events_data,
                        st.session_state.sales_model,
                        st.session_state.customers_model,
                        st.session_state.future_weather_inputs
                    )
                st.session_state.forecast_df = forecast_df # Store generated forecast in session state
                st.success("Forecast generated!")
    
    # Display forecast if available
    if 'forecast_df' in st.session_state and not st.session_state.forecast_df.empty:
        st.subheader("Forecasted Data (with 95% Confidence Interval)")
        st.dataframe(st.session_state.forecast_df, use_container_width=True)

        # Download button for forecast data
        csv = st.session_state.forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast as CSV",
            data=csv,
            file_name="sales_customer_forecast.csv",
            mime="text/csv",
        )

        st.subheader("Forecast Visualization")
        # Prepare data for plotting: Historical actuals + Forecasts with confidence intervals
        historical_df_plot = st.session_state.sales_data.copy()
        historical_df_plot['Date'] = pd.to_datetime(historical_df_plot['Date'])

        forecast_df_plot = st.session_state.forecast_df.copy()
        forecast_df_plot['Date'] = pd.to_datetime(forecast_df_plot['Date'])

        # Plot Sales: Actual vs. Forecast
        fig_sales, ax_sales = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=historical_df_plot, x='Date', y='Sales', label='Actual Sales', marker='o', ax=ax_sales, color='blue')
        sns.lineplot(data=forecast_df_plot, x='Date', y='Forecasted Sales', label='Forecasted Sales', marker='x', linestyle='--', ax=ax_sales, color='red')
        
        ax_sales.fill_between(forecast_df_plot['Date'], forecast_df_plot['Sales Lower Bound (95%)'], forecast_df_plot['Sales Upper Bound (95%)'], color='red', alpha=0.2, label='95% Confidence Interval')

        ax_sales.set_title(f'Sales: Actual vs. Forecast ({st.session_state.model_type} Model)')
        ax_sales.set_xlabel('Date')
        ax_sales.set_ylabel('Sales')
        ax_sales.legend()
        ax_sales.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_sales)

        # Plot Customers: Actual vs. Forecast
        fig_customers, ax_customers = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=historical_df_plot, x='Date', y='Customers', label='Actual Customers', marker='o', ax=ax_customers, color='blue')
        sns.lineplot(data=forecast_df_plot, x='Date', y='Forecasted Customers', label='Forecasted Customers', marker='x', linestyle='--', ax=ax_customers, color='red')
        
        ax_customers.fill_between(forecast_df_plot['Date'], forecast_df_plot['Customers Lower Bound (95%)'], forecast_df_plot['Customers Upper Bound (95%)'], color='red', alpha=0.2, label='95% Confidence Interval')

        ax_customers.set_title(f'Customers: Actual vs. Forecast ({st.session_state.model_type} Model)')
        ax_customers.set_xlabel('Date')
        ax_customers.set_ylabel('Customers')
        ax_customers.legend()
        ax_customers.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_customers)
    else:
        st.info("Click 'Generate 10-Day Forecast' to see predictions.")


with tab3:
    st.header("Forecast Accuracy Tracking")
    st.markdown("Compare past forecasts with actual data to track AI performance.")

    if st.button("Calculate Accuracy", key='calculate_accuracy_btn'):
        # Ensure sufficient data and models are ready
        if st.session_state.sales_data.empty:
            st.warning("No sales data available to calculate accuracy.")
        elif st.session_state.sales_data.shape[0] < 2:
            st.warning("Please enter at least 2 days of sales data to calculate accuracy.")
        elif st.session_state.sales_model is None or st.session_state.customers_model is None:
             st.warning("AI models are not yet trained or loaded. Please ensure you have sufficient data and select a model type in the sidebar.")
        else:
            with st.spinner(f"Calculating accuracy using {st.session_state.model_type}... This may take a moment."):
                if st.session_state.model_type == "RandomForest":
                    X_hist, y_sales_hist, y_customers_hist, _ = preprocess_rf_data(st.session_state.sales_data, st.session_state.events_data)
                    
                    if not X_hist.empty and X_hist.shape[0] > 1:
                        # Re-predict on historical data to evaluate accuracy
                        predicted_sales_hist = st.session_state.sales_model.predict(X_hist)
                        predicted_customers_hist = st.session_state.customers_model.predict(X_hist)

                        accuracy_plot_df = pd.DataFrame({
                            'Date': st.session_state.sales_data['Date'].iloc[X_hist.index],
                            'Actual Sales': y_sales_hist,
                            'Predicted Sales': predicted_sales_hist,
                            'Actual Customers': predicted_customers_hist
                        })

                        mae_sales = mean_absolute_error(accuracy_plot_df['Actual Sales'], accuracy_plot_df['Predicted Sales'])
                        r2_sales = r2_score(accuracy_plot_df['Actual Sales'], accuracy_plot_df['Predicted Sales'])

                        mae_customers = mean_absolute_error(accuracy_plot_df['Actual Customers'], accuracy_plot_df['Predicted Customers'])
                        r2_customers = r2_score(accuracy_plot_df['Actual Customers'], accuracy_plot_df['Predicted Customers'])
                        
                        st.subheader(f"Overall Model Accuracy ({st.session_state.model_type})")
                        st.write(f"**Sales MAE (Mean Absolute Error):** {mae_sales:.2f}")
                        st.write(f"**Sales R² Score:** {r2_sales:.2f}")
                        st.write(f"**Customers MAE (Mean Absolute Error):** {mae_customers:.2f}")
                        st.write(f"**Customers R² Score:** {r2_customers:.2f}")
                        st.info("An R² score closer to 1 indicates a better fit. MAE shows average error in the units of the target variable.")

                        # Plotting Actual vs. Predicted for Sales
                        fig_acc_sales, ax_acc_sales = plt.subplots(figsize=(12, 6))
                        sns.lineplot(data=accuracy_plot_df, x='Date', y='Actual Sales', label='Actual Sales', marker='o', ax=ax_acc_sales)
                        sns.lineplot(data=accuracy_plot_df, x='Date', y='Predicted Sales', label='Predicted Sales', marker='x', linestyle='--', ax=ax_acc_sales)
                        ax_acc_sales.set_title(f'Historical Sales: Actual vs. Predicted ({st.session_state.model_type})')
                        ax_acc_sales.set_xlabel('Date')
                        ax_acc_sales.set_ylabel('Sales')
                        ax_acc_sales.legend()
                        ax_acc_sales.grid(True)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig_acc_sales)

                        # Plotting Actual vs. Predicted for Customers
                        fig_acc_customers, ax_acc_customers = plt.subplots(figsize=(12, 6))
                        sns.lineplot(data=accuracy_plot_df, x='Date', y='Actual Customers', label='Actual Customers', marker='o', ax=ax_acc_customers)
                        sns.lineplot(data=accuracy_plot_df, x='Date', y='Predicted Customers', label='Predicted Customers', marker='x', linestyle='--', ax=ax_acc_customers)
                        ax_acc_customers.set_title(f'Historical Customers: Actual vs. Predicted ({st.session_state.model_type})')
                        ax_acc_customers.set_xlabel('Date')
                        ax_acc_customers.set_ylabel('Customers')
                        ax_acc_customers.legend()
                        ax_acc_customers.grid(True)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig_acc_customers)
                    else:
                        st.warning("Not enough data points after preprocessing for accuracy calculation. Please ensure you have at least 2 sales records.")

                elif st.session_state.model_type == "Prophet":
                    # Prophet cross-validation for robust accuracy assessment
                    if st.session_state.sales_data.shape[0] < 30: # Prophet needs a good amount of data for meaningful CV
                        st.warning("Prophet cross-validation requires at least 30 days of historical data for meaningful results. Please add more records.")
                    else:
                        st.info("Running Prophet cross-validation. This might take a while for larger datasets.")
                        
                        sales_prophet_df_cv, holidays_df_cv = preprocess_prophet_data(st.session_state.sales_data, st.session_state.events_data, 'Sales')
                        customers_prophet_df_cv, _ = preprocess_prophet_data(st.session_state.sales_data, st.session_state.events_data, 'Customers')

                        if sales_prophet_df_cv.empty or customers_prophet_df_cv.empty or len(sales_prophet_df_cv) < 2:
                            st.warning("Prophet preprocessed data is empty or insufficient. Cannot run cross-validation.")
                        else:
                            try:
                                with st.spinner("Performing cross-validation for Sales model..."):
                                    df_cv_sales = cross_validation(
                                        st.session_state.sales_model, initial='30 days', period='15 days', horizon='10 days'
                                    )
                                with st.spinner("Calculating performance metrics for Sales..."):
                                    df_p_sales = performance_metrics(df_cv_sales)
                                
                                with st.spinner("Performing cross-validation for Customers model..."):
                                    df_cv_customers = cross_validation(
                                        st.session_state.customers_model, initial='30 days', period='15 days', horizon='10 days'
                                    )
                                with st.spinner("Calculating performance metrics for Customers..."):
                                    df_p_customers = performance_metrics(df_cv_customers)

                                st.subheader(f"Prophet Model Performance Metrics (Cross-Validation)")
                                st.write("Sales Model Performance:")
                                st.dataframe(df_p_sales.head(), use_container_width=True)
                                st.write("Customers Model Performance:")
                                st.dataframe(df_p_customers.head(), use_container_width=True)
                                st.info("Metrics are calculated over various forecast horizons. MAE and RMSE are typically desired to be lower.")

                                # Plot Prophet cross-validation metrics
                                fig_sales_rmse = plot_cross_validation_metric(df_cv_sales, metric='rmse')
                                fig_sales_mae = plot_cross_validation_metric(df_cv_sales, metric='mae')
                                fig_customers_rmse = plot_cross_validation_metric(df_cv_customers, metric='rmse')
                                fig_customers_mae = plot_cross_validation_metric(df_cv_customers, metric='mae')

                                fig_sales_rmse.update_layout(title_text='Sales: RMSE vs. Horizon')
                                fig_sales_mae.update_layout(title_text='Sales: MAE vs. Horizon')
                                fig_customers_rmse.update_layout(title_text='Customers: RMSE vs. Horizon')
                                fig_customers_mae.update_layout(title_text='Customers: MAE vs. Horizon')

                                st.subheader("Prophet Cross-Validation Plots")
                                st.write("Sales RMSE:")
                                st.pyplot(fig_sales_rmse)
                                st.write("Sales MAE:")
                                st.pyplot(fig_sales_mae)
                                st.write("Customers RMSE:")
                                st.pyplot(fig_customers_rmse)
                                st.write("Customers MAE:")
                                st.pyplot(fig_customers_mae)

                            except Exception as e:
                                st.error(f"Error during Prophet cross-validation: {e}. Ensure sufficient data and model setup.")
        else:
            st.error("AI models are not ready. Please ensure you have sufficient data and the models are trained first.")
    else:
        st.info("Click 'Calculate Accuracy' to see how well the AI performs on past data.")
