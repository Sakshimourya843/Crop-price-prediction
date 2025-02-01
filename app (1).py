# import joblib
# import pandas as pd
# import streamlit as st
# from statsmodels.tsa.arima.model import ARIMA
# import datetime

# # Load your dataset  "C:\Users\khyat\Documents\cropPricePred"
# data = pd.read_csv('C:/Users/khyat/Downloads/Crop Price Datasetttttttt.csv')  # Replace with your dataset path

# # Preprocess data (melting the dataframe)
# data_melted = pd.melt(data, id_vars=['Crop Type', 'State'], var_name='Month-Year', value_name='Price')

# # Remove rows with "Average" in the 'Month-Year' column
# data_melted = data_melted[data_melted['Month-Year'] != 'Average']

# data_melted['Month-Year'] = pd.to_datetime(data_melted['Month-Year'], format='%b-%y')
# data_melted = data_melted.sort_values(by=['Crop Type', 'State', 'Month-Year'])

# # Load the pre-trained ARIMA model
# loaded_model = joblib.load('C:/Users/khyat/Downloads/crop_pricce_model.pkl')

# # Function to get the price for the current year and future year
# def forecast_prices(crop, state, months_ahead):
#     # Filter data for the selected crop and state
#     crop_state_data = data_melted[(data_melted['Crop Type'] == crop) & (data_melted['State'] == state)]
#     crop_state_data.set_index('Month-Year', inplace=True)

#     # Handle case where months_ahead is 0 (i.e., return the latest price)
#     if months_ahead == 0:
#         return [crop_state_data['Price'].iloc[-1]]  # Return the most recent price in the data
    
#     # Fit ARIMA model using the specific crop-state data
#     model = ARIMA(crop_state_data['Price'], order=(5, 1, 0))  # Example ARIMA parameters
#     model_fit = model.fit()

#     # Forecast future prices for the selected number of months
#     forecast = model_fit.forecast(steps=months_ahead)

#     return forecast.tolist()


# # Streamlit app setup
# st.title("Crop Price Prediction for Farmers")

# # User inputs for crop, state, month, current year, and future year
# crop = st.selectbox("Select Crop Type", ["Wheat", "Rice", "Gram Dal", "Tur/Arhar", "Urad", "Moong", "Masoor", "Potato", "Onion", "Tomato"])
# state = st.selectbox("Select State", ["Karnataka", "Punjab", "Maharashtra", "Andaman and Nicobar", "Andhra Pradesh", "Arunachal pradesh", "Assam", "Bihar", "Chandigarh", "Chhattisgarh", "Delhi", "Goa", "Gujrat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Puducherry", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"])
# month = st.selectbox("Select Month", list(range(1, 13)))
# current_year = st.slider("Select Current Year", 2014, 2020, 2023)
# future_year = st.slider("Select Future Year", 2024, 2030, 2025)

# # Function to display both current price and predicted price
# def predict_interface(crop, state, month, current_year, future_year):
#     # Current date to calculate how many months ahead we need to forecast
#     selected_month = datetime.datetime.now().replace(month=month, year=current_year, day=1)

#     # Forecast for the current year price (0 months ahead means the current price)
#     current_price = forecast_prices(crop, state, 0)[-1]  # Using the last value of forecast for current year

#     # Calculate the number of months between current date and selected future date
#     future_date = datetime.datetime(future_year, month, 1)
#     months_ahead = (future_date.year - selected_month.year) * 12 + (future_date.month - selected_month.month)

#     if months_ahead < 0:
#         return "Please select a future date."

#     # Get the forecasted prices for the future year
#     future_price = forecast_prices(crop, state, months_ahead)[-1]

#     # Return both current year price and predicted future year price
#     return f"Current price for {crop} in {state} for {selected_month.strftime('%B, %Y')} is {current_price:.2f}\n" \
#            f"Predicted price for {crop} in {state} for {future_date.strftime('%B, %Y')} is {future_price:.2f}"

# # When the user clicks the "Predict" button, show the prediction
# if st.button("Predict"):
#     result = predict_interface(crop, state, month, current_year, future_year)
#     st.write(result)


# Import necessary libraries


import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the Random Forest model
model_path = 'C:/Users/khyat/OneDrive/文档/cropPricePred/crop_prediction_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load and preprocess the dataset with encoding specified
dataset_path = 'C:/Users/khyat/OneDrive/文档/cropPricePred/reshaped_data.csv'
try:
    df = pd.read_csv(dataset_path, encoding='latin-1')  # Try 'latin1' encoding
except UnicodeDecodeError:
    st.error("Error decoding the CSV file. Please check the file encoding.")
    st.stop()

# Data Preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Encoding categorical features
label_encoder = LabelEncoder()
df['Crop Type'] = label_encoder.fit_transform(df['Crop Type'])
df['State'] = label_encoder.fit_transform(df['State'])

# Features and target variable
X = df[['Crop Type', 'State', 'Year', 'Month', 'Day', 'Average Temperature (°C)', 'Average Humidity (%)']]
y = df['Value']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Streamlit app title
st.title("Crop Price Prediction with Environmental Factors")

# User inputs for crop type, state, month, temperature, humidity, current year, and future year
crop = st.selectbox("Select Crop Type", ["Wheat", "Rice", "Gram Dal", "Tur/Arhar", "Urad", "Moong", "Masoor", "Potato", "Onion", "Tomato"])
state = st.selectbox("Select State", ["Karnataka", "Punjab", "Maharashtra", "Andaman and Nicobar", "Andhra Pradesh", "Arunachal Pradesh", 
                                      "Assam", "Bihar", "Chandigarh", "Chhattisgarh", "Delhi", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", 
                                      "Jammu and Kashmir", "Jharkhand", "Kerala", "Madhya Pradesh", "Manipur", "Meghalaya", "Mizoram", "Nagaland", 
                                      "Odisha", "Puducherry", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", 
                                      "Uttarakhand", "West Bengal"])
month = st.selectbox("Select Month", list(range(1, 13)))

temperature = st.number_input("Average Temperature (°C)", step=0.1)
humidity = st.number_input("Average Humidity (%)", step=0.1)

current_year = st.number_input("Enter Current Year", min_value=2014, max_value=2024, value=2023)
future_year = st.number_input("Enter Future Year", min_value=2024, value=2024)

# Function to prepare input data for prediction
def prepare_input_data(crop, state, month, temperature, humidity, year):
    crop_encoded = label_encoder.transform([crop])[0]
    state_encoded = label_encoder.transform([state])[0]
    input_data = pd.DataFrame({
        'Crop Type': [crop_encoded],
        'State': [state_encoded],
        'Year': [year],
        'Month': [month],
        'Day': [1],  # Defaulting to the 1st day of the month
        'Average Temperature (°C)': [temperature],
        'Average Humidity (%)': [humidity]
    })
    return input_data

# Button to trigger prediction
if st.button("Predict"):
    # Prepare input data for current year and future year
    input_data_current = prepare_input_data(crop, state, month, temperature, humidity, current_year)
    input_data_future = prepare_input_data(crop, state, month, temperature, humidity, future_year)

    try:
        # Predict prices
        current_price = model.predict(input_data_current)[0]
        future_price = model.predict(input_data_future)[0]

        # Display the results
        st.success(f"Predicted price for {crop} in {state} for {month}/{future_year} with temp {temperature}°C and humidity {humidity}%: {future_price:.2f}")
        st.info(f"Current price for {crop} in {state} for {month}/{current_year} with temp {temperature}°C and humidity {humidity}%: {current_price:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Display model evaluation metrics
st.write("### Model Evaluation Metrics")
st.write(f"Root Mean Squared Error: {rmse:.2f}")
st.write(f"R^2 Score: {r2:.2f}")

# Display feature importances
feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

st.write("### Feature Importances")
st.write(importance_df)
