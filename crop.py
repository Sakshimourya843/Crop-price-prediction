from flask import Flask, render_template, request, send_file
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import datetime
import numpy as np
import pickle
import lightgbm as lgb
import logging

app = Flask(__name__)

# Load dataset and pre-trained models
data = pd.read_csv(r'C:\Users\khyat\OneDrive\文档\cropPricePred\reshaped_data.csv', encoding='ISO-8859-1')
data_melted = pd.melt(data, id_vars=['Crop Type', 'State'], var_name='Month-Year', value_name='Price')
data_melted = data_melted[data_melted['Month-Year'] != 'Average']
data_melted['Month-Year'] = pd.to_datetime(data_melted['Month-Year'], format='%b-%y', errors='coerce')
data_melted = data_melted.dropna(subset=['Month-Year'])
data_melted = data_melted.sort_values(by=['Crop Type', 'State', 'Month-Year'])

# Load models
def load_models():
    try:
        # Load LGBM model
        with open(r'C:\Users\khyat\OneDrive\文档\cropPricePred\cpp_lgbm.pkl', 'rb') as file:
            lgbm_model = pickle.load(file)
        
        # Load ARIMA model
        with open(r'C:\Users\khyat\OneDrive\文档\cropPricePred\arima_cpp.pkl', 'rb') as file:
            arima_models = pickle.load(file)
        
        # Load Random Forest model
        with open(r'C:\Users\khyat\OneDrive\文档\cropPricePred\Random_Forest.pkl', 'rb') as file:
            rf_model = pickle.load(file)
        
        return lgbm_model, arima_models, rf_model
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, None

lgbm_model, arima_models, rf_model = load_models()

# ARIMA forecast function
def forecast_arima(crop, state, months_ahead):
    try:
        crop_state_data = data_melted[(data_melted['Crop Type'] == crop) & 
                                      (data_melted['State'] == state)]
        crop_state_data.set_index('Month-Year', inplace=True)
        
        if months_ahead == 0:
            return crop_state_data['Price'].iloc[-1]
        
        model = ARIMA(crop_state_data['Price'], order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=months_ahead)
        return forecast[-1]
    except Exception as e:
        print(f"ARIMA Error: {str(e)}")
        return None

# LGBM prediction function
def predict_lgbm(model, crop, state, date):
    try:
        month, year = date.month, date.year
        avg_temp, avg_humidity = 25, 65  # Default values
        input_data = pd.DataFrame({
            'Crop Type': [crop],
            'State': [state],
            'Average Temperature (°C)': [avg_temp],
            'Average Humidity (%)': [avg_humidity],
            'Month': [month],
            'Year': [year]
        })
        
        input_data['Crop Type'] = model['le_crop'].transform(input_data['Crop Type'])
        input_data['State'] = model['le_state'].transform(input_data['State'])
        
        prediction = model['model'].predict(input_data)[0]
        return prediction
    except Exception as e:
        print(f"LGBM Error: {str(e)}")
        return None

# Random Forest prediction function
def predict_rf(model, crop, state, date):
    try:
        month, year = date.month, date.year
        avg_temp, avg_humidity = 25, 65  # Default values
        input_data = pd.DataFrame([[model['le_crop'].transform([crop])[0],
                                    model['le_state'].transform([state])[0],
                                    avg_temp, avg_humidity, month, year]], 
                                  columns=model['feature_columns'])
        
        prediction = model['model'].predict(input_data)[0]
        return prediction
    except Exception as e:
        print(f"RF Error: {str(e)}")
        return None

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            crop = request.form['crop']
            state = request.form['state']
            month = int(request.form['month'])
            current_year = int(request.form['current_year'])
            future_year = int(request.form['future_year'])
            
            selected_month = datetime.datetime(current_year, month, 1)
            future_date = datetime.datetime(future_year, month, 1)
            
            months_ahead = (future_date.year - selected_month.year) * 12 + (future_date.month - selected_month.month)
            
            if months_ahead < 0:
                return "Please select a future date."
            
            # Get predictions
            arima_prediction = forecast_arima(crop, state, months_ahead)
            lgbm_prediction = predict_lgbm(lgbm_model, crop, state, future_date)
            rf_prediction = predict_rf(rf_model, crop, state, future_date)
            
            valid_predictions = [p for p in [arima_prediction, lgbm_prediction, rf_prediction] if p is not None]
            average_prediction = np.mean(valid_predictions) if valid_predictions else None
            
            result = f"Predictions for {crop} in {state} for {future_date.strftime('%B, %Y')}:<br><br>"
            result += f"ARIMA Model: ₹{arima_prediction:.2f}/quintal<br>" if arima_prediction else "ARIMA Model: Prediction failed<br>"
            result += f"LGBM Model: ₹{lgbm_prediction:.2f}/quintal<br>" if lgbm_prediction else "LGBM Model: Prediction failed<br>"
            result += f"Random Forest Model: ₹{rf_prediction:.2f}/quintal<br>" if rf_prediction else "Random Forest Model: Prediction failed<br>"
            
            if average_prediction:
                result += f"<br>Average Prediction: ₹{average_prediction:.2f}/quintal"
            
            # Generate forecasted graph
            last_date = data_melted['Month-Year'].max()
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, months_ahead + 1)]
            
            # Prepare forecasts for graph
            arima_forecasts = []
            lgbm_forecasts = []
            rf_forecasts = []
            
            for future_date in future_dates:
                arima_pred = forecast_arima(crop, state, (future_date.year - last_date.year) * 12 + (future_date.month - last_date.month))
                arima_forecasts.append(arima_pred if arima_pred is not None else np.nan)
                
                lgbm_pred = predict_lgbm(lgbm_model, crop, state, future_date)
                lgbm_forecasts.append(lgbm_pred if lgbm_pred is not None else np.nan)
                
                rf_pred = predict_rf(rf_model, crop, state, future_date)
                rf_forecasts.append(rf_pred if rf_pred is not None else np.nan)
            
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'ARIMA': arima_forecasts,
                'LGBM': lgbm_forecasts,
                'Random Forest': rf_forecasts
            }).set_index('Date')
            
            # Plot and save graph
            plt.figure(figsize=(12, 6))
            forecast_df.plot(ax=plt.gca())
            plt.title(f"Forecasted Prices for {crop} in {state}")
            plt.xlabel('Date')
            plt.ylabel('Price (₹)')
            plt.legend(title='Model', loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            
            forecast_graph_path = 'static/forecasted_graph.png'
            plt.savefig(forecast_graph_path)
            plt.close()
            
            # Pass prediction and graph to the result page
            return render_template('result.html', result=result, graph_path=forecast_graph_path)
            
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    return render_template('predict.html')


# Configure logging
logging.basicConfig(
    filename='app.log',  # Log messages will be written to 'app.log'
    filemode='a',        # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Set log level to capture DEBUG and higher
)
@app.route('/forecasted_graph', methods=['POST'])
def forecasted_graph():
    try:
        # Get input parameters from user
        crop = request.form.get('crop')
        state = request.form.get('state')
        months_ahead = int(request.form.get('months_ahead', 12))  # Default to 12 months
        
        # Validate inputs
        if not crop or not state:
            return "Please provide both crop and state for forecasting."
        
        # Prepare future dates for the forecast
        last_date = data_melted['Month-Year'].max()
        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, months_ahead + 1)]
        
        # Generate forecasts from all models
        arima_forecasts = []
        lgbm_forecasts = []
        rf_forecasts = []
        
        for future_date in future_dates:
            # ARIMA
            arima_pred = forecast_arima(crop, state, (future_date.year - last_date.year) * 12 + (future_date.month - last_date.month))
            arima_forecasts.append(arima_pred if arima_pred is not None else np.nan)
            
            # LGBM
            lgbm_pred = predict_lgbm(lgbm_model, crop, state, future_date)
            lgbm_forecasts.append(lgbm_pred if lgbm_pred is not None else np.nan)
            
            # Random Forest
            rf_pred = predict_rf(rf_model, crop, state, future_date)
            rf_forecasts.append(rf_pred if rf_pred is not None else np.nan)
        
        # Combine predictions into a DataFrame
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'ARIMA': arima_forecasts,
            'LGBM': lgbm_forecasts,
            'Random Forest': rf_forecasts
        }).set_index('Date')
        
        # Plot the forecasted graph
        plt.figure(figsize=(12, 6))
        forecast_df.plot(ax=plt.gca())
        plt.title(f"Forecasted Prices for {crop} in {state}")
        plt.xlabel('Date')
        plt.ylabel('Price (₹)')
        plt.legend(title='Model', loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        # Save the graph to the static folder
        forecast_graph_path = 'static/forecasted_graph.png'
        plt.savefig(forecast_graph_path)
        plt.close()
        
        return send_file(forecast_graph_path, mimetype='image/png')
    
    except Exception as e:
        logging.error(f"Forecasted graph generation error: {str(e)}")
        return f"Error generating forecasted graph: {str(e)}"


@app.route('/help')
def help_page():
    return render_template('help.html')

import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    app.run(debug=True)



  
