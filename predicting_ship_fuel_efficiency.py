# -*- coding: utf-8 -*-
"""
Predicting ship fuel Efficiency

Original file is located at
    https://colab.research.google.com/drive/1rwhyqk8UDEsDbqqOfdc6ni53tXFKTt0q
"""

# -------------------------
# 1. Data Loading
# -------------------------
import pandas as pd

# Load the data
df = pd.read_csv('ship_fuel_efficiency.csv')
# Uncomment below if running in a notebook
# display(df.head())

# -------------------------
# 2. Data Preprocessing
# -------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Identify categorical and numerical columns, excluding 'ship_id' from features
categorical_features = ['ship_type', 'route_id', 'month', 'fuel_type', 'weather_conditions']
numerical_features = ['distance', 'fuel_consumption', 'CO2_emissions']

# Drop non-feature columns
X = df.drop(['engine_efficiency', 'ship_id'], axis=1)
y = df['engine_efficiency']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Fit and transform the features
X_processed = preprocessor.fit_transform(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

print("Data preprocessing complete. Shapes:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

# -------------------------
# 3. Model Training
# -------------------------
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
print("Linear Regression model trained successfully.")

# -------------------------
# 4. Model Evaluation
# -------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# -------------------------
# 5. Save Model and Preprocessor
# -------------------------
import joblib

joblib.dump(model, 'linear_regression_model.joblib')
joblib.dump(preprocessor, 'preprocessor.joblib')
print("Model and preprocessor saved successfully.")

# -------------------------
# 6. Streamlit App Code
# -------------------------
import streamlit as st

# Load model and preprocessor (for Streamlit app)
@st.cache_resource
def load_artifacts():
    model = joblib.load('linear_regression_model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    return model, preprocessor

model, preprocessor = load_artifacts()

st.title('Ship Fuel Efficiency Prediction')
st.write('Enter the ship parameters to predict the engine efficiency.')

# Define the features in order
features = [
    'ship_type', 'route_id', 'month', 'distance',
    'fuel_type', 'fuel_consumption', 'CO2_emissions', 'weather_conditions'
]

# Generate unique values for categorical fields from the data (for real deployment).
# Here, fallback to placeholder values if df is not available.
try:
    ship_types = sorted(df['ship_type'].unique())
    route_ids = sorted(df['route_id'].unique())
    months = sorted(df['month'].unique(), key=lambda m: pd.to_datetime(m, format='%B').month if isinstance(m, str) and len(m) > 2 else 0)
    fuel_types = sorted(df['fuel_type'].unique())
    weather_conditions = sorted(df['weather_conditions'].unique())
except Exception:
    # Fallback hardcoded values (should be replaced with actual data)
    ship_types = ['Oil Service Boat', 'Fishing Trawler', 'Cargo Ship', 'Tanker']
    route_ids = ['Warri-Bonny', 'Port Harcourt-Lagos', 'Lagos-Apapa']
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    fuel_types = ['HFO', 'Diesel']
    weather_conditions = ['Stormy', 'Moderate', 'Calm']

st.sidebar.header('Ship Parameters')

ship_type = st.sidebar.selectbox('Ship Type', ship_types)
route_id = st.sidebar.selectbox('Route ID', route_ids)
month = st.sidebar.selectbox('Month', months)
distance = st.sidebar.number_input('Distance (km)', min_value=0.0, value=100.0)
fuel_type = st.sidebar.selectbox('Fuel Type', fuel_types)
fuel_consumption = st.sidebar.number_input('Fuel Consumption (liters)', min_value=0.0, value=2000.0)
co2_emissions = st.sidebar.number_input('CO2 Emissions (kg)', min_value=0.0, value=6000.0)
weather_conditions = st.sidebar.selectbox('Weather Conditions', weather_conditions)

if st.button('Predict Engine Efficiency'):
    input_data = pd.DataFrame([[
        ship_type, route_id, month, distance, fuel_type,
        fuel_consumption, co2_emissions, weather_conditions
    ]], columns=features)

    # Ensure correct column order
    input_data = input_data[features]

    try:
        input_processed = preprocessor.transform(input_data)
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        st.stop()

    prediction = model.predict(input_processed)
    st.subheader('Predicted Engine Efficiency')
    st.write(f'{prediction[0]:.2f}%')
