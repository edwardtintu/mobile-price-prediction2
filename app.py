import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib

# Add CSS styling for the webpage background, inputs, and headers
st.markdown("""
    <style>
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #FF6347;
            text-align: center;
            background-color: #FFF5EE;
            padding: 20px;
            border-radius: 10px;
        }
        .subheader {
            font-size: 25px;
            font-weight: bold;
            color: #4682B4;
        }
        .main {
            background-color: #F5F5F5;
            padding: 20px;
            border-radius: 10px;
        }
        .sidebar {
            background-color: #ADD8E6;
            padding: 10px;
            border-radius: 10px;
        }
        .price {
            font-size: 40px;
            color: #32CD32;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            background-color: #FFF8DC;
            border-radius: 10px;
        }
        .error {
            font-size: 20px;
            color: #FF6347;
            text-align: center;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the web app with color and background styling
st.markdown('<div class="title">Mobile Price Prediction</div>', unsafe_allow_html=True)

# Path to the dataset
# Corrected path to the dataset
dataset_path = r"C:\Users\EDWARD\Downloads\mobile-prediction-main2\mobile-prediction-main\Mobile Price Prediction Datatset.csv"

# Load the dataset
df = pd.read_csv(dataset_path)

# Preprocess the data
df = df.dropna(subset=['Brand me', 'RAM', 'ROM', 'Primary_Cam', 'Selfi_Cam', 'Battery_Power', 'Price'])

# Encoding the categorical data (Brand name, etc.)
label_encoder = LabelEncoder()
df['Brand me'] = label_encoder.fit_transform(df['Brand me'])

# Features and target
X = df[['Brand me', 'RAM', 'ROM', 'Mobile_Size', 'Primary_Cam', 'Selfi_Cam', 'Battery_Power']]
y = df['Price']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model (on test set)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Save the trained model (optional)
# joblib.dump(model, "price_predictor_model.pkl")

# User Input for prediction
st.markdown('<div class="subheader">Enter Mobile Details to Predict Price</div>', unsafe_allow_html=True)

# List of mobile brands
brand_options = df['Brand me'].unique()
brand_dict = {label_encoder.inverse_transform([i])[0]: i for i in brand_options}

# Display the mobile brand names in a selectbox
brand = st.selectbox("Select Mobile Brand", list(brand_dict.keys()), help="Choose the mobile brand from the list")

# Text input fields with valid limits in the placeholder
ram = st.text_input("RAM (in GB) [1-12]", help="Enter the RAM size in GB (e.g., 4, 6, 8, 12 GB)")

rom = st.text_input("ROM (in GB) [64-256]", help="Enter the ROM size in GB (e.g., 64, 128, 256 GB)")

mobile_size = st.text_input("Mobile Size (in inches) [4.0-7.0]", help="Enter the screen size in inches (e.g., 6.0, 5.5 inches)")

primary_cam = st.text_input("Primary Camera (in MP) [8-108]", help="Enter the primary camera resolution in MP (e.g., 48 MP, 64 MP)")

selfi_cam = st.text_input("Selfie Camera (in MP) [5-48]", help="Enter the selfie camera resolution in MP (e.g., 12 MP, 16 MP)")

battery_power = st.text_input("Battery Power (mAh) [1000-6000]", help="Enter the battery capacity in mAh (e.g., 4000, 5000 mAh)")

# Convert inputs to appropriate data types
if brand and ram and rom and mobile_size and primary_cam and selfi_cam and battery_power:
    try:
        ram = float(ram)
        rom = float(rom)
        mobile_size = float(mobile_size)
        primary_cam = int(primary_cam)
        selfi_cam = int(selfi_cam)
        battery_power = int(battery_power)
    except ValueError:
        st.error("Please enter valid numeric values for all fields.")
        ram = rom = mobile_size = primary_cam = selfi_cam = battery_power = None

# Convert the selected brand to its encoded form
brand_encoded = brand_dict.get(brand, -1)

# Prepare input data for prediction only if inputs are valid
if brand_encoded != -1 and ram and rom and mobile_size and primary_cam and selfi_cam and battery_power:
    user_input = np.array([[brand_encoded, ram, rom, mobile_size, primary_cam, selfi_cam, battery_power]])

    # Add button to calculate price
    if st.button('Calculate Price'):
        predicted_price = model.predict(user_input)

        # Show the predicted price only if it is positive
        if predicted_price[0] > 0:
            st.markdown(f'<div class="price">₹ {predicted_price[0]:,.2f}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error">This specification of mobile is not available.</div>', unsafe_allow_html=True)

# Make the app look more interactive with custom sidebar
st.sidebar.markdown("""
    <div class="sidebar">
        <h2>Mobile Price Prediction</h2>
        <p>This web app predicts the price of a mobile based on its features like RAM, ROM, Camera, Battery, etc.</p>
        <p>Fill in the details of the mobile, and the app will predict the mobile's name and price.</p>
    </div>
""", unsafe_allow_html=True)

# Extra UI elements for better explanation
st.markdown("""
### About the Model
This model uses a linear regression algorithm to predict the price of a mobile phone based on various features such as RAM, ROM, camera resolution, battery power, and screen size.
""")
