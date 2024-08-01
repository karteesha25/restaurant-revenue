import streamlit as st
import numpy as np
import joblib

# Load the model, scaler, and PCA
model = joblib.load('restaurant_revenue_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

# Get the list of original feature names
feature_names = ['Name','Franchise','Category','City','No_Of_Item','Order_Placed']  # Update with actual feature names

# Title
st.title('Restaurant Revenue Prediction')

# Input fields for the original features
input_data = {feature: st.number_input(feature, min_value=0, value=0) for feature in feature_names}

# Prepare input data
input_data_array = np.array([list(input_data.values())])

# Scale the input data
input_data_scaled = scaler.transform(input_data_array)

# Apply PCA transformation
input_data_pca = pca.transform(input_data_scaled)

# Button to make prediction
if st.button('Predict'):
    # Make prediction
    prediction = model.predict(input_data_pca)
    
    st.write(f'The predicted revenue is: ${prediction[0]:.2f}')
