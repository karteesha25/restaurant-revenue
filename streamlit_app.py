import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('restaurant_revenue_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title
st.title('Restaurant Revenue Prediction')

# Define the input fields based on your feature set
No_Of_Item = st.number_input('Number of Items', min_value=1, max_value=1000, value=10)
Order_Placed = st.number_input('Order Placed', min_value=1, max_value=1000, value=50)
# Add more input fields if there are more features in your dataset

# When 'Predict' button is clicked, make the prediction and display it
if st.button('Predict'):
    # Prepare the input data
    input_data = np.array([[No_Of_Item, Order_Placed]])  # Ensure all features are included
    
    # Debugging statements
    st.write(f"Input data shape: {input_data.shape}")
    st.write(f"Scaler expected input shape: {scaler.n_features_in_}")

    try:
        input_data_scaled = scaler.transform(input_data)  # Scale the input data
        # Make the prediction
        prediction = model.predict(input_data_scaled)
        st.write(f'The predicted revenue is: {prediction[0]:.2f}')
    except ValueError as e:
        st.error(f"Error: {e}")
        st.error("Check the number of features in the input data and the scaler.")

