import streamlit as st
import numpy as np
import joblib

# Load the model, scaler, PCA, and label encoder
model = joblib.load('restaurant_revenue_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
le_city = joblib.load('le_city.pkl')

# Get the list of original feature names used during training
# Ensure this matches with your training feature names
feature_names = ['City', 'No_Of_Item', 'Order_Placed']

# Title
st.title('Restaurant Revenue Prediction')

# Dropdown for City
cities = le_city.classes_  # Get the list of city names used during training
selected_city = st.selectbox('Select City', cities)

# Input fields for No_Of_Item and Order_Placed
no_of_item = st.number_input('Number of Items', min_value=0, value=0)
order_placed = st.number_input('Order Placed', min_value=0, value=0)

# Prepare input data
input_data = {
    'City': selected_city,
    'No_Of_Item': no_of_item,
    'Order_Placed': order_placed
}

# Convert to DataFrame
input_data_df = pd.DataFrame([input_data])

# Encode the city feature
input_data_df['City'] = le_city.transform(input_data_df['City'])

# Extract features
input_data_array = input_data_df[['City', 'No_Of_Item', 'Order_Placed']].values

# Scale the input data
input_data_scaled = scaler.transform(input_data_array)

# Apply PCA transformation
input_data_pca = pca.transform(input_data_scaled)

# Button to make prediction
if st.button('Predict'):
    try:
        # Make prediction
        prediction = model.predict(input_data_pca)
        st.write(f'The predicted revenue is: ${prediction[0]:.2f}')
    except Exception as e:
        st.write(f'Error: {e}')
