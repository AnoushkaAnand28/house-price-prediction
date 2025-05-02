import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('house_price_model.pkl')

st.title("🏠 House Price Prediction Web App")
st.write("Upload a CSV file with house data to get price predictions.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read uploaded file
        input_data = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data")
        st.write(input_data.head())

        # Preprocess the input like training data
        data = pd.get_dummies(input_data)

        # Align input features to model features
        model_features = model.feature_names_in_
        data = data.reindex(columns=model_features, fill_value=0)

        # Make predictions
        predictions = model.predict(data)

        # Show results
        input_data['Predicted Price'] = predictions
        st.subheader("💰 Predicted House Prices")
        st.write(input_data[['Predicted Price']])
        st.bar_chart(input_data['Predicted Price'])

    except Exception as e:
        st.error(f"Something went wrong: {e}")