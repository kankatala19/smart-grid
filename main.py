import streamlit as st
import numpy as np
import tensorflow as tf
import datetime
import joblib  # For loading the scalers

# Load the trained model
try:
    model = tf.keras.models.load_model("dnn_lstm_model.keras", custom_objects={})
except Exception as e:
    st.error(f"Error loading model: {e}")

# Load scalers
try:
    scaler_dnn = joblib.load("scaler_dnn.pkl")
    scaler_lstm = joblib.load("scaler_lstm.pkl")
except Exception as e:
    st.error(f"Error loading scalers: {e}")

# Proceed with the rest of the Streamlit app setup if the model and scalers load successfully
if model and 'scaler_dnn' in locals() and 'scaler_lstm' in locals():
    # Streamlit app title
    st.title("DNN-LSTM Model Prediction Interface")

    st.write("### Enter input values for each feature")

    # Define features for DNN and LSTM branches
    common_features = [
        'QV2M_toc', 'TQL_toc', 'W2M_toc',
        'QV2M_san', 'TQL_san', 'W2M_san',
        'QV2M_dav', 'TQL_dav', 'W2M_dav'
    ]
    extra_lstm_features = ['T2M_toc', 'T2M_san', 'T2M_dav']

    # Default values for features
    default_values = {
        'T2M_toc': 26.750329589843773,
        'QV2M_toc': 0.020441001,
        'TQL_toc': 0.063964844,
        'W2M_toc': 10.19834579068856,
        'T2M_san': 25.125329589843773,
        'QV2M_san': 0.019281333,
        'TQL_san': 0.10018921,
        'W2M_san': 2.7857505868286183,
        'T2M_dav': 23.562829589843773,
        'QV2M_dav': 0.018403953,
        'TQL_dav': 0.16259766,
        'W2M_dav': 3.286878007712242
    }

    # Date and Time input for datetime feature
    st.write("#### DateTime Feature")
    date = st.date_input("Date (for datetime feature)", datetime.date.today())
    time = st.time_input("Time (for datetime feature)", datetime.time(0, 0))
    datetime_combined = datetime.datetime.combine(date, time).timestamp()

    # Collect inputs for all numeric features in one place
    st.write("#### Numeric Features")
    all_numeric_features = common_features + extra_lstm_features
    input_values = {}
    columns = st.columns(3)  # Arrange inputs in three columns

    for i, feature in enumerate(all_numeric_features):
        with columns[i % 3]:  # Distribute inputs across columns
            default_value = default_values.get(feature, 0.0)  # Use default value if available
            input_values[feature] = st.number_input(
                f"{feature}",
                value=default_value,
                format="%.5f",  # Accept up to 5 decimal places
                key=f"input_{feature}"
            )

    # Prepare the input data for prediction
    # DNN data includes common features and datetime
    dnn_data = np.array([input_values[feature] for feature in common_features] + [datetime_combined]).reshape(1, -1)

    # LSTM data includes common features, datetime, and additional LSTM-only features
    lstm_features = [input_values[feature] for feature in common_features] + \
                    [datetime_combined] + \
                    [input_values[feature] for feature in extra_lstm_features]

    # Ensure the LSTM input has the shape (1, 1, 13)
    lstm_data = np.array(lstm_features).reshape(1, 1, -1)  # Shape should be (1, 1, 13)

    # Scale the input data
    dnn_data_scaled = scaler_dnn.transform(dnn_data)
    lstm_data_scaled = scaler_lstm.transform(lstm_data.reshape(1, -1)).reshape(1, 1, -1)

    # Make predictions
    if st.button("Predict"):
        with st.spinner("Making prediction..."):
            try:
                prediction = model.predict([dnn_data_scaled, lstm_data_scaled])
                st.write(f"Prediction: {prediction[0][0]:.5f}")  # Format prediction output
            except Exception as prediction_error:
                st.error(f"Error during prediction: {prediction_error}")

