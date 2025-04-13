import streamlit as st
import joblib
import numpy as np

# Set the page configuration as the first Streamlit command
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")

# Load the saved model and SMOTE scaler
model = joblib.load('fraud_detection_model.pkl')
smote = joblib.load('smote_model.pkl')

# Adding background image using CSS
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://your_image_url.jpg'); /* Add your image URL here */
            background-size: cover;
            background-position: center center;
        }
        .sidebar .sidebar-content {
            background-color: #f4f4f9;
            color: #333;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #1E90FF;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #4682B4;
        }
        .stTitle, .stHeader {
            font-family: 'Arial', sans-serif;
            font-weight: bold;
        }
        .stMarkdown p {
            font-family: 'Helvetica', sans-serif;
            color: #555;
        }
        .stAlert {
            background-color: #ff4d4d;
            color: white;
            border-radius: 5px;
            padding: 20px;
        }
        .stSuccess {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("💳 Credit Card Fraud Detection")
st.write("Enter the transaction features to predict whether it's a **Fraudulent** or **Non-Fraudulent** transaction.")

# Sidebar for user inputs
st.sidebar.header("Transaction Feature Input")

# Input fields for the features
input_features = []
feature_names = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
    'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22',
    'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

# Adding input fields to sidebar dynamically
for feature in feature_names:
    value = st.sidebar.number_input(f"{feature}", min_value=0.0, step=0.01)
    input_features.append(value)

# Predict button with custom style
if st.sidebar.button("🔍 Predict Fraud or Not"):
    input_data = np.array([input_features])

    # Resample the data using SMOTE before prediction
    input_resampled = smote.transform(input_data)

    # Prediction
    prediction = model.predict(input_resampled)

    # Displaying results in a more interactive way
    if prediction[0] == 1:
        st.markdown("""
            <div style="background-color: red; color: white; padding: 10px; border-radius: 5px;">
                <strong>🚨 Fraudulent Transaction Detected!</strong>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background-color: green; color: white; padding: 10px; border-radius: 5px;">
                <strong>✅ This is a **Non-Fraudulent** Transaction.</strong>
            </div>
        """, unsafe_allow_html=True)
