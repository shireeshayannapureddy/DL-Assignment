import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("heart_disease_mlp.pkl", "rb") as file:
    model = pickle.load(file)

W1 = model["W1"]
b1 = model["b1"]

W2 = model["W2"]
b2 = model["b2"]

W3 = model["W3"]
b3 = model["b3"]


# ReLU activation
def relu(z):
    return np.maximum(0, z)

# Sigmoid activation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Forward propagation
def mlp_forward_propagation(X):

    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)

    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)

    return A3

# Streamlit UI
st.title("Heart Disease Prediction System")

st.write("Enter patient medical details")

# Input fields
age = st.number_input("Age", min_value=1)

sex = st.number_input("Sex (0 = Female, 1 = Male)", min_value=0, max_value=1)

cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3)

trestbps = st.number_input("Resting Blood Pressure")

chol = st.number_input("Cholesterol Level")

fbs = st.number_input("Fasting Blood Sugar (0 or 1)", min_value=0, max_value=1)

restecg = st.number_input("Rest ECG (0-2)", min_value=0, max_value=2)

thalach = st.number_input("Maximum Heart Rate")

exang = st.number_input("Exercise Induced Angina (0 or 1)", min_value=0, max_value=1)

oldpeak = st.number_input("Oldpeak")

slope = st.number_input("Slope (0-2)", min_value=0, max_value=2)

ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4)

thal = st.number_input("Thalassemia (0-3)", min_value=0, max_value=3)

# Prediction button
if st.button("Predict"):

    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

   

    # Prediction
    prediction = mlp_forward_propagation(input_data)

    probability = prediction[0][0]

    st.write(f"Prediction Probability: {probability:.4f}")

    if probability >= 0.5:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")