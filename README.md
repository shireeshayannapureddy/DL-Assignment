# Intelligent Clinical Decision-Support System for Heart Disease Prediction

## Project Overview

This project presents the development of an intelligent clinical decision-support system capable of predicting heart disease risk using historical patient medical observations and diagnostic indicators.

The system was developed as part of a Deep Learning assignment focused on implementing machine learning and neural network models entirely from scratch using NumPy without relying on deep learning frameworks such as TensorFlow or PyTorch.

The project demonstrates:
- supervised classification,
- deep learning fundamentals,
- backpropagation using chain rule,
- gradient descent optimization,
- healthcare-oriented performance evaluation,
- and deployment of a real-time prediction interface.

# Problem Statement

A healthcare institution seeks to develop an intelligent predictive system capable of identifying patients at risk of heart disease using clinical observations and diagnostic attributes.

The objective is to design a supervised classification framework that predicts disease probability and minimizes critical medical prediction errors.


# Dataset Information

Dataset Used:
Heart Disease Dataset (`heart.csv`)

Dataset Characteristics:
- Total Samples: 1888
- Total Features: 13
- Target Variable: Binary Classification
    - 0 ? No Heart Disease
    - 1 ? Heart Disease Present

Features Included:
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Rest ECG
- Maximum Heart Rate
- Exercise Induced Angina
- Oldpeak
- Slope
- Number of Major Vessels
- Thalassemia


The following models were implemented entirely from scratch using NumPy:

## Logistic Regression
Implemented Components:
- Weight Initialization
- Forward Propagation
- Sigmoid Activation
- Binary Cross Entropy Loss
- Backpropagation
- Gradient Descent Optimization
- Prediction Generation


Implemented Components:
- Layer-wise Parameter Initialization
- ReLU Activation for Hidden Layers
- Sigmoid Activation for Output Layer
- Forward Propagation
- Backpropagation using Chain Rule
- Gradient Updates
- Loss Tracking
- Prediction Generation

---

# Data Preprocessing

The following preprocessing steps were performed:
- Dataset loading using Pandas
- Missing value handling
- Feature normalization using StandardScaler
- Train-test data partitioning
- Feature preparation for neural network training

# Evaluation Metrics

The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Loss Convergence Analysis


# Healthcare-Oriented Interpretation

Within healthcare prediction systems, False Negative predictions are highly critical because undetected disease conditions may delay treatment and increase medical risk.

The Multi-Layer Perceptron achieved a higher Recall score and significantly reduced False Negative predictions compared to Logistic Regression.

Therefore, Recall represents a more appropriate evaluation metric than Accuracy in healthcare-oriented intelligent systems.


# Loss Convergence Analysis

Training loss was tracked throughout the optimization process for both models.

Observations:
- smooth convergence,
- stable optimization,
- effective gradient descent learning,
- and successful neural network training behavior.


# Deployment

A real-time prediction interface was developed using Streamlit.

The deployed application allows users to:
- enter patient clinical details,
- perform disease risk prediction,
- and receive prediction probability in real time.
