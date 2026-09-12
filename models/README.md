#  Heart Disease Prediction System

A Machine Learning based Heart Disease Prediction System built with Python and Streamlit.

This project uses multiple Machine Learning classification algorithms to predict whether a person is likely to have heart disease based on medical and clinical features.

##  Features

-  Heart disease prediction
-  Multiple Machine Learning models
-  Model performance comparison
-  Bulk prediction using CSV files
-  Interactive Streamlit web application
-  User-friendly interface

##  Machine Learning Models

The following models were trained and evaluated:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 86.41% |
| SVM | 72.83% |
| Decision Tree | 80.98% |
| Random Forest | 87.50% |

**Best performing model:** Random Forest with **87.50% accuracy**.

##  Dataset

The project uses a heart disease dataset containing clinical features such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Angina
- Oldpeak
- ST Slope

The target variable is:

`HeartDisease`

##  Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Pickle

##  Project Structure

```text
HeartDiseasePredictor/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── heart.csv
│
└── models/
    ├── decision_tree_model.pkl
    ├── logistic_regression_model.pkl
    ├── random_forest_model.pkl
    └── svm_model.pkl