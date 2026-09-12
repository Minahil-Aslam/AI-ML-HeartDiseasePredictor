
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import plotly.express as px


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# =========================================================
# MODEL FILES
# =========================================================

MODEL_FILES = {
    "Decision Tree": "decision_tree_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Support Vector Machine": "svm_model.pkl"
}


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(model_name):
    """
    Load a trained model from the models folder.
    """

    model_path = MODEL_DIR / MODEL_FILES[model_name]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{model_path}\n\n"
            f"Please make sure '{MODEL_FILES[model_name]}' "
            f"is inside the models folder."
        )

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model


# =========================================================
# PREPROCESS INPUT DATA
# =========================================================

def preprocess_input(
    age,
    sex,
    chest_pain,
    resting_bp,
    cholesterol,
    fasting_bs,
    resting_ecg,
    max_hr,
    exercise_angina,
    oldpeak,
    slope
):
    """
    Convert user-friendly input values into the
    numerical format used during model training.
    """

    # Sex
    sex_mapping = {
        "Male": 0,
        "Female": 1
    }

    # Chest Pain Type
    # Dataset:
    # ATA = 0
    # NAP = 1
    # ASY = 2
    # TA  = 3
    chest_pain_mapping = {
        "Atypical Angina": 0,
        "Non-Anginal Pain": 1,
        "Asymptomatic": 2,
        "Typical Angina": 3
    }

    # Fasting Blood Sugar
    fasting_bs_value = 1 if fasting_bs == ">120 mg/dl" else 0

    # Resting ECG
    resting_ecg_mapping = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }

    # Exercise Angina
    exercise_angina_mapping = {
        "No": 0,
        "Yes": 1
    }

    # ST Slope
    slope_mapping = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }

    # Create numerical input
    input_data = {
        "Age": age,
        "Sex": sex_mapping[sex],
        "ChestPainType": chest_pain_mapping[chest_pain],
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs_value,
        "RestingECG": resting_ecg_mapping[resting_ecg],
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina_mapping[exercise_angina],
        "Oldpeak": oldpeak,
        "ST_Slope": slope_mapping[slope]
    }

    return pd.DataFrame([input_data])


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_heart_disease(input_data):
    """
    Make predictions using all four trained models.
    """

    predictions = {}

    for model_name in MODEL_FILES.keys():

        model = load_model(model_name)

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            predictions[model_name] = "Heart Disease"
        else:
            predictions[model_name] = "No Heart Disease"

    return predictions


# =========================================================
# STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("❤️ Heart Disease Predictor")

st.write(
    "This application uses machine learning models to predict "
    "whether a patient is likely to have heart disease."
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    ["🔍 Predict", "📁 Bulk Predict", "📊 Model Information"]
)


# =========================================================
# TAB 1 — INDIVIDUAL PREDICTION
# =========================================================

with tab1:

    st.header("Patient Information")

    col1, col2, col3 = st.columns(3)

    # -----------------------------------------------------
    # Column 1
    # -----------------------------------------------------

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50
        )

        sex = st.selectbox(
            "Sex",
            ["Male", "Female"]
        )

        chest_pain = st.selectbox(
            "Chest Pain Type",
            [
                "Typical Angina",
                "Atypical Angina",
                "Non-Anginal Pain",
                "Asymptomatic"
            ]
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure",
            min_value=50,
            max_value=250,
            value=120
        )

    # -----------------------------------------------------
    # Column 2
    # -----------------------------------------------------

    with col2:

        cholesterol = st.number_input(
            "Cholesterol",
            min_value=50,
            max_value=700,
            value=200
        )

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar",
            [
                "<=120 mg/dl",
                ">120 mg/dl"
            ]
        )

        resting_ecg = st.selectbox(
            "Resting ECG",
            [
                "Normal",
                "ST-T Wave Abnormality",
                "Left Ventricular Hypertrophy"
            ]
        )

        max_hr = st.number_input(
            "Maximum Heart Rate",
            min_value=50,
            max_value=250,
            value=150
        )

    # -----------------------------------------------------
    # Column 3
    # -----------------------------------------------------

    with col3:

        exercise_angina = st.selectbox(
            "Exercise Induced Angina",
            [
                "No",
                "Yes"
            ]
        )

        oldpeak = st.number_input(
            "Oldpeak",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        slope = st.selectbox(
            "ST Slope",
            [
                "Upsloping",
                "Flat",
                "Downsloping"
            ]
        )

    st.divider()

    # -----------------------------------------------------
    # Prediction Button
    # -----------------------------------------------------

    if st.button(
        "🔮 Predict Heart Disease",
        type="primary"
    ):

        try:

            input_data = preprocess_input(
                age,
                sex,
                chest_pain,
                resting_bp,
                cholesterol,
                fasting_bs,
                resting_ecg,
                max_hr,
                exercise_angina,
                oldpeak,
                slope
            )

            predictions = predict_heart_disease(input_data)

            st.subheader("Prediction Results")

            result_columns = st.columns(4)

            for column, (model_name, result) in zip(
                result_columns,
                predictions.items()
            ):

                with column:

                    st.write(f"**{model_name}**")

                    if result == "Heart Disease":
                        st.error(f"⚠️ {result}")
                    else:
                        st.success(f"✅ {result}")

        except FileNotFoundError as error:

            st.error(str(error))

        except Exception as error:

            st.error(
                f"An error occurred while making the prediction: {error}"
            )


# =========================================================
# TAB 2 — BULK PREDICTION
# =========================================================

with tab2:

    st.header("Bulk Prediction")

    st.write(
        "Upload a CSV file containing patient records. "
        "The CSV must contain the same input columns as the training dataset."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    required_columns = [
        "Age",
        "Sex",
        "ChestPainType",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "RestingECG",
        "MaxHR",
        "ExerciseAngina",
        "Oldpeak",
        "ST_Slope"
    ]

    if uploaded_file is not None:

        try:

            bulk_data = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data")

            st.dataframe(
                bulk_data,
                use_container_width=True
            )

            missing_columns = [
                column
                for column in required_columns
                if column not in bulk_data.columns
            ]

            if missing_columns:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

            else:

                if st.button(
                    "🚀 Run Bulk Prediction",
                    type="primary"
                ):

                    model = load_model(
                        "Logistic Regression"
                    )

                    # Create copy so original uploaded data
                    # is not modified
                    prediction_data = bulk_data[
                        required_columns
                    ].copy()

                    # Dataset mappings
                    prediction_data["Sex"] = (
                        prediction_data["Sex"]
                        .map({
                            "M": 0,
                            "F": 1,
                            "Male": 0,
                            "Female": 1
                        })
                    )

                    prediction_data["ChestPainType"] = (
                        prediction_data["ChestPainType"]
                        .map({
                            "ATA": 0,
                            "NAP": 1,
                            "ASY": 2,
                            "TA": 3,
                            "Atypical Angina": 0,
                            "Non-Anginal Pain": 1,
                            "Asymptomatic": 2,
                            "Typical Angina": 3
                        })
                    )

                    prediction_data["RestingECG"] = (
                        prediction_data["RestingECG"]
                        .map({
                            "Normal": 0,
                            "ST": 1,
                            "LVH": 2,
                            "ST-T Wave Abnormality": 1,
                            "Left Ventricular Hypertrophy": 2
                        })
                    )

                    prediction_data["ExerciseAngina"] = (
                        prediction_data["ExerciseAngina"]
                        .map({
                            "N": 0,
                            "Y": 1,
                            "No": 0,
                            "Yes": 1
                        })
                    )

                    prediction_data["ST_Slope"] = (
                        prediction_data["ST_Slope"]
                        .map({
                            "Up": 0,
                            "Flat": 1,
                            "Down": 2,
                            "Upsloping": 0,
                            "Downsloping": 2
                        })
                    )

                    prediction_data["FastingBS"] = (
                        prediction_data["FastingBS"]
                        .replace({
                            "<=120 mg/dl": 0,
                            ">120 mg/dl": 1
                        })
                    )

                    # Check for invalid categorical values
                    if prediction_data.isnull().any().any():

                        st.error(
                            "Some CSV values could not be converted. "
                            "Please check your categorical values."
                        )

                    else:

                        predictions = model.predict(
                            prediction_data
                        )

                        result_data = bulk_data.copy()

                        result_data[
                            "Prediction"
                        ] = np.where(
                            predictions == 1,
                            "Heart Disease",
                            "No Heart Disease"
                        )

                        st.subheader(
                            "Prediction Results"
                        )

                        st.dataframe(
                            result_data,
                            use_container_width=True
                        )

                        csv_data = result_data.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="⬇️ Download Predictions",
                            data=csv_data,
                            file_name="heart_disease_predictions.csv",
                            mime="text/csv"
                        )

        except Exception as error:

            st.error(
                f"Error processing the CSV file: {error}"
            )


# =========================================================
# TAB 3 — MODEL INFORMATION
# =========================================================

with tab3:

    st.header("Model Information")

    model_info = pd.DataFrame({
        "Model": [
            "Decision Tree",
            "Logistic Regression",
            "Random Forest",
            "Support Vector Machine"
        ],
        "Accuracy (%)": [
            80.97,
            85.86,
            84.23,
            84.22
        ]
    })

    st.dataframe(
        model_info,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Model Accuracy Comparison")

    fig = px.bar(
        model_info,
        x="Model",
        y="Accuracy (%)",
        text="Accuracy (%)",
        title="Machine Learning Model Accuracy"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_range=[0, 100]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "The models were trained using the Heart Disease dataset "
        "and evaluated using classification performance metrics."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Heart Disease Prediction | Machine Learning Project"
)

