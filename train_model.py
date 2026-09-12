import pandas as pd
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "heart.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

heart_df = pd.read_csv(DATA_FILE)

print(f"Dataset loaded successfully: {heart_df.shape}")


# ============================================================
# PREPROCESSING
# ============================================================

# Convert categorical columns to the same numeric encoding
# used by the Streamlit application.

mappings = {
    "Sex": {
        "M": 0,
        "F": 1
    },

    "ChestPainType": {
        "ATA": 0,
        "NAP": 1,
        "ASY": 2,
        "TA": 3
    },

    "RestingECG": {
        "Normal": 0,
        "ST": 1,
        "LVH": 2
    },

    "ExerciseAngina": {
        "N": 0,
        "Y": 1
    },

    "ST_Slope": {
        "Up": 0,
        "Flat": 1,
        "Down": 2
    }
}


for column, mapping in mappings.items():
    heart_df[column] = heart_df[column].map(mapping)


# ============================================================
# HANDLE ZERO VALUES
# ============================================================

# In the original dataset, zero cholesterol and zero resting
# blood pressure values are treated as missing values.

heart_df["Cholesterol"] = heart_df["Cholesterol"].replace(0, float("nan"))
heart_df["RestingBP"] = heart_df["RestingBP"].replace(0, float("nan"))


# Separate features and target

X = heart_df.drop("HeartDisease", axis=1)
y = heart_df["HeartDisease"]


# Convert missing values to numeric and impute them

imputer = KNNImputer(n_neighbors=3)

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training and testing data prepared.")


# ============================================================
# 1. LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...")

logistic_model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000
)

logistic_model.fit(X_train, y_train)

with open(
    MODEL_DIR / "logistic_regression_model.pkl",
    "wb"
) as file:
    pickle.dump(logistic_model, file)

print("Logistic Regression saved.")


# ============================================================
# 2. SUPPORT VECTOR MACHINE
# ============================================================

print("\nTraining Support Vector Machine...")

svm_model = SVC(
    kernel="rbf"
)

svm_model.fit(X_train, y_train)

with open(
    MODEL_DIR / "svm_model.pkl",
    "wb"
) as file:
    pickle.dump(svm_model, file)

print("SVM saved.")


# ============================================================
# 3. DECISION TREE
# ============================================================

print("\nTraining Decision Tree...")

dt = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced"
)

dt_params = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

dt_grid = GridSearchCV(
    dt,
    dt_params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

dt_grid.fit(X_train, y_train)

decision_tree_model = dt_grid.best_estimator_

with open(
    MODEL_DIR / "decision_tree_model.pkl",
    "wb"
) as file:
    pickle.dump(decision_tree_model, file)

print("Decision Tree saved.")
print("Best Decision Tree parameters:", dt_grid.best_params_)


# ============================================================
# 4. RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    random_state=42
)

rf_params = {
    "n_estimators": [50, 100, 150],
    "max_features": ["sqrt", "log2", None],
    "max_depth": [3, 6, 9, None],
    "max_leaf_nodes": [None, 10, 20]
}

rf_grid = GridSearchCV(
    rf,
    rf_params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

random_forest_model = rf_grid.best_estimator_

with open(
    MODEL_DIR / "random_forest_model.pkl",
    "wb"
) as file:
    pickle.dump(random_forest_model, file)

print("Random Forest saved.")
print("Best Random Forest parameters:", rf_grid.best_params_)


# ============================================================
# MODEL ACCURACY
# ============================================================

print("\n==============================")
print("MODEL ACCURACIES")
print("==============================")

print(
    "Logistic Regression:",
    round(logistic_model.score(X_test, y_test) * 100, 2),
    "%"
)

print(
    "SVM:",
    round(svm_model.score(X_test, y_test) * 100, 2),
    "%"
)

print(
    "Decision Tree:",
    round(decision_tree_model.score(X_test, y_test) * 100, 2),
    "%"
)

print(
    "Random Forest:",
    round(random_forest_model.score(X_test, y_test) * 100, 2),
    "%"
)

print("\nAll models have been trained successfully!")
print(f"Models saved in: {MODEL_DIR}")