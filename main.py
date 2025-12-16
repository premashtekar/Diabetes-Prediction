import streamlit as st
import pandas as pd
import pickle
import os

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# ---------------- CUSTOM THEME ----------------
st.markdown("""
<style>
/* App Background */
.stApp {
    background-color: #0b0f1a;
}

/* Titles */
h1 {
    color: #1f77ff;
    text-align: center;
    font-weight: 700;
}

h2, h3 {
    color: #ff4da6;
}

/* Text */
p, label {
    color: #e0e0e0;
}

/* Number inputs */
input {
    background-color: #111827 !important;
    color: #e0e0e0 !important;
    border: 1px solid #1f77ff !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #ff4da6, #1f77ff);
    color: white;
    font-size: 16px;
    border-radius: 14px;
    padding: 10px 18px;
    border: none;
}
.stButton > button:hover {
    opacity: 0.85;
}

/* File uploader */
.css-1cpxqw2 {
    background-color: #111827 !important;
    border: 2px dashed #ff4da6 !important;
}

/* Dataframe */
.stDataFrame {
    background-color: #111827;
}

/* Success box */
.stSuccess {
    background-color: #111827;
    color: #e0e0e0;
    border-left: 6px solid #1f77ff;
}

/* Download button */
.stDownloadButton > button {
    background-color: #1f77ff;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🩺 Diabetes Prediction System")
st.write("Predict the probability of diabetes using a trained ML model.")

MODEL_FILE = "model.pkl"

# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model():
    train = pd.read_csv("train.csv")

    X = train.drop("diagnosed_diabetes", axis=1)
    y = train["diagnosed_diabetes"]

    for col in X.columns:
        if X[col].dtype == "object":
            X[col].fillna(X[col].mode()[0], inplace=True)
        else:
            X[col].fillna(X[col].median(), inplace=True)

    X = pd.get_dummies(X, drop_first=True)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    model.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump((model, X.columns), f)

    return model, X.columns

# Load model
if not os.path.exists(MODEL_FILE):
    model, feature_cols = train_model()
else:
    with open(MODEL_FILE, "rb") as f:
        model, feature_cols = pickle.load(f)

# ---------------- MANUAL INPUT ----------------
st.subheader("🔢 Manual Patient Input")

user_input = {}
for col in feature_cols:
    user_input[col] = st.number_input(col, value=0.0)

if st.button("Predict Diabetes Probability"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict_proba(input_df)[0][1]
    st.success(f"🧪 Predicted Diabetes Probability: {prediction:.4f}")

# ---------------- CSV UPLOAD ----------------
st.subheader("📂 Bulk Prediction (Upload CSV)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    for col in data.columns:
        if data[col].dtype == "object":
            data[col].fillna(data[col].mode()[0], inplace=True)
        else:
            data[col].fillna(data[col].median(), inplace=True)

    data = pd.get_dummies(data, drop_first=True)

    for col in feature_cols:
        if col not in data.columns:
            data[col] = 0

    data = data[feature_cols]

    preds = model.predict_proba(data)[:, 1]
    data["diabetes_probability"] = preds

    st.dataframe(data.head())

    st.download_button(
        "⬇ Download Predictions",
        data.to_csv(index=False),
        "predictions.csv",
        "text/csv"
    )
