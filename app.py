"""Streamlit interface for the student-performance demonstration."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"

st.set_page_config(page_title="Student Marks Analyzer", page_icon="📊",
                   layout="centered")

st.title("📊 Student Marks Analyzer")
st.caption("Student Performance Prediction Using Machine Learning")

st.warning(
    "Demonstration only: this project uses synthetic data, not real student "
    "records. Predictions are not validated for academic decisions."
)

if not MODEL_PATH.exists():
    st.error("The trained model is not available yet.")
    st.code("python src/generate_data.py\npython src/train_model.py")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.subheader("Enter academic indicators")
with st.form("prediction_form"):
    c1, c2 = st.columns(2)
    with c1:
        attendance = st.number_input("Attendance (%)", 0.0, 100.0, 80.0, 1.0)
        previous = st.number_input("Previous marks (%)", 0.0, 100.0, 65.0, 1.0)
        assignments = st.number_input("Assignment completion (%)", 0.0, 100.0, 75.0, 1.0)
    with c2:
        quiz = st.number_input("Quiz average (%)", 0.0, 100.0, 68.0, 1.0)
        study = st.number_input("Study hours per day", 0.0, 16.0, 3.0, 0.5)
        trend = st.number_input("Recent marks trend (-30 to +30)", -30.0, 30.0, 2.0, 1.0)
    submitted = st.form_submit_button("Predict performance", type="primary")

if submitted:
    row = pd.DataFrame([{
        "attendance_pct": attendance,
        "previous_marks_pct": previous,
        "assignment_completion_pct": assignments,
        "quiz_average_pct": quiz,
        "study_hours_per_day": study,
        "recent_marks_trend": trend,
    }])
    prediction = model.predict(row)[0]
    st.subheader("Demonstration result")
    if prediction == "Good":
        st.success(f"Predicted category: {prediction}")
    elif prediction == "Moderate":
        st.info(f"Predicted category: {prediction}")
    else:
        st.warning(f"Predicted category: {prediction}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row)[0]
        probability_frame = pd.DataFrame({
            "Category": model.classes_,
            "Model estimate": probabilities,
        }).sort_values("Model estimate", ascending=False)
        st.write("Model class estimates (not certainty):")
        st.bar_chart(probability_frame.set_index("Category"))

st.divider()
st.caption(
    "This tool is an academic prototype. A real deployment would require "
    "approved data, privacy safeguards, independent validation, fairness "
    "assessment and educator oversight."
)
