
import streamlit as st
import pandas as pd
import joblib

# Load saved model and threshold
model = joblib.load("student_dropout_model.pkl")
threshold = joblib.load("dropout_threshold.pkl")

st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Dropout Prediction")
st.write("Enter student information to estimate dropout risk.")

st.divider()

# Student inputs
age = st.number_input("Age", min_value=10, max_value=100, value=20)
gender = st.selectbox("Gender", ["Male", "Female"])
country = st.text_input("Country", "Pakistan")
gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.0, step=0.1)
attendance = st.number_input(
    "Attendance Rate (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)
assignments = st.number_input(
    "Assignments Completed",
    min_value=0.0,
    value=8.0
)

extracurricular = st.selectbox(
    "Extracurricular Participation",
    ["Yes", "No"]
)

part_time = st.selectbox(
    "Part-Time Job",
    ["Yes", "No"]
)

internet = st.selectbox(
    "Internet Access At Home",
    ["Yes", "No"]
)

parent_education = st.selectbox(
    "Parent Education Level",
    ["High School", "Bachelor", "Master", "PhD"]
)

if st.button("Predict Dropout Risk"):

    attendance = min(attendance, 100)

    student = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Country": country,
        "GPA": gpa,
        "AttendanceRate(%)": attendance,
        "AssignmentsCompleted": assignments,
        "ExtracurricularParticipation": extracurricular,
        "PartTimeJob": part_time,
        "InternetAccessAtHome": internet,
        "ParentEducationLevel": parent_education
    }])

    # Missing-value indicators
    student["AssignmentsCompleted_Missing"] = 0
    student["Gender_Missing"] = 0
    student["ExtracurricularParticipation_Missing"] = 0
    student["PartTimeJob_Missing"] = 0
    student["InternetAccessAtHome_Missing"] = 0
    student["ParentEducationLevel_Missing"] = 0

    # Feature engineering
    student["Low_GPA"] = (student["GPA"] < 2.5).astype(int)
    student["Low_Attendance"] = (student["AttendanceRate(%)"] < 70).astype(int)

    student["Academic_Risk_Score"] = (
        student["Low_GPA"] +
        student["Low_Attendance"]
    )

    student["Assignment_Risk"] = (
        student["AssignmentsCompleted"] < 5
    ).astype(int)

    student["GPA_Attendance"] = (
        student["GPA"] * student["AttendanceRate(%)"]
    )

    student["GPA_Assignments"] = (
        student["GPA"] * student["AssignmentsCompleted"]
    )

    student["Attendance_Assignments"] = (
        student["AttendanceRate(%)"] *
        student["AssignmentsCompleted"]
    )

    student["Risk_Interaction"] = (
        student["Academic_Risk_Score"] *
        student["Assignment_Risk"]
    )

    student["Academic_Performance_Index"] = (
        student["GPA"] *
        student["AttendanceRate(%)"] *
        student["AssignmentsCompleted"]
    )

    # Prediction
    probability = model.predict_proba(student)[0][1]
    prediction = int(probability >= threshold)

    # Risk category
    if probability < 0.20:
        risk = "LOW RISK"
    elif probability < 0.50:
        risk = "MEDIUM RISK"
    else:
        risk = "HIGH RISK"

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        "Dropout Probability",
        f"{probability * 100:.2f}%"
    )

    if risk == "LOW RISK":
        st.success(f"Risk Category: {risk}")
    elif risk == "MEDIUM RISK":
        st.warning(f"Risk Category: {risk}")
    else:
        st.error(f"Risk Category: {risk}")

    if prediction == 1:
        st.error("Predicted Outcome: DROPOUT")
    else:
        st.success("Predicted Outcome: NO DROPOUT")

    st.info(
        "This prediction is an early-warning estimate and should not "
        "be treated as a definitive outcome."
    )
