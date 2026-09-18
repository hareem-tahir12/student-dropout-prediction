# Student Dropout Prediction

A Machine Learning project that predicts the probability of student dropout and classifies students into different risk levels using student-related academic and personal information.

## Project Overview

Student dropout can negatively affect both students and educational institutions. This project uses supervised machine learning to analyze student information and estimate the likelihood of dropout.

The project also includes a Streamlit web application where users can enter student information and receive an instant prediction.

## Features

* Student dropout probability prediction
* Low, Medium, and High risk classification
* Predicted dropout outcome
* Interactive Streamlit interface
* Machine learning model integration
* Feature engineering and preprocessing

## Dataset

The dataset contains student-related information such as:

* Age
* Gender
* Country
* GPA
* Attendance Rate
* Assignments Completed
* Extracurricular Participation
* Part-Time Job
* Internet Access at Home
* Parent Education Level

The target variable is **Dropout**.

## Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Missing Value Handling
4. Exploratory Data Analysis
5. Feature Engineering
6. Data Preprocessing
7. Model Training
8. Model Evaluation
9. Prediction Application Development
10. Streamlit Deployment

## Feature Engineering

Additional features were created to improve the prediction process, including:

* Low GPA indicator
* Low Attendance indicator
* Academic Risk Score
* Assignment Risk
* GPA × Attendance
* GPA × Assignments
* Attendance × Assignments
* Risk Interaction
* Academic Performance Index
* Missing-value indicators

## Model

The project uses **Logistic Regression** for binary classification.

The trained model and optimized prediction threshold are saved using Joblib:

* `student_dropout_model.pkl`
* `dropout_threshold.pkl`

## Streamlit Application

The application allows users to enter student information and displays:

* Dropout Probability
* Risk Category
* Predicted Outcome

The application is intended as an early-warning prediction tool and should not be considered a definitive decision about a student's future.

## Project Files

```text
student-dropout-prediction/
│
├── app.py
├── requirements.txt
├── student_dropout_model.pkl
├── dropout_threshold.pkl
└── README.md
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Project Goal

The goal of this project is to demonstrate how machine learning can be used to identify students who may be at higher risk of dropping out and provide an interactive application for early risk assessment.
