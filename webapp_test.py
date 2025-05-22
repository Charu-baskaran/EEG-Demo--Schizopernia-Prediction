import streamlit as st
import pickle
import pandas as pd
from streamlit_lottie import st_lottie
import json
import time

# Load animations
def load_lottie(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

loading_animation = load_lottie("loading.json")  # Make sure you have a Lottie JSON file for loading animation

# Load models
with open('voting_pipeline_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('best_pipeline_model.pkl', 'rb') as f:
    depression_model = pickle.load(f)

# Initialize session state
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "schizophrenia"
if "schizophrenia_prediction" not in st.session_state:
    st.session_state.schizophrenia_prediction = None

# Tab switching logic
if st.session_state.active_tab == "schizophrenia":
    st.header("Schizophrenia Proneness Prediction")

    gender = st.selectbox('Gender', ['Male', 'Female'])
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Widowed'])
    age = st.number_input('Age', min_value=18, max_value=100, value=30)

    fatigue = st.text_input('Fatigue', value="0.036")
    slowing = st.text_input('Slowing', value="0.580")
    pain = st.text_input('Pain', value="0.005")
    hygiene = st.text_input('Hygiene', value="0.306")
    movement = st.text_input('Movement', value="0.813")

    try:
        fatigue = float(fatigue)
        slowing = float(slowing)
        pain = float(pain)
        hygiene = float(hygiene)
        movement = float(movement)
    except ValueError:
        st.error("Please enter valid numbers.")

    input_data = pd.DataFrame({
        'Gender': [gender],
        'Marital_Status': [marital_status],
        'Age': [age],
        'Fatigue': [fatigue],
        'Slowing': [slowing],
        'Pain': [pain],
        'Hygiene': [hygiene],
        'Movement': [movement]
    })

    if st.button('Predict Schizophrenia Proneness'):
        prediction = model.predict(input_data)[0]
        st.session_state.schizophrenia_prediction = prediction  # Store prediction

        proneness_map = {
            1: "Elevated Proneness",
            2: "High Proneness",
            3: "Moderate Proneness",
            4: "Low Proneness / Not on the Spectrum",
            5: "Very High Proneness"
        }
        st.subheader('Prediction Result:')
        st.write(proneness_map.get(prediction, "Unknown"))

    # Show button only if schizophrenia proneness is 1, 2, or 3
    if st.session_state.schizophrenia_prediction is not None and st.session_state.schizophrenia_prediction <= 3:
        if st.button("Proceed to Depression Test"):
            st.session_state.active_tab = "depression"
            st_lottie(loading_animation, speed=1, width=300, height=300)  # Show loading animation
            time.sleep(2)  # Simulate processing delay
            st.rerun()

elif st.session_state.active_tab == "depression":
    st.header("Depression Level Prediction")
    name = 'John Doe'
    gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    city = st.text_input('City', 'New York')
    working_status = st.selectbox('Are you a working professional or a student?', ['Student', 'Working Professional'])
    profession = st.text_input('Profession', 'Engineer')
    sleep_duration = st.selectbox('Sleep Duration', ['Less than 5 hours', '5-7 hours', 'More than 7 hours'])
    dietary_habits = st.selectbox('Dietary Habits', ['Vegetarian', 'Non-Vegetarian', 'Vegan'])
    degree = st.text_input('Degree', 'Bachelor')
    suicidal_thoughts = st.selectbox('Have you ever had suicidal thoughts?', ['Yes', 'No'])
    family_history = st.selectbox('Family history of mental illness?', ['Yes', 'No'])

    age = st.number_input('Age', min_value=18, max_value=100, value=25)
    academic_pressure = st.slider('Academic Pressure', 0.0, 10.0, 5.0)
    work_pressure = st.slider('Work Pressure', 0.0, 10.0, 5.0)
    cgpa = st.slider('CGPA', 0.0, 10.0, 7.0)
    study_satisfaction = st.slider('Study Satisfaction', 0.0, 10.0, 5.0)
    job_satisfaction = st.slider('Job Satisfaction', 0.0, 10.0, 5.0)
    work_study_hours = st.slider('Work/Study Hours', 0.0, 24.0, 8.0)
    financial_stress = st.slider('Financial Stress', 0.0, 10.0, 5.0)
    id_value = 1

    input_data = pd.DataFrame({
        'gender': [gender],
        'city': [city],
        'working_professional_or_student': [working_status],
        'profession': [profession],
        'sleep_duration': [sleep_duration],
        'dietary_habits': [dietary_habits],
        'degree': [degree],
        'have_you_ever_had_suicidal_thoughts_?': [suicidal_thoughts],
        'family_history_of_mental_illness': [family_history],
        'age': [age],
        'academic_pressure': [academic_pressure],
        'work_pressure': [work_pressure],
        'cgpa': [cgpa],
        'study_satisfaction': [study_satisfaction],
        'job_satisfaction': [job_satisfaction],
        'work/study_hours': [work_study_hours],
        'financial_stress': [financial_stress],
        'id': [id_value],
        'name': [name]
    })

    if st.button('Predict Depression Level'):
        prediction = depression_model.predict(input_data)
        st.subheader('Prediction Result:')
        st.write("Predicted: Depression" if prediction[0] == 1 else "Predicted: No Depression")
