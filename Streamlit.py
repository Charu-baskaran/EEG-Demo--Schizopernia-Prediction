import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from streamlit_lottie import st_lottie

# Load models
with open('voting_pipeline_model.pkl', 'rb') as f:
    demo_model = pickle.load(f)

with open('best_pipeline_model.pkl', 'rb') as f:
    depression_model = pickle.load(f)

eeg_model = joblib.load("schizophrenia_model.pkl")

# Load Lottie animation
def load_lottie(path):
    with open(path, "r") as file:
        return json.load(file)

loading_animation = load_lottie("loading.json")

# Initialize session states
if "schizo_result" not in st.session_state:
    st.session_state.schizo_result = None
if "enable_depression_test" not in st.session_state:
    st.session_state.enable_depression_test = False
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# Sidebar navigation
section = st.sidebar.radio("🧭 Navigate", ["🧠 Schizophrenia Proneness", "📉 Depression Level", "🔬 EEG-based Detection"])

# ================= TAB 1: Schizophrenia Proneness =================
if section == "🧠 Schizophrenia Proneness":
    st.header("🧠 Demographic-Based Schizophrenia Proneness Prediction")

    gender = st.selectbox('Gender', ['Male', 'Female'])
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Widowed'])
    age = st.number_input('Age', min_value=18, max_value=100, value=30)

    fatigue = st.text_input('Fatigue', value="0.036")
    slowing = st.text_input('Slowing', value="0.580")
    pain = st.text_input('Pain', value="0.005")
    hygiene = st.text_input('Hygiene', value="0.306")
    movement = st.text_input('Movement', value="0.813")

    try:
        fatigue, slowing, pain, hygiene, movement = map(float, [fatigue, slowing, pain, hygiene, movement])
    except ValueError:
        st.error("Please enter valid numbers.")

    if st.button('Predict Schizophrenia Proneness'):
        input_df = pd.DataFrame({
            'Gender': [gender], 'Marital_Status': [marital_status], 'Age': [age],
            'Fatigue': [fatigue], 'Slowing': [slowing], 'Pain': [pain],
            'Hygiene': [hygiene], 'Movement': [movement]
        })
        result = demo_model.predict(input_df)[0]
        st.session_state.schizo_result = result

        proneness_map = {
            1: "Elevated Proneness", 2: "High Proneness", 3: "Moderate Proneness",
            4: "Low / Not on the Spectrum", 5: "Very High Proneness"
        }

        st.subheader("🧾 Result:")
        st.info(proneness_map.get(result, "Unknown"))

        if result in [1, 2, 3]:
            st.session_state.enable_depression_test = True
            st.success("You may proceed to the Depression Test from the sidebar.")

# ================= TAB 2: Depression Test =================
elif section == "📉 Depression Level":
    if not st.session_state.enable_depression_test:
        st.warning("Please complete the Schizophrenia test with moderate or high results to access this section.")
    else:
        st.header("📉 Depression Level Prediction")

        name = st.text_input("Name", "John Doe")
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
        city = st.text_input('City', 'New York')
        working_status = st.selectbox('Working Status', ['Student', 'Working Professional'])
        profession = st.text_input('Profession', 'Engineer')
        sleep_duration = st.selectbox('Sleep Duration', ['Less than 5 hours', '5-7 hours', 'More than 7 hours'])
        dietary = st.selectbox('Dietary Habits', ['Vegetarian', 'Non-Vegetarian', 'Vegan'])
        degree = st.text_input('Degree', 'Bachelor')
        suicidal = st.selectbox('Suicidal Thoughts?', ['Yes', 'No'])
        family = st.selectbox('Family History of Mental Illness?', ['Yes', 'No'])

        age = st.number_input('Age', min_value=18, max_value=100, value=25)
        academic = st.slider('Academic Pressure', 0.0, 10.0, 5.0)
        work = st.slider('Work Pressure', 0.0, 10.0, 5.0)
        cgpa = st.slider('CGPA', 0.0, 10.0, 7.0)
        study = st.slider('Study Satisfaction', 0.0, 10.0, 5.0)
        job = st.slider('Job Satisfaction', 0.0, 10.0, 5.0)
        hours = st.slider('Work/Study Hours', 0.0, 24.0, 8.0)
        stress = st.slider('Financial Stress', 0.0, 10.0, 5.0)

        input_df = pd.DataFrame({
            'gender': [gender], 'city': [city], 'working_professional_or_student': [working_status],
            'profession': [profession], 'sleep_duration': [sleep_duration], 'dietary_habits': [dietary],
            'degree': [degree], 'have_you_ever_had_suicidal_thoughts_?': [suicidal],
            'family_history_of_mental_illness': [family], 'age': [age], 'academic_pressure': [academic],
            'work_pressure': [work], 'cgpa': [cgpa], 'study_satisfaction': [study],
            'job_satisfaction': [job], 'work/study_hours': [hours], 'financial_stress': [stress],
            'id': [1], 'name': [name]
        })

        if st.button("Predict Depression Level"):
            result = depression_model.predict(input_df)[0]
            st.subheader("🔍 Prediction:")
            st.success("Predicted: Depression" if result == 1 else "Predicted: No Depression")

# ================= TAB 3: EEG Schizophrenia Detection =================
# elif section == "🔬 EEG-based Detection":
#     st.header("🔬 EEG-Based Schizophrenia Detection")

#     uploaded = st.file_uploader("Upload EEG File (.eea)", type="eea")

#     if uploaded is not None:
#         data = np.loadtxt(uploaded)

#         if len(data) < 7680:
#             data = np.pad(data, (0, 7680 - len(data)), 'constant')
#         else:
#             data = data[:7680]

#         data = data.reshape(1, -1)

#         prediction = eeg_model.predict(data)[0]

#         if prediction == 1:
#             st.success("✅ Normal EEG Pattern Detected")
#         else:
#             st.error("⚠ Schizophrenia Risk Detected")

#         st.session_state.prediction_history.append(prediction)

#         # EEG Plot
#         st.subheader("📊 EEG Signal Plot")
#         fig, ax = plt.subplots()
#         ax.plot(data.flatten(), color='purple')
#         ax.set_title("EEG Signal")
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Amplitude")
#         st.pyplot(fig)

#         # Bar Graph
#         st.subheader("📈 Prediction Distribution")
#         counts = [st.session_state.prediction_history.count(0), st.session_state.prediction_history.count(1)]
#         fig, ax = plt.subplots()
#         ax.bar(['Schizophrenia', 'Normal'], counts, color=['red', 'green'])
#         st.pyplot(fig)

#         # Optional confusion matrix
#         if len(st.session_state.prediction_history) > 1:
#             st.subheader("📊 Confusion Matrix (Simulated Labels)")
#             y_true = [0, 1] * (len(st.session_state.prediction_history) // 2 + 1)
#             y_pred = st.session_state.prediction_history
#             cm = confusion_matrix(y_true[:len(y_pred)], y_pred)
#             fig, ax = plt.subplots()
#             sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Schizophrenia', 'Normal'], yticklabels=['Schizophrenia', 'Normal'])
#             ax.set_title("Confusion Matrix")
#             st.pyplot(fig)
elif section == "🔬 EEG-based Detection":
    st.header("🔬 EEG-Based Schizophrenia Detection")

    uploaded = st.file_uploader("Upload EEG File (.eea)", type="eea")

    if uploaded is not None:
        data = np.loadtxt(uploaded)

        if len(data) < 7680:
            data = np.pad(data, (0, 7680 - len(data)), 'constant')
        else:
            data = data[:7680]

        data = data.reshape(1, -1)

        prediction = eeg_model.predict(data)[0]

        if prediction == 1:
            st.success("✅ Normal EEG Pattern Detected")
        else:
            st.error("⚠ Schizophrenia Risk Detected")

        st.session_state.prediction_history.append(prediction)

        # --- Side-by-side layout for EEG Plot and Prediction Distribution ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 EEG Signal Plot")
            fig1, ax1 = plt.subplots()
            ax1.plot(data.flatten(), color='purple')
            ax1.set_title("EEG Signal")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Amplitude")
            st.pyplot(fig1)

        with col2:
            st.subheader("📈 Prediction Distribution")
            counts = [
                st.session_state.prediction_history.count(0),
                st.session_state.prediction_history.count(1)
            ]
            fig2, ax2 = plt.subplots()
            ax2.bar(['Schizophrenia', 'Normal'], counts, color=['red', 'green'])
            st.pyplot(fig2)