# import streamlit as st
# import pickle
# import pandas as pd
# import sklearn 
# print(sklearn.__version__)
# # Load the saved pipeline model


# with open('voting_pipeline_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Streamlit app header
# st.title('Model Prediction App')



# # Feature inputs
# st.header('Enter feature values:')
# gender = st.selectbox('Gender', ['Male', 'Female'])
# marital_status = st.selectbox('Marital Status', ['Single', 'Married','Widowed'])
# age = st.number_input('Age', min_value=18, max_value=100, value=30)
# fatigue = st.text_input('Fatigue', value="0.0363237389187373")
# slowing = st.text_input('Slowing', value="0.5808079689048639")
# pain = st.text_input('Pain', value="0.0053555198838411")
# hygiene = st.text_input('Hygiene', value="0.3069676027888867")
# movement = st.text_input('Movement', value="0.8136175955953819")

# # Converting string inputs to float after they are entered
# try:
#     fatigue = float(fatigue)
#     slowing = float(slowing)
#     pain = float(pain)
#     hygiene = float(hygiene)
#     movement = float(movement)
# except ValueError:
#     st.error("Please enter valid float values.")




# # Prepare the input data as a DataFrame
# input_data = pd.DataFrame({
#     'Gender': [gender],
#     'Marital_Status': [marital_status],
#     'Age': [age],
#     'Fatigue': [fatigue],
#     'Slowing': [slowing],
#     'Pain': [pain],
#     'Hygiene': [hygiene],
#     'Movement': [movement]
# })

# # Button for prediction
# if st.button('Predict'):
#     # Make the prediction
#     prediction = model.predict(input_data)
    
#     # Display the prediction result
#     st.subheader('Prediction Result:')
#     # st.write(f'The predicted value is: {prediction[0]}')
#     # if(prediction==1):
#     #     st.write('Not On the Schiznophrenic spectrum')
#     # else:
#     #     st.write('Not on the Schiznophrenic spectrum')
#     if(prediction==1):
#         st.write("Elevated proneness")
#     elif(prediction==2) :
#         st.write("High Proness")
#     elif(prediction==3):
#         st.write("Moderate Proneness")
#     elif(prediction==4):
#         st.write("Low proneness/not on the spectrum")
#     else:
#         st.write("Very High Proneness")


        
import streamlit as st
import pandas as pd
import pickle

# Load the models
with open('voting_pipeline_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('best_pipeline_model.pkl', 'rb') as f:
    depression_model = pickle.load(f)

# Create tabs
tab1, tab2 = st.tabs(["Schizophrenia Proneness", "Depression Level"])

# Tab 1: Schizophrenia Proneness Prediction
with tab1:
    st.header("Schizophrenia Proneness Prediction")

    gender = st.selectbox('Gender', ['Male', 'Female'])
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Widowed'])
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    
    # Feature inputs
    fatigue = st.text_input('Fatigue', value="0.0363237389187373")
    slowing = st.text_input('Slowing', value="0.5808079689048639")
    pain = st.text_input('Pain', value="0.0053555198838411")
    hygiene = st.text_input('Hygiene', value="0.3069676027888867")
    movement = st.text_input('Movement', value="0.8136175955953819")

    # Convert to float
    try:
        fatigue = float(fatigue)
        slowing = float(slowing)
        pain = float(pain)
        hygiene = float(hygiene)
        movement = float(movement)
    except ValueError:
        st.error("Please enter valid float values.")

    # Prepare DataFrame
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

    # Prediction button
    if st.button('Predict Schizophrenia Proneness'):
        prediction = model.predict(input_data)
        
        st.subheader('Prediction Result:')
        if prediction == 1:
            st.write("Elevated Proneness")
        elif prediction == 2:
            st.write("High Proneness")
        elif prediction == 3:
            st.write("Moderate Proneness")
        elif prediction == 4:
            st.write("Low Proneness / Not on the Spectrum")
        else:
            st.write("Very High Proneness")

# Tab 2: Depression Level Prediction
with tab2:
    st.header("Depression Level Prediction")

    # Categorical Inputs
    # name = st.text_input('Name', 'John Doe')
    # gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    # city = st.text_input('City', 'New York')
    # working_status = st.selectbox('Are you a working professional or a student?', ['Student', 'Working Professional'])
    # profession = st.text_input('Profession', 'Engineer')
    # sleep_duration = st.selectbox('Sleep Duration', ['Less than 5 hours', '5-7 hours', 'More than 7 hours'])
    # dietary_habits = st.selectbox('Dietary Habits', ['Vegetarian', 'Non-Vegetarian', 'Vegan'])
    # degree = st.text_input('Degree', 'Bachelor')
    # suicidal_thoughts = st.selectbox('Have you ever had suicidal thoughts?', ['Yes', 'No'])
    # family_history = st.selectbox('Family history of mental illness?', ['Yes', 'No'])

    name =  'John Doe'
    gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    city = st.text_input('City', 'New York')
    working_status = st.selectbox('Are you a working professional or a student?', ['Student', 'Working Professional'])
    profession = st.text_input('Profession', 'Engineer')
    sleep_duration = st.selectbox('Sleep Duration', ['Less than 5 hours', '5-7 hours', 'More than 7 hours'])
    dietary_habits = st.selectbox('Dietary Habits', ['Vegetarian', 'Non-Vegetarian', 'Vegan'])
    degree = st.text_input('Degree', 'Bachelor')
    suicidal_thoughts = st.selectbox('Have you ever had suicidal thoughts?', ['Yes', 'No'])
    family_history = st.selectbox('Family history of mental illness?', ['Yes', 'No'])

    # Numeric Inputs
    # id_value = st.number_input('ID', min_value=1, value=101)
    id_value=1
    age = st.number_input('Age', min_value=18, max_value=100, value=25)
    academic_pressure = st.slider('Academic Pressure', 0.0, 10.0, 5.0)
    work_pressure = st.slider('Work Pressure', 0.0, 10.0, 5.0)
    cgpa = st.slider('CGPA', 0.0, 10.0, 7.0)
    study_satisfaction = st.slider('Study Satisfaction', 0.0, 10.0, 5.0)
    job_satisfaction = st.slider('Job Satisfaction', 0.0, 10.0, 5.0)
    work_study_hours = st.slider('Work/Study Hours', 0.0, 24.0, 8.0)
    financial_stress = st.slider('Financial Stress', 0.0, 10.0, 5.0)

    # Prepare DataFrame
    input_data = pd.DataFrame({
        'name': [name],
        'gender': [gender],
        'city': [city],
        'working_professional_or_student': [working_status],
        'profession': [profession],
        'sleep_duration': [sleep_duration],
        'dietary_habits': [dietary_habits],
        'degree': [degree],
        'have_you_ever_had_suicidal_thoughts_?': [suicidal_thoughts],
        'family_history_of_mental_illness': [family_history],
        'id': [id_value],
        'age': [age],
        'academic_pressure': [academic_pressure],
        'work_pressure': [work_pressure],
        'cgpa': [cgpa],
        'study_satisfaction': [study_satisfaction],
        'job_satisfaction': [job_satisfaction],
        'work/study_hours': [work_study_hours],
        'financial_stress': [financial_stress]
    })

    # Prediction button
    if st.button('Predict Depression Level'):
        prediction = depression_model.predict(input_data)

        st.subheader('Prediction Result:')

        if prediction[0]==1:
            st.write("Predicted : Depression ")
        else:
            st.write("Predicted : No Depression")



