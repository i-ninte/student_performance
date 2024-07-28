import streamlit as st
import joblib
import numpy as np
import gzip

# Load the compressed model
with gzip.open('rf_model.pkl.gz', 'rb') as f:
    model = joblib.load(f)

# Streamlit app
st.title('Student Performance Prediction')

st.write("""
# Predict the Performance Index
Enter the details below to predict the performance index of a student.
""")

# Input fields
hours_studied = st.number_input('Hours Studied', min_value=0, max_value=24, value=5)
previous_scores = st.number_input('Previous Scores', min_value=0, max_value=100, value=70)
extracurricular_activities = st.selectbox('Extracurricular Activities', options=['yes', 'no'])
sleep_hours = st.number_input('Sleep Hours', min_value=0, max_value=24, value=6)
sample_question_papers_practiced = st.number_input('Sample Question Papers Practiced', min_value=0, max_value=10, value=3)

# Convert categorical input
extracurricular_activities = 1 if extracurricular_activities == 'yes' else 0

# Prediction button
if st.button('Predict'):
    # Input array
    input_data = np.array([[hours_studied, previous_scores, extracurricular_activities, sleep_hours, sample_question_papers_practiced]])
    
    # Check for NaN values in the input
    if np.isnan(input_data).any():
        st.write('Please fill in all fields with valid data.')
    else:
        # Prediction
        prediction = model.predict(input_data)[0]
        
        st.write(f'The predicted performance index is {prediction:.2f}')
