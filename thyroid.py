#Importing the necessary libraries
import streamlit as st
import joblib
import numpy as np

##model = joblib.load('C:/Users/User/Dropbox/Engage revised materials/model/weigth_model.pkl')
model = joblib.load("C:\ENGAGE\Supervised_ML_Logistic_Regression_Model_ReccurenceLikelihood.joblib")
st.title('Thyroid Occurence Prediction App')
st.write("This app predicts the likelihood of thyroid recurrence based on clinical features and patient demographics.")
##age = st.number_input('Age', min_value=0.0, max_value=100.0, value=6.0)
##if st.button('Predict'):
##    input_features = np.array([[age]])
##    prediction = model.predict(input_features)
##    st.write(f'Predicted weight: {[prediction][0]}')

# Input fields
age = st.number_input('Age', min_value=0.0, max_value=100.0, value=6.0)
# Binary features using selectbox
gender = st.selectbox('Gender', options=[0, 1], index=0)
smoking = st.selectbox('Smoking', options=[0, 1], index=0)
hx_smoking = st.selectbox('Hx Smoking', options=[0, 1], index=0)
hx_radiotherapy = st.selectbox('Hx Hx Radiothreapy', options=[0, 1], index=0)
thyroid_function = st.selectbox('Thyroid Function', options=[0, 1, 2, 3, 4], index=0)
physical_examination = st.selectbox('Physical Examination', options=[0, 1, 2, 3, 4], index=0)
adenopathy = st.selectbox('Adenopathy', options=[0, 1, 2, 3, 4, 5], index=0)
pathology = st.selectbox('Pathology', options=[0, 1, 2, 3], index=0)
focality = st.selectbox('Focality', options=[0, 1], index=0)
risk = st.selectbox('Risk', options=[0, 1, 2], index=0)
tumor = st.selectbox('T', options=[0, 1, 2, 3, 4, 5, 6], index=0)
lymph_node = st.selectbox('N', options=[0, 1, 2], index=0)
metastasis = st.selectbox('M', options=[0, 1], index=0)
stage = st.selectbox('Stage', options=[0, 1, 2, 3, 4], index=0)

if st.button('Predict'):
    # Prepare input features for prediction
    #input_features = np.array([age])--for single input or simple linear model
    input_features = np.array([[age, gender, smoking, hx_smoking, hx_radiotherapy, thyroid_function, physical_examination, adenopathy, pathology, focality, risk, tumor, lymph_node, metastasis, stage]])  # Ensure this is a 2D array

    # Make prediction
    prediction = model.predict(input_features)

    # Display prediction
    st.write(f'Predicted Recurrence: {prediction[0]:.2f}')