import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from fpdf import FPDF

# Load the Keras model
model = load_model('alzhemiers_prediction.keras')

# Initialize the scaler (use the same scaler used during training)
scaler = StandardScaler()

# Function to create PDF report
def create_pdf_report(inputs, prediction, confidence, name, email):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Alzheimer's Disease Diagnosis Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Email: {email}", ln=True)
    pdf.ln(10)
    
    pdf.cell(200, 10, txt="General Details", ln=True, bold=True)
    pdf.cell(200, 10, txt=f"Age: {inputs['Age'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Gender: {inputs['Gender'][0]}", ln=True)
    
    pdf.ln(10)
    
    pdf.cell(200, 10, txt="Lifestyle Factors", ln=True, bold=True)
    pdf.cell(200, 10, txt=f"BMI: {inputs['BMI'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Smoking: {inputs['Smoking'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Alcohol Consumption: {inputs['AlcoholConsumption'][0]}", ln=True)
    
    pdf.ln(10)
    
    pdf.cell(200, 10, txt="Medical History", ln=True, bold=True)
    pdf.cell(200, 10, txt=f"Family History of Alzheimer’s: {inputs['FamilyHistoryAlzheimers'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Cardiovascular Disease: {inputs['CardiovascularDisease'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Diabetes: {inputs['Diabetes'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Depression: {inputs['Depression'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Head Injury: {inputs['HeadInjury'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Hypertension: {inputs['Hypertension'][0]}", ln=True)
    
    pdf.ln(10)
    
    pdf.cell(200, 10, txt="Clinical Measurements", ln=True, bold=True)
    pdf.cell(200, 10, txt=f"Total Cholesterol: {inputs['CholesterolTotal'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"LDL Cholesterol: {inputs['CholesterolLDL'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"HDL Cholesterol: {inputs['CholesterolHDL'][0]}", ln=True)
    
    pdf.ln(10)
    
    pdf.cell(200, 10, txt="Cognitive and Functional Assessments", ln=True, bold=True)
    pdf.cell(200, 10, txt=f"MMSE Score: {inputs['MMSE'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Functional Assessment Score: {inputs['FunctionalAssessment'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Memory Complaints: {inputs['MemoryComplaints'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Behavioral Problems: {inputs['BehavioralProblems'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"ADL Score: {inputs['ADL'][0]}", ln=True)
    
    pdf.ln(10)
    
    pdf.cell(200, 10, txt="Symptoms", ln=True, bold=True)
    pdf.cell(200, 10, txt=f"Confusion: {inputs['Confusion'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Disorientation: {inputs['Disorientation'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Personality Changes: {inputs['PersonalityChanges'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Difficulty Completing Tasks: {inputs['DifficultyCompletingTasks'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Forgetting: {inputs['Forgetfulness'][0]}", ln=True)
    
    pdf.ln(10)
    
    pdf.cell(200, 10, txt=f"Prediction (0: No Alzheimer’s, 1: Alzheimer’s): {int(prediction[0][0] > 0.5)}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence Score: {confidence:.2f}", ln=True)
    
    pdf_file = "diagnosis_report.pdf"
    pdf.output(pdf_file)
    return pdf_file

# Streamlit app
st.title('Alzheimer\'s Disease Diagnosis Prediction')

# Input section
st.header('General Details')
st.write("These details help in understanding the basic demographic information of the patient, which is essential for tailoring the prediction model.")
name = st.text_input('Name')
email = st.text_input('Email')
age = st.slider('Age', 60, 90)
gender = st.selectbox('Gender', [0, 1])

st.header('Lifestyle Factors')
st.write("Lifestyle factors such as BMI, smoking, and alcohol consumption can significantly influence the risk of Alzheimer’s disease. This section helps capture those aspects.")
bmi = st.slider('BMI', 15.0, 40.0)
smoking = st.selectbox('Smoking Status', [0, 1])
alcohol_consumption = st.slider('Alcohol Consumption (units per week)', 0, 20)

st.header('Medical History')
st.write("A patient’s medical history, including family history and chronic conditions, provides important context for assessing Alzheimer's risk. This section collects relevant medical information.")
family_history_alzheimers = st.selectbox('Family History of Alzheimer\'s Disease', [0, 1])
cardiovascular_disease = st.selectbox('Cardiovascular Disease', [0, 1])
diabetes = st.selectbox('Diabetes', [0, 1])
depression = st.selectbox('Depression', [0, 1])
head_injury = st.selectbox('History of Head Injury', [0, 1])
hypertension = st.selectbox('Hypertension', [0, 1])

st.header('Clinical Measurements')
st.write("Clinical measurements such as cholesterol levels are important indicators of overall health and can influence the risk of Alzheimer’s disease. This section gathers those metrics.")
cholesterol_total = st.slider('Total Cholesterol (mg/dL)', 150, 300)
cholesterol_ldl = st.slider('LDL Cholesterol (mg/dL)', 50, 200)
cholesterol_hdl = st.slider('HDL Cholesterol (mg/dL)', 20, 100)

st.header('Cognitive and Functional Assessments')
st.write("Assessments of cognitive and functional abilities provide insights into the patient’s mental state and daily functioning, which are crucial for diagnosing Alzheimer’s disease.")
mmse = st.slider('MMSE Score', 0, 30)
functional_assessment = st.slider('Functional Assessment Score', 0, 10)
memory_complaints = st.selectbox('Memory Complaints', [0, 1])
behavioral_problems = st.selectbox('Behavioral Problems', [0, 1])
adl = st.slider('Activities of Daily Living Score', 0, 10)

st.header('Symptoms')
st.write("Symptoms such as confusion, disorientation, and personality changes are key indicators of cognitive decline. This section collects information on these symptoms to aid in the diagnosis.")
confusion = st.selectbox('Confusion', [0, 1])
disorientation = st.selectbox('Disorientation', [0, 1])
personality_changes = st.selectbox('Personality Changes', [0, 1])
difficulty_completing_tasks = st.selectbox('Difficulty Completing Tasks', [0, 1])
forgetfulness = st.selectbox('Forgetfulness', [0, 1])

# Create a DataFrame with the user inputs
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'BMI': [bmi],
    'Smoking': [smoking],
    'AlcoholConsumption': [alcohol_consumption],
    'FamilyHistoryAlzheimers': [family_history_alzheimers],
    'CardiovascularDisease': [cardiovascular_disease],
    'Diabetes': [diabetes],
    'Depression': [depression],
    'HeadInjury': [head_injury],
    'Hypertension': [hypertension],
    'CholesterolTotal': [cholesterol_total],
    'CholesterolLDL': [cholesterol_ldl],
    'CholesterolHDL': [cholesterol_hdl],
    'MMSE': [mmse],
    'FunctionalAssessment': [functional_assessment],
    'MemoryComplaints': [memory_complaints],
    'BehavioralProblems': [behavioral_problems],
    'ADL': [adl],
    'Confusion': [confusion],
    'Disorientation': [disorientation],
    'PersonalityChanges': [personality_changes],
    'DifficultyCompletingTasks': [difficulty_completing_tasks],
    'Forgetfulness': [forgetfulness]
})

# Separate columns for scaling
columns_to_scale = ['Age', 'BMI', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL']
scaled_data = input_data.copy()

# Scale only specified columns
scaled_data[columns_to_scale] = scaler.fit_transform(input_data[columns_to_scale])

# Predict
if st.button('Predict'):
    prediction = model.predict(scaled_data)
    confidence = prediction[0][0]
    result = int(confidence > 0.5)
    
    # Show prediction result
    st.write(f'Prediction (0: No Alzheimer\'s, 1: Alzheimer\'s): {result}')
    st.write(f'Confidence Score: {confidence:.2f}')
    
    # Generate and display the PDF report
    pdf_file = create_pdf_report(input_data, prediction, confidence, name, email)
    with open(pdf_file, "rb") as f:
        st.download_button(
            label="Download Report",
            data=f,
            file_name=pdf_file,
            mime="application/pdf"
        )
