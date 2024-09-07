import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
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
    
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Email: {email}", ln=True)
    
    pdf.ln(10)
    
    # General Details
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="General Details", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="These details help in understanding the basic demographic information of the patient, which is essential for tailoring the prediction model.")
    pdf.cell(200, 10, txt=f"Age: {inputs['Age'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Gender: {inputs['Gender'][0]}", ln=True)
    
    pdf.ln(10)
    
    # Lifestyle Factors
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="Lifestyle Factors", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="Lifestyle factors such as BMI, smoking, and alcohol consumption can significantly influence the risk of Alzheimer’s disease. This section helps capture those aspects.")
    pdf.cell(200, 10, txt=f"BMI: {inputs['BMI'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Smoking: {inputs['Smoking'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Alcohol Consumption: {inputs['AlcoholConsumption'][0]}", ln=True)
    
    pdf.ln(10)
    
    # Medical History
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="Medical History", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="A patient’s medical history, including family history and chronic conditions, provides important context for assessing Alzheimer's risk. This section collects relevant medical information.")
    pdf.cell(200, 10, txt=f"Family History of Alzheimer’s: {inputs['FamilyHistoryAlzheimers'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Cardiovascular Disease: {inputs['CardiovascularDisease'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Diabetes: {inputs['Diabetes'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Depression: {inputs['Depression'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Head Injury: {inputs['HeadInjury'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Hypertension: {inputs['Hypertension'][0]}", ln=True)
    
    pdf.ln(10)
    
    # Clinical Measurements
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="Clinical Measurements", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="Clinical measurements such as cholesterol levels and triglycerides are important indicators of overall health and can influence the risk of Alzheimer’s disease. This section gathers those metrics.")
    pdf.cell(200, 10, txt=f"Total Cholesterol: {inputs['CholesterolTotal'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"LDL Cholesterol: {inputs['CholesterolLDL'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"HDL Cholesterol: {inputs['CholesterolHDL'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Triglycerides: {inputs['CholesterolTriglycerides'][0]}", ln=True)
    
    pdf.ln(10)
    
    # Cognitive and Functional Assessments
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="Cognitive and Functional Assessments", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="Assessments of cognitive and functional abilities provide insights into the patient’s mental state and daily functioning, which are crucial for diagnosing Alzheimer’s disease.")
    pdf.cell(200, 10, txt=f"MMSE Score: {inputs['MMSE'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Functional Assessment Score: {inputs['FunctionalAssessment'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Memory Complaints: {inputs['MemoryComplaints'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Behavioral Problems: {inputs['BehavioralProblems'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"ADL Score: {inputs['ADL'][0]}", ln=True)
    
    pdf.ln(10)
    
    # Symptoms
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="Symptoms", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="Symptoms such as confusion, disorientation, and personality changes are key indicators of cognitive decline. This section collects information on these symptoms to aid in the diagnosis.")
    pdf.cell(200, 10, txt=f"Confusion: {inputs['Confusion'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Disorientation: {inputs['Disorientation'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Personality Changes: {inputs['PersonalityChanges'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Difficulty Completing Tasks: {inputs['DifficultyCompletingTasks'][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Forgetting: {inputs['Forgetfulness'][0]}", ln=True)
    
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Prediction (0: No Alzheimer’s, 1: Alzheimer’s): {int(prediction[0][0] > 0.5)}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence Score: {confidence:.2f}", ln=True)
    
    pdf_file = "diagnosis_report.pdf"
    pdf.output(pdf_file)
    return pdf_file

# Streamlit app
st.title('Alzheimer\'s Disease Diagnosis Prediction')

# Input section
st.header('General Details')
st.write("**These details help in understanding the basic demographic information of the patient, which is essential for tailoring the prediction model.**")
name = st.text_input('Name')
email = st.text_input('Email')
age = st.slider('Age', 20, 90)
st.write('0 for male and 1 for female.')
gender = st.selectbox('Gender', [0, 1])

st.header('Lifestyle Factors')
st.write("**Lifestyle factors such as BMI, smoking, and alcohol consumption can significantly influence the risk of Alzheimer’s disease. This section helps capture those aspects.**")
bmi = st.slider('BMI', 15.0, 40.0)
st.write('Smoking status, where 0 indicates No and 1 indicates Yes.')
smoking = st.selectbox('Smoking Status', [0, 1])
st.write('Weekly alcohol consumption in units, ranging from 0 to 20.')
alcohol_consumption = st.slider('Alcohol Consumption (units per week)', 0, 20)

st.header('Medical History')
st.write("**A patient’s medical history, including family history and chronic conditions, provides important context for assessing Alzheimer's risk. This section collects relevant medical information.**")
st.write("Family history of Alzheimer's Disease, where 0 indicates No and 1 indicates Yes.")
family_history_alzheimers = st.selectbox('Family History of Alzheimer\'s Disease', [0, 1])
st.write('Presence of cardiovascular disease, where 0 indicates No and 1 indicates Yes.')
cardiovascular_disease = st.selectbox('Cardiovascular Disease', [0, 1])
st.write('Presence of diabetes, where 0 indicates No and 1 indicates Yes.')
diabetes = st.selectbox('Diabetes', [0, 1])
st.write(' Presence of depression, where 0 indicates No and 1 indicates Yes.')
depression = st.selectbox('Depression', [0, 1])
st.write('History of head injury, where 0 indicates No and 1 indicates Yes.')
head_injury = st.selectbox('History of Head Injury', [0, 1])
st.write('Presence of hypertension, where 0 indicates No and 1 indicates Yes.')
hypertension = st.selectbox('Hypertension', [0, 1])

st.header('Clinical Measurements')
st.write("**Clinical measurements such as cholesterol levels and triglycerides are important indicators of overall health and can influence the risk of Alzheimer’s disease. This section gathers those metrics.**")
st.write('Total cholesterol levels, ranging from 150 to 300 mg/dL.')
cholesterol_total = st.slider('Total Cholesterol (mg/dL)', 150, 300)
st.write('Low-density lipoprotein cholesterol levels, ranging from 50 to 200 mg/dL.')
cholesterol_ldl = st.slider('LDL Cholesterol (mg/dL)', 50, 200)
st.write('High-density lipoprotein cholesterol levels, ranging from 20 to 100 mg/dL.')
cholesterol_hdl = st.slider('HDL Cholesterol (mg/dL)', 20, 100)
st.write('Triglycerides levels, ranging from 50 to 400 mg/dL.')
cholesterol_triglycerides = st.slider('Triglycerides (mg/dL)', 50, 300)

st.header('Cognitive and Functional Assessments')
st.write("**Assessments of cognitive and functional abilities provide insights into the patient’s mental state and daily functioning, which are crucial for diagnosing Alzheimer’s disease.**")
st.write('Mini-Mental State Examination score, ranging from 0 to 30. Lower scores indicate cognitive impairment.')
mmse = st.slider('MMSE Score', 0, 30)
st.write('Functional assessment score, ranging from 0 to 10. Lower scores indicate greater impairment.')
functional_assessment = st.slider('Functional Assessment Score', 0, 10)
st.write('Presence of memory complaints, where 0 indicates No and 1 indicates Yes.')
memory_complaints = st.selectbox('Memory Complaints', [0, 1])
st.write('Presence of behavioral problems, where 0 indicates No and 1 indicates Yes.')
behavioral_problems = st.selectbox('Behavioral Problems', [0, 1])
st.write('Activities of Daily Living score, ranging from 0 to 10. Lower scores indicate greater impairment.')
adl = st.slider('Activities of Daily Living Score', 0, 10)

st.header('Symptoms')
st.write("**Symptoms such as confusion, disorientation, and personality changes are key indicators of cognitive decline. This section collects information on these symptoms to aid in the diagnosis.**")
st.write('Presence of confusion, where 0 indicates No and 1 indicates Yes.')
confusion = st.selectbox('Confusion', [0, 1])
st.write('Presence of disorientation, where 0 indicates No and 1 indicates Yes.')
disorientation = st.selectbox('Disorientation', [0, 1])
st.write('Presence of personality changes, where 0 indicates No and 1 indicates Yes.')
personality_changes = st.selectbox('Personality Changes', [0, 1])
st.write('Presence of difficulty completing tasks, where 0 indicates No and 1 indicates Yes.')
difficulty_completing_tasks = st.selectbox('Difficulty Completing Tasks', [0, 1])
st.write('Presence of forgetfulness, where 0 indicates No and 1 indicates Yes.')
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
    'CholesterolTriglycerides': [cholesterol_triglycerides],
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
columns_to_scale = ['Age', 'BMI', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides']
scaled_data = input_data.copy()

input_data[columns_to_scale] = scaler.fit_transform(input_data[columns_to_scale])

# Predict
st.header('Prediction')
st.write("**Prediction Explanation**")
st.write("The prediction indicates the likelihood of Alzheimer’s Disease based on the provided inputs.")
st.write("A value of 0 means 'No Alzheimer’s Disease' and a value of 1 means 'Alzheimer’s Disease'.")

if st.button('Predict'):
    prediction = model.predict(input_data)
    confidence = prediction[0][0]
    result = int(confidence > 0.5)
    
    # Show prediction result
    st.write(f'Prediction: {result}')
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
