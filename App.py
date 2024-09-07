import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from io import BytesIO

# Load the Keras model
model = load_model('alzhemiers_prediction.keras')

# Initialize the scaler (use the same scaler used during training)
scaler = StandardScaler()

# Function to create PDF report
def create_pdf_report(inputs, prediction, confidence, name, email):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    def draw_styled_title(title, y):
        c.setFont("Helvetica-Bold", 16)
        c.setFillColor(HexColor("#003366"))  # Dark blue color
        c.drawString(100, y, title)
        c.setFillColor(HexColor("#000000"))  # Black color for subsequent text
        return y - 30
    
    def draw_section_header(title, y):
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(HexColor("#0066CC"))  # Blue color for section headers
        c.drawString(100, y, title)
        c.setFillColor(HexColor("#000000"))  # Black color for subsequent text
        return y - 20
    
    def draw_text(label, text, y):
        c.setFont("Helvetica", 12)
        c.drawString(100, y, f"{label}: {text}")
        return y - 20
    
    # Function to create a new page
    def new_page():
        c.showPage()
        c.setFont("Helvetica", 12)
        return height - 50  # Reset y for the new page
    
    # Start drawing the PDF
    y = height - 50
    
    y = draw_styled_title("Alzheimer's Disease Diagnosis Report", y)
    y -= 10
    c.setFont("Helvetica", 12)
    c.drawString(100, y, f"Name: {name}")
    y -= 20
    c.drawString(100, y, f"Email: {email}")
    y -= 30
    
    y = draw_section_header("General Details", y)
    y = draw_text("Age", inputs['Age'][0], y)
    y = draw_text("Gender", inputs['Gender'][0], y)
    
    if y < 100:  # Check if content is too low, start a new page
        y = new_page()
    
    y = draw_section_header("Lifestyle Factors", y)
    y = draw_text("BMI", inputs['BMI'][0], y)
    y = draw_text("Smoking", inputs['Smoking'][0], y)
    y = draw_text("Alcohol Consumption", inputs['AlcoholConsumption'][0], y)
    
    if y < 100:
        y = new_page()
    
    y = draw_section_header("Medical History", y)
    y = draw_text("Family History of Alzheimer’s", inputs['FamilyHistoryAlzheimers'][0], y)
    y = draw_text("Cardiovascular Disease", inputs['CardiovascularDisease'][0], y)
    y = draw_text("Diabetes", inputs['Diabetes'][0], y)
    y = draw_text("Depression", inputs['Depression'][0], y)
    y = draw_text("Head Injury", inputs['HeadInjury'][0], y)
    y = draw_text("Hypertension", inputs['Hypertension'][0], y)
    
    if y < 100:
        y = new_page()
    
    y = draw_section_header("Clinical Measurements", y)
    y = draw_text("Total Cholesterol", inputs['CholesterolTotal'][0], y)
    y = draw_text("LDL Cholesterol", inputs['CholesterolLDL'][0], y)
    y = draw_text("HDL Cholesterol", inputs['CholesterolHDL'][0], y)
    y = draw_text("Triglycerides", inputs['CholesterolTriglycerides'][0], y)
    
    if y < 100:
        y = new_page()
    
    y = draw_section_header("Cognitive and Functional Assessments", y)
    y = draw_text("MMSE Score", inputs['MMSE'][0], y)
    y = draw_text("Functional Assessment Score", inputs['FunctionalAssessment'][0], y)
    y = draw_text("Memory Complaints", inputs['MemoryComplaints'][0], y)
    y = draw_text("Behavioral Problems", inputs['BehavioralProblems'][0], y)
    y = draw_text("ADL Score", inputs['ADL'][0], y)
    
    if y < 100:
        y = new_page()
    
    y = draw_section_header("Symptoms", y)
    y = draw_text("Confusion", inputs['Confusion'][0], y)
    y = draw_text("Disorientation", inputs['Disorientation'][0], y)
    y = draw_text("Personality Changes", inputs['PersonalityChanges'][0], y)
    y = draw_text("Difficulty Completing Tasks", inputs['DifficultyCompletingTasks'][0], y)
    y = draw_text("Forgetting", inputs['Forgetfulness'][0], y)
    
    if y < 100:
        y = new_page()
    
    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y, f"Prediction (0: No Alzheimer’s, 1: Alzheimer’s): {int(prediction[0][0] > 0.5)}")
    y -= 20
    c.drawString(100, y, f"Confidence Score: {confidence:.2f}")
    
    c.save()
    
    buffer.seek(0)
    return buffer

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
cholesterol_triglycerides = st.slider('Triglycerides (mg/dL)', 50, 400)

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

columns_to_scale = ['Age', 'BMI', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides']

# Separate columns for scaling
input_data_scaled = input_data.copy()

# Scale only specified columns
input_data_scaled[columns_to_scale] = scaler.fit_transform(input_data[columns_to_scale])

# Combine scaled columns with unscaled columns
# Drop the original unscaled columns and merge with the scaled ones
scaled_data_full = pd.concat([input_data_scaled[columns_to_scale], 
                               input_data.drop(columns=columns_to_scale)], axis=1)

# Predict
st.header('Prediction')
st.write("**Prediction Explanation**")
st.write("The prediction indicates the likelihood of Alzheimer’s Disease based on the provided inputs.")
st.write("A value of 0 means 'No Alzheimer’s Disease' and a value of 1 means 'Alzheimer’s Disease'.")

if st.button('Predict'):
    prediction = model.predict(scaled_data_full)
    confidence = prediction[0][0]
    result = int(confidence > 0.5)
    
    # Show prediction result
    st.write(f'Prediction: {result}')
    st.write(f'Confidence Score: {confidence:.2f}')
    
    # Generate and display the PDF report
    pdf_bytes = create_pdf_report(input_data, prediction, confidence, name=name, email=email)

    # Create a download button
    pdf_buffer = create_pdf_report(input_data, prediction, confidence, name=name, email=email)

    pdf_bytes = create_pdf_report(input_data, prediction, confidence, name=name, email=email)

    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="diagnosis_report.pdf",
        mime="application/pdf"
    )
