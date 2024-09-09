# Alzheimer's Disease Prediction
This project leverages machine learning to predict Alzheimer's disease based on patient health data. It includes a comprehensive pipeline for data preprocessing, feature selection, model training, and evaluation. The project uses several models including Logistic Regression, Random Forest Classifier, and a Neural Network built with Keras. Additionally, an interactive Streamlit application is provided for making real-time predictions based on user inputs.

## Features
- Data Preprocessing: Cleans and preprocesses the dataset, including feature scaling and handling categorical data.
- Feature Selection: Uses Recursive Feature Elimination (RFE) and Logistic Regression with Lasso to identify important features.
- Model Training: Trains and evaluates multiple models, including a Neural Network, to predict Alzheimer's disease.
- Streamlit App: An easy-to-use web application for real-time predictions based on health data inputs.

## Installation and Setup
- Clone the repository: Download the project files.
- Install dependencies: Install required Python libraries using pip.
- Prepare the dataset: Ensure the dataset is correctly formatted and located in the project directory.
- Train the models: Run the model training script to preprocess data, select features, and train the models.
- Run the Streamlit app: Launch the app to make predictions based on user-provided health metrics.

## Streamlit App Usage
The Streamlit application allows users to input various health-related features, such as age, cholesterol levels, BMI, and functional assessments, to predict the likelihood of Alzheimer's disease. The app also provides a downloadable report with input details and the prediction result.

## Key Technologies
- Pandas: For data handling and preprocessing.
- Scikit-learn: For model training and evaluation.
- Keras: For building and training the Neural Network.
- Matplotlib/Seaborn: For data visualization.
- Streamlit: For creating an interactive web app.

## Results and Evaluation
The models are evaluated based on accuracy and performance on test data. The best-performing model is deployed in the Streamlit app for real-time predictions.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.
