import streamlit as st
import time
import pandas as pd
import pickle

# Load the trained machine learning model
with open('log_model.pkl', 'rb') as f:
    reg = pickle.load(f)

# Define function to preprocess user input
def preprocess_input(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Convert categorical variables to numeric representations
    sex = 1 if sex == 'Male' else 0
    cp = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp)
    fbs = 1 if fbs == 'Yes' else 0
    restecg = ['Normal', 'LV Hypertrophy', 'Other'].index(restecg)
    exang = 1 if exang == 'Yes' else 0
    slope = ['Downsloping', 'Flat', 'Upsloping'].index(slope)
    thal = ['Normal', 'Fixed Defect', 'Reversible Defect', 'Other'].index(thal)

    # Return preprocessed input as a list
    return [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Define function to make prediction
def predict_heart_disease(preprocessed_input):
    # Simulate long-running computation
    time.sleep(2)  # Simulate computation time
    prediction = reg.predict([preprocessed_input])[0]
    return prediction

# Create Streamlit app
def main():
    st.title('Heart Disease Prediction 🏥')

    # Add input fields for user data
    name = st.text_input("Enter Your Name:")
    age = st.number_input('Age', 10, 100)
    sex = st.radio('Gender', ['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    trestbps = st.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
    chol = st.slider('Serum Cholesterol (mg/dl)', 100, 600, 200)
    fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
    restecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'LV Hypertrophy', 'Other'])
    thalach = st.slider('Maximum Heart Rate Achieved', 60, 220, 150)
    exang = st.radio('Exercise Induced Angina', ['No', 'Yes'])
    oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.0, 0.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Downsloping', 'Flat', 'Upsloping'])
    ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 0)
    thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect', 'Other'])

    # When user clicks the 'Predict' button
    if st.button('Predict'):
        # Display spinner while computing prediction
        with st.spinner('Predicting...'):
            # Preprocess input
            preprocessed_input = preprocess_input(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
            # Make prediction
            prediction = predict_heart_disease(preprocessed_input)
            if prediction == 1:
                st.subheader("Test result")
                st.error(f"The {name} is  having heart disease.")
            else:
                st.subheader("Test result")
                st.success(f"The {name} is  not having heart disease.")
                st.balloons()
            
            st.subheader("Test Report")
            result_text = "The person is predicted to have heart disease." if prediction == 1 else "The person is predicted to not have heart disease."
            report = f"""
                Name:                                           {name} 
                Age:                                            {age}
                Gender:                                         {sex} 
                Chest pain type:                                {cp}
                Resting Blood Pressure (mm Hg):                 {trestbps}
                Serum Cholesterol (mg/dl):                      {chol}
                Fasting Blood Sugar > 120 mg/dl:                {fbs}
                Resting Electrocardiographic Results:           {restecg}
                Maximum Heart Rate Achieved:                    {thalach}       
                Exercise Induced Angina:                        {exang}
                ST Depression Induced by Exercise:              {oldpeak}
                Slope of the Peak Exercise ST Segment:          {slope}
                Number of Major Vessels Colored by Fluoroscopy: {ca}
                Thalassemia:                                    {thal}
                Result:                                         {result_text}

                """
            st.code(report, language='python')

            st.warning(
                "Note: This M.L application only based on your details as our model achieved 89% accuracy. However, it's always recommended to consult a doctor for a comprehensive evaluation if you are dealing with any heart disease problem")

# Run the app
if __name__ == '__main__':
    main()
