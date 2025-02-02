import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
model_path = "Random_forest_model.pkl"  # Ensure this path matches your model's path
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Streamlit app
st.title("Heart Disease Prediction")

# Input features
st.sidebar.header("Input Features")
age = st.sidebar.slider("Age", 29, 77, 54)
sex = st.sidebar.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.sidebar.slider("Chest Pain Type (1-4)", 1, 4, 1)
trestbps = st.sidebar.slider("Resting Blood Pressure", 94, 200, 130)
chol = st.sidebar.slider("Cholesterol", 126, 564, 247)
fbs = st.sidebar.selectbox("Fasting Blood Sugar (1 = True, 0 = False)", [1, 0])
restecg = st.sidebar.slider("Resting ECG (0-2)", 0, 2, 1)
thalach = st.sidebar.slider("Max Heart Rate Achieved", 71, 202, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.sidebar.slider("ST Depression Induced", 0.0, 6.2, 1.0)
slope = st.sidebar.slider("Slope of ST Segment (1-3)", 1, 3, 2)
ca = st.sidebar.slider("Number of Major Vessels (0-3)", 0, 3, 0)
thal = st.sidebar.slider("Thalassemia (0-2)", 0, 2, 1)

# Input to model
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
})

# Align input_data columns with model's expected feature names
input_data = input_data[model.feature_names_in_]

# Debugging feature names (optional)
print("Model expects features:", model.feature_names_in_)
print("Input data features:", input_data.columns)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]  # Correct indexing
    result = "Positive for Heart Disease" if prediction == 1 else "Negative for Heart Disease"
    st.subheader("Prediction")
    st.write(result)
