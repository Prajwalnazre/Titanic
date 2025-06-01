import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# App title
st.title("🚢 Titanic Survival Prediction")

# User input
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ['male', 'female'])
age = st.slider("Age", 1, 80)
sibsp = st.slider("Siblings/Spouses Aboard", 0, 8)
parch = st.slider("Parents/Children Aboard", 0, 6)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0)

# Convert input to dataframe (match your model's features)
input_df = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': 1 if sex == 'male' else 0,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare
}])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"
    st.subheader(result)