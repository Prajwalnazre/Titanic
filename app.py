import streamlit as st
import pandas as pd
import pickle
import sklearn as skl

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# App title
st.title("🚢 Titanic Survival Prediction")

# User input
title = st.text_input("Title : ")
pclass = st.selectbox("Passenger Class : ", [1, 2, 3])
sex = st.selectbox("Sex : ", ['male', 'female'])
age = st.slider("Age : ", 1, 80)
sibsp = st.slider("Siblings/Spouses Aboard : ", 0, 8)
parch = st.slider("Parents/Children Aboard : ", 0, 6)
embarked = st.selectbox("Passenger Class : ", ['S', 'C', 'Q'])
fare = st.number_input("Fare : ", min_value=0.0, max_value=600.0)

# Convert input to dataframe (match your model's features)
input_df = pd.DataFrame([{
    'Title' : title,
    'Pclass': pclass,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked
}])

print("Processing your inputs.....\n")

# Creating new family_size column
input_df['Family_Size']=input_df['SibSp']+input_df['Parch']+1

categorical_features_pred = input_df.select_dtypes(exclude=['number'])

def encode_categorical_features(categorical_features) :
    # label_encoders = {}
    X_encoded = categorical_features.copy()
    i = 0
    for col in categorical_features:
        print("Encoding ", categorical_features.columns[i])
        # le = skl.preprocessing.LabelEncoder()
        # X_encoded[col] = le.fit_transform(X_encoded[col])
        # label_encoders[col] = le
        X_encoded_dummies = pd.get_dummies(categorical_features[col], prefix=categorical_features.columns[i])
        X_encoded = pd.concat([X_encoded, X_encoded_dummies], axis=1)
        X_encoded.drop(col, axis=1, inplace=True)
        i += 1
        
    return X_encoded

categorical_features_encoded_pred = encode_categorical_features(categorical_features_pred)

X_Pred = pd.concat([input_df, categorical_features_encoded_pred], axis=1)

# Add new feature Is_Alone
X_Pred['Is_Alone'] = False
X_Pred.loc[X_Pred['Family_Size'] == 1, 'Is_Alone'] = True

# Add new feature Age_Missing
X_Pred['Age_Missing'] = X_Pred['Age'].isnull().astype(int)

scaler = skl.preprocessing.StandardScaler()
X_Pred[['Age','SibSp','Parch','Fare','Family_Size']] = scaler.transform(X_Pred[['Age','SibSp','Parch','Fare','Family_Size']])

print(X_Pred.dtypes)

# Predict button
if st.button("Predict"):
    prediction = model.predict(X_Pred)[0]
    result = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"
    st.subheader(result)