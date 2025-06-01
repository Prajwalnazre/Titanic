import streamlit as st
import pandas as pd
import pickle
import sklearn as skl

# Load model
model = pickle.load(open('random_forest.pkl', 'rb'))

# App title
st.title("🚢 Titanic Survival Prediction")

# User input
title = st.text_input("Title : ")
pclass = st.selectbox("Passenger Class : ", [1, 2, 3])
sex = st.selectbox("Sex : ", ['male', 'female'])
age = st.slider("Age : ", 5, 54)
sibsp = st.slider("Siblings/Spouses Aboard : ", 0, 3)
parch = st.slider("Parents/Children Aboard : ", 0, 2)
embarked = st.selectbox("Passenger Class : ", ['S', 'C', 'Q'])
fare = st.number_input("Fare : ", min_value=7, max_value=112)

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

# categorical_features_pred = input_df.select_dtypes(exclude=['number'])

# def encode_categorical_features(categorical_features) :
#     # label_encoders = {}
#     X_encoded = categorical_features.copy()
#     i = 0
#     for col in categorical_features:
#         print("Encoding ", categorical_features.columns[i])
#         # le = skl.preprocessing.LabelEncoder()
#         # X_encoded[col] = le.fit_transform(X_encoded[col])
#         # label_encoders[col] = le
#         X_encoded_dummies = pd.get_dummies(categorical_features[col], prefix=categorical_features.columns[i])
#         X_encoded = pd.concat([X_encoded, X_encoded_dummies], axis=1)
#         X_encoded.drop(col, axis=1, inplace=True)
#         i += 1
        
#     return X_encoded

# categorical_features_encoded_pred = encode_categorical_features(categorical_features_pred)

# X_Pred = pd.concat([input_df, categorical_features_encoded_pred], axis=1)

X_Pred = input_df.copy()

col_names = ["Sex_female", 
             "Sex_male", 
             "Embarked_C", 
             "Embarked_Q", 
             "Embarked_S", 
             "Pclass_1", 
             "Pclass_2",
             "Pclass_3",
             "Title_Master",
             "Title_Miss",
             "Title_Mr",
             "Title_Mrs",
             "Title_Rare_Mr"
             ]

for col_name in col_names :
    X_Pred[col_name] = False

if X_Pred["Sex"][0] == "male" :
    X_Pred["Sex_male"][0] = True

if X_Pred["Sex"][0] == "female" :
    X_Pred["Sex_female"][0] = True

if X_Pred["Embarked"][0] == "S" :
    X_Pred["Embarked_S"][0] = True

if X_Pred["Embarked"][0] == "C" :
    X_Pred["Embarked_C"][0] = True

if X_Pred["Embarked"][0] == "Q" :
    X_Pred["Embarked_Q"][0] = True

if X_Pred["Pclass"][0] == 1 :
    X_Pred["Pclass_1"][0] = True

if X_Pred["Pclass"][0] == 2 :
    X_Pred["Pclass_2"][0] = True

if X_Pred["Pclass"][0] == 3 :
    X_Pred["Pclass_3"][0] = True

if X_Pred["Title"][0] == "Master" :
    X_Pred["Title_Master"][0] = True
elif X_Pred["Title"][0] == "Miss" :
    X_Pred["Title_Miss"][0] = True
elif X_Pred["Title"][0] == "Mr" :
    X_Pred["Title_Mr"][0] = True
elif X_Pred["Title"][0] == "Mrs" :
    X_Pred["Title_Mrs"][0] = True
else :
    X_Pred["Title_Rare_Mr"][0] = True

X_Pred.drop(columns=["Title", "Pclass", "Embarked", "Sex"], inplace=True)

# Add new feature Is_Alone
X_Pred['Is_Alone'] = False
X_Pred.loc[X_Pred['Family_Size'] == 1, 'Is_Alone'] = True

# Add new feature Age_Missing
X_Pred['Age_Missing'] = X_Pred['Age'].isnull().astype(int)

with open("standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
    
X_Pred[['Age','SibSp','Parch','Fare','Family_Size']] = scaler.transform(X_Pred[['Age','SibSp','Parch','Fare','Family_Size']])

print(X_Pred.dtypes)

# Predict button
if st.button("Predict"):
    prediction = model.predict(X_Pred)[0]
    result = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"
    st.subheader(result)