import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Prediction App")
st.write("Predict whether a passenger would survive the Titanic disaster.")

# Load model
model = joblib.load("titanic_model.pkl")

# ---- USER INPUTS ----
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["male", "female"])
age = st.slider("Age", 1, 80, 30)
fare = st.slider("Ticket Fare", 0.0, 500.0, 50.0)
sibsp = st.number_input("Siblings / Spouse", 0, 5, 0)
parch = st.number_input("Parents / Children", 0, 5, 0)
embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

# 🔥 NEW: Title input
title = st.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Rare"])

# ---- PREDICTION ----
if st.button("Predict Survival"):

    # Encoding
    sex = 1 if sex == "female" else 0
    embarked = {"C": 0, "Q": 1, "S": 2}[embarked]
    title_map = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}
    title = title_map[title]

    # Feature engineering
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0

    # Input DataFrame (EXACT same columns as training)
    input_df = pd.DataFrame([[
        pclass,
        sex,
        age,
        sibsp,
        parch,
        fare,
        embarked,
        title,
        family_size,
        is_alone
    ]], columns=[
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Title",
        "FamilySize",
        "IsAlone"
    ])

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("🎉 Passenger is likely to SURVIVE")
    else:
        st.error("❌ Passenger is NOT likely to survive")
