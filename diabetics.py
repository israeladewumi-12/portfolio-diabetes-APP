import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# (should fall around 0.35 – 0.55 probability)
# Load trained objects
rf_model = joblib.load(r"C:\portfolio1\project\rf_model.pkl")
scaler = joblib.load(r"C:\portfolio1\project\scaler.pkl")


st.sidebar.header("Enter Patient Data")
def user_input_features():
    Pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    Glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=200, value=120)
    BloodPressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=122, value=70)
    SkinThickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    Insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=80)
    BMI = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    DiabetesPedigreeFunction = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    Age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=30)

    data = {
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Scale user input
input_scaled = scaler.transform(input_df)

# Predict
threshold = 0.4
prob = rf_model.predict_proba(input_scaled)[:,1][0]
prediction = int(prob >= threshold)

if prob < 0.25:
    risk = "Low Risk"
elif prob < 0.45:
    risk = "Moderate Risk"
elif prob < 0.70:
    risk = "High Risk"
else:
    risk = "Very High Risk"

st.subheader("Prediction Result")
st.write(f"Predicted Outcome: {'Diabetic' if prediction==1 else 'Non-Diabetic'}")
st.write(f"Probability of Diabetes: {prob:.2f}")
st.write(f"stage risk of Diabetes: {risk}")


# Feature importance
st.subheader("Feature Importance")
feat_importances = pd.Series(rf_model.feature_importances_, index=input_df.columns).sort_values(ascending=False)

plt.figure(figsize=(6,4))
plt.barh(feat_importances.index, feat_importances.values)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.gca().invert_yaxis()
st.pyplot(plt)
