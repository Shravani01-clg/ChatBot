import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re

# -----------------------------
# Load Models
# -----------------------------
model = pickle.load(open("academic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))
cluster_model = pickle.load(open("cluster_model.pkl", "rb"))

st.set_page_config(page_title="AI Academic Intelligence System")
st.title("🎓 AI Academic Intelligence System")
st.write("Describe your academic habits naturally.")

# -----------------------------
# Session Memory
# -----------------------------
if "last_input" not in st.session_state:
    st.session_state.last_input = None

# -----------------------------
# Extract Numbers
# -----------------------------
def extract_number(pattern, text):
    match = re.search(pattern, text)
    if match:
        numbers = re.findall(r"\d+\.?\d*", match.group())
        if numbers:
            return float(numbers[-1])
    return None

# -----------------------------
# Chat Input
# -----------------------------
user_input = st.chat_input("Tell me about your academic habits...")

if user_input:

    st.chat_message("user").write(user_input)
    text = user_input.lower()

    attendance = extract_number(r"attendance.*?\d+", text)
    study = extract_number(r"\d+\s*hours", text)
    gpa = extract_number(r"(gpa|cgpa).*?\d+", text)
    sleep = extract_number(r"sleep.*?\d+", text)

    provided = []
    if attendance is not None:
        provided.append("attendance")
    if study is not None:
        provided.append("study")
    if gpa is not None:
        provided.append("gpa")
    if sleep is not None:
        provided.append("sleep")

    # Convert 10-scale GPA to 4-scale
    if gpa is not None and gpa > 4:
        gpa = (gpa / 10) * 4

    # Default values (only for prediction)
    attendance = attendance if attendance is not None else 75
    study = study if study is not None else 4
    sleep = sleep if sleep is not None else 7
    gpa = gpa if gpa is not None else 3.0

    input_data = {
        "Age": 20,
        "Attendance_Pct": attendance,
        "Study_Hours_Per_Day": study,
        "Previous_GPA": gpa,
        "Sleep_Hours": sleep,
        "Social_Hours_Week": 10,
        "Gender_Male": 1
    }

    # Fill missing encoded columns
    for col in model_columns:
        if col not in input_data:
            input_data[col] = 0

    input_df = pd.DataFrame([input_data])[model_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict CGPA
    prediction = model.predict(scaled_input)[0]

    # Predict Cluster
    cluster = cluster_model.predict(input_df)[0]

    # Cluster Explanation
    if cluster == 0:
        cluster_type = "High Achiever Cluster"
    elif cluster == 1:
        cluster_type = "Moderate Consistency Cluster"
    else:
        cluster_type = "At-Risk Performance Cluster"

    # Performance Category
    if prediction < 2.5:
        category = "Below Average"
    elif prediction < 3.2:
        category = "Average"
    elif prediction < 3.7:
        category = "Good"
    else:
        category = "Excellent"

    # Feature Importance
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-3:]
    top_features = [model_columns[i] for i in top_indices]

    # Response
    response = f"""
### 📊 Predicted CGPA: **{round(prediction,2)}**
### 🏷 Performance Category: **{category}**

### 🧠 Student Segment:
You resemble students in the **{cluster_type}**.

### 🔍 Key Influencing Factors:
- {top_features[2]}
- {top_features[1]}
- {top_features[0]}

"""

    if "study" in provided:
        if study < 3:
            response += "- Increasing study hours may significantly improve performance.\n"
        else:
            response += "- Your study routine is healthy. Focus on active recall.\n"

    if "attendance" in provided:
        if attendance < 75:
            response += "- Improving attendance can enhance academic consistency.\n"

    if "sleep" in provided:
        if sleep < 6:
            response += "- Adequate sleep improves cognitive performance.\n"

    missing = []
    if "attendance" not in provided:
        missing.append("attendance")
    if "study" not in provided:
        missing.append("study hours")
    if "sleep" not in provided:
        missing.append("sleep duration")
    if "gpa" not in provided:
        missing.append("previous GPA")

    if len(missing) > 0:
        response += "\nFor more personalized insights, you may also share your " + ", ".join(missing) + ".\n"

    response += "\nYou can ask: *What if I increase study hours to 6?*"

    st.chat_message("assistant").write(response)