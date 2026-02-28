import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------
# Page Config
# ----------------------------------

st.set_page_config(page_title="AI Startup Risk Analyzer", page_icon="🚀")

st.title("🚀 AI Startup Risk Analyzer")
st.write("AI-powered tool to predict startup success probability.")

# ----------------------------------
# Create Larger Synthetic Dataset
# ----------------------------------

np.random.seed(42)

data = {
    "funding_amount": np.random.randint(10000, 200000, 300),
    "team_experience": np.random.randint(0, 15, 300),
    "market_competition": np.random.randint(1, 11, 300),
}

df = pd.DataFrame(data)

df["success"] = (
    (df["funding_amount"] > 50000) &
    (df["team_experience"] > 3) &
    (df["market_competition"] < 7)
).astype(int)

X = df[["funding_amount", "team_experience", "market_competition"]]
y = df["success"]

model = RandomForestClassifier()
model.fit(X, y)

accuracy = model.score(X, y)

st.sidebar.write("📊 Model Accuracy:", round(accuracy * 100, 2), "%")

# ----------------------------------
# Generative AI Pipeline
# ----------------------------------

# ----------------------------------
# User Input
# ----------------------------------

st.subheader("Enter Startup Details")

funding = st.number_input("Funding Amount ($)", min_value=1000, step=1000)
experience = st.slider("Team Experience (Years)", 0, 15)
competition = st.slider("Market Competition Level (1 = Low, 10 = High)", 1, 10)

# ----------------------------------
# Prediction Section
# ----------------------------------

if st.button("Analyze Startup"):

    input_data = np.array([[funding, experience, competition]])

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)

    success_probability = round(prob[0][1] * 100, 2)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        result = "Success"
        st.success(f"🎉 High chance of success! ({success_probability}% probability)")
    else:
        result = "Failure"
        st.error(f"⚠ High risk of failure! ({success_probability}% probability)")

    # Chart
    st.subheader("📊 Input Analysis Chart")
    fig, ax = plt.subplots()
    ax.bar(["Funding", "Experience", "Competition"],
           [funding, experience, competition])
    st.pyplot(fig)

    # AI Explanation
    st.subheader("🤖 AI Explanation")

    if prediction[0] == 1:
        st.write("The startup shows strong funding and team experience with manageable competition.")
    else:
        st.write("The startup may struggle due to low funding, limited experience, or high competition.")