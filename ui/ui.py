import streamlit as st

st.title("Welcome To AutoML Project")
st.header("Motivation")
st.write("""
This project was built with one goal: **understand what happens under the hood** when you call `model.fit()` and `model.predict()` in scikit-learn.
Instead of treating ML libraries as black boxes, I implemented everything from scratch using only **NumPy and Pandas**""")
st.header("Project Description")
st.write("""
**AutoML from Scratch** is an end-to-end machine learning pipeline that automates the full workflow:

1. **Upload** a CSV dataset and specify the target column
2. **Detect** whether it's a classification or regression problem
3. **Preprocess** the data (handle missing values, scale features, encode categories)
4. **Train** all available models and evaluate them with relevant metrics
5. **Select** the best-performing model automatically

Built entirely from scratch — no scikit-learn, no AutoML libraries. Just math, NumPy, and curiosity.
""")

if st.button("Test The App"):
   st.switch_page("pages/1_Test_App.py")
st.caption("Made by Graia Ramine")

