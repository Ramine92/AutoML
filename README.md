---
title: AutoML From Scratch
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker        ← this is the critical one
app_port: 7860     ← for Docker spaces, use this instead of app_file
pinned: false
---
**AutoML from Scratch** is an end-to-end machine learning pipeline that automates the full workflow:

1. **Upload** a CSV dataset and specify the target column
2. **Detect** whether it's a classification or regression problem
3. **Preprocess** the data (handle missing values, scale features, encode categories)
4. **Train** all available models and evaluate them with relevant metrics
5. **Select** the best-performing model automatically

Built entirely from scratch — no scikit-learn, no AutoML libraries. Just math, NumPy, and curiosity.
