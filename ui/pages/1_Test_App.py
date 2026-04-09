import streamlit as st
import pandas as pd
import requests
import json
st.title("Test AutoML")
st.header("Upload CSV")
uploaded_file = st.file_uploader("Choose a CSV file",type="csv")
if uploaded_file is not None:
    st.subheader("Data Preview")
    df_preview = pd.read_csv(uploaded_file)
    df_preview.reset_index(drop=True,inplace=True)
    st.dataframe(df_preview.head())
    uploaded_file.seek(0)

if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = None
if "trained" not in st.session_state:
    st.session_state.trained = False
st.header("target Column / Variable name")
target_column = st.text_input("Enter target column Name")

if uploaded_file is not None and target_column:
    if st.button("Run AutoML"):
        try:
            response = requests.post("http://localhost:8000/run",files={"file":uploaded_file},data={"target_column":target_column})
            result = response.json()
            if response.status_code != 200:
                st.error(f"API error {response.status_code}: {response.text}")
            else:
                st.success(f"Best model: {result['best_model']}")
                best_model_name = result['best_model']
                #data persistance using st.session_state
                st.session_state.trained = True
                st.session_state.best_model_name = result['best_model']
                metrics_data = []
                for model in result["results"]:
                    row = {"Model":model["model"]}
                    row.update(model["metrics"])
                    metrics_data.append(row)
                    
                df_metrics = pd.DataFrame(metrics_data)    
                if not df_metrics.empty:
                    if "acc" in df_metrics.columns:
                        st.bar_chart(data=df_metrics, x="Model", y="acc", width=400, use_container_width=False)
                    elif "R2" in df_metrics.columns:
                        st.bar_chart(data=df_metrics, x="Model", y="R2", width=400, use_container_width=False)
                st.write("Raw Metrics")
                st.dataframe(df_metrics)

        except Exception as e:
            st.error(f"Pipeline failed: {e}")
if st.session_state.trained:
        st.header("Make Prediction")
        st.subheader("Upload Test CSV")
        uploaded_test_file = st.file_uploader("Choose a Test CSV file",type="csv")
        if uploaded_test_file is not None:
            if st.button("Predict from File"):
                try:
                    response = requests.post("http://localhost:8000/predict/file",files={"data":uploaded_test_file},data={"model_name":st.session_state.best_model_name})
                    result = response.json()
                    st.write(result)
                except Exception as e:
                    st.error("Error occured please verify target_column not included")
        default_json = default_json = [{
                 "feature1": 10,
                 "feature2": 5.2,
                 "feature3": "A"
        },
        {        "feature1": 20,
                 "feature2": 3.6,
                 "feature3": "B"
        }]
        st.subheader("Predict Using Json format")
        json_data = st.text_area("Json Request Body",value=json.dumps(default_json),height=200)
        if st.button("Predict from JSON"):
            try:
                payload = {"data":json.loads(json_data),"model_name":st.session_state.best_model_name}
                response = requests.post("http://localhost:8000/predict/json",json=payload)
                result = response.json()
                st.write(result)
            except Exception as e:
                st.error("error occured please check requests format")