import streamlit as st
import pandas as pd
import requests
import json
import time
import plotly.express as px
st.title("Test AutoML")
st.header("Upload CSV")
uploaded_file = st.file_uploader("Choose a CSV file",type="csv")
if uploaded_file is not None:
    st.subheader("Data Preview")
    df_preview = pd.read_csv(uploaded_file)
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
            response = requests.post("http://localhost:8000/run",files={"file":uploaded_file},data={"target_column":target_column},timeout=30)
            result = response.json()
            job_id = response.json()["job_id"]
            with st.spinner("Training models...."):
                while True:
                    status_resp = requests.get(f"http://localhost:8000/status/{job_id}").json()
                    if status_resp["status"] == "done":
                        result = status_resp
                        break
                    elif status_resp["status"] == "failed":
                        st.error(f"Failed:{status_resp['error']}")
                        break
                    time.sleep(5)  # poll every 5 seconds
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
            df_melted = df_metrics.melt(id_vars="Model",var_name="Metric",value_name="Score")
            fig = px.bar(df_melted,x="Model",y="Score",color="Metric",barmode="group",title="Model Comparison",range_y=[0,1],text_auto=".2f")
            st.plotly_chart(fig,use_container_width=True)
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