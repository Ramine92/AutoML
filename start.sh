#!/bin/bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 & 
sleep 3
streamlit run ui/ui.py --server.port 7860 --server.address 0.0.0.0