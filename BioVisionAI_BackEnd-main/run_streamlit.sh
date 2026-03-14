#!/bin/bash
# Run Streamlit demo from project root
cd "$(dirname "$0")"
streamlit run frontend/app.py --server.port 8501
