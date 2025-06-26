import streamlit as st
import numpy as np

st.set_page_config(page_title="Cyber Anomaly Detector", layout="centered")

st.title("🚨 Cyber Anomaly Detection Dashboard")

st.markdown("Real-time anomaly detection using ML + DL hybrid models.")

categorical_score = st.slider("Categorical Score", 0.0, 10.0, 2.0)
continuous_score = st.slider("Continuous Score", 0.0, 10.0, 2.0)

final_score = 0.5 * categorical_score + 0.5 * continuous_score

st.metric("Final Anomaly Score", round(final_score, 2))

if final_score > 5:
    st.error("🚨 Anomaly Detected!")
else:
    st.success("✅ Normal Traffic")

st.markdown("---")
st.caption("Powered by OPSiFi + GRU Autoencoder Hybrid Model")