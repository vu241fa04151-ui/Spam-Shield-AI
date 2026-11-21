
import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Spam Shield AI",
    page_icon="🛡️",
    layout="wide"
)

# Load trained model pipeline
MODEL_PATH = Path(__file__).parent / "models" / "spam_shield_model.pkl"

@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

try:
    pipeline = load_pipeline()
    model = pipeline["model"]
    vectorizer = pipeline["vectorizer"]
    feature_indices = pipeline["feature_indices"]
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()

# Dashboard styling
st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: #FFFFFF;
}

h1, h2, h3, p, label, span {
    color: #FFFFFF;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #55D6BE;
}

.subtitle {
    color: #D0D9E8;
    font-size: 16px;
}

div[data-testid="stMetric"] {
    background-color: #1D2A3D;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #34445C;
}

div[data-testid="stMetricLabel"] p {
    color: #D0D9E8 !important;
}

div[data-testid="stMetricValue"] {
    color: #55D6BE !important;
}

textarea {
    background-color: #FFFFFF !important;
    color: #111827 !important;
}

</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-title">🛡️ Spam Shield AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Email Spam and Phishing Detection</div>',
    unsafe_allow_html=True
)

st.divider()

# Model information
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Machine Learning Model", "LightGBM")

with col2:
    st.metric("Model Accuracy", "97.92%")

with col3:
    st.metric("Feature Selection", "Hybrid")

st.divider()

# Email input
st.subheader("📩 Analyze an Email")

email_text = st.text_area(
    "Paste your email message below:",
    height=200,
    placeholder="Paste your email content here..."
)

if st.button("🔍 Analyze Email", type="primary"):

    if not email_text.strip():
        st.warning("Please enter an email message first.")

    else:
        try:
            # Convert email text into TF-IDF features
            features = vectorizer.transform([email_text]).toarray()

            # Select hybrid features
            selected_features = features[:, feature_indices]

            # Predict email class
            prediction = int(model.predict(selected_features)[0])

            st.divider()
            st.subheader("📊 Analysis Result")

            # Dataset label mapping:
            # 0 = Legitimate
            # 1 = Spam / Phishing

            if prediction == 1:
                st.error("🚨 Spam / Phishing Email Detected!")
                st.write(
                    "Warning: This email is classified as spam or phishing. "
                    "Avoid clicking suspicious links or sharing personal information."
                )

            elif prediction == 0:
                st.success("✅ Legitimate Email")
                st.write(
                    "This email is classified as legitimate by the machine learning model."
                )

            else:
                st.warning(f"Unknown prediction class: {prediction}")

            # Prediction confidence
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(selected_features)[0]
                class_index = list(model.classes_).index(prediction)
                confidence = float(probabilities[class_index]) * 100

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.divider()

# Model performance
st.subheader("📈 Model Performance")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Accuracy", "97.92%")
m2.metric("Precision", "95.20%")
m3.metric("Recall", "94.85%")
m4.metric("F1 Score", "95.03%")

st.caption(
    "Results from the Hybrid Feature Selection + LightGBM experiment."
)

st.caption("🛡️ Spam Shield AI | Machine Learning Project")
