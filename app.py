"""
Streamlit Web Application for CKD Classification
==================================================
Accepts patient clinical parameters via a form and predicts
whether the patient has Chronic Kidney Disease or not.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    label_encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    best_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl"))
    return model, scaler, label_encoders, feature_names, best_name


def main():
    # --- Page config ---
    st.set_page_config(
        page_title="CKD Classifier",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- Custom CSS ---
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
    }
    .main-header p {
        color: #e0e0e0;
        font-size: 1.1rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .ckd-positive {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
    }
    .ckd-negative {
        background: linear-gradient(135deg, #26de81, #20bf6b);
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Chronic Kidney Disease Classifier</h1>
        <p>Enter patient clinical parameters to predict CKD status</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Load model ---
    try:
        model, scaler, label_encoders, feature_names, best_name = load_model()
        st.sidebar.success(f"✅ Model loaded: **{best_name}**")
    except Exception as e:
        st.error(f"⚠️ Model not found. Run `python src/train.py` first.\n\n{e}")
        return

    # --- Sidebar: info ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 About")
    st.sidebar.info(
        "This application classifies patients as **CKD** (Chronic Kidney Disease) "
        "or **Not CKD** based on 24 clinical parameters using a trained ML model."
    )

    # --- Input Form ---
    st.markdown("### 📝 Patient Clinical Parameters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**🔬 Blood & Urine**")
        age = st.number_input("Age (years)", 1, 120, 50, key="age")
        bp = st.number_input("Blood Pressure (mm/Hg)", 40, 200, 80, key="bp")
        sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=2, key="sg")
        al = st.selectbox("Albumin (0-5)", [0, 1, 2, 3, 4, 5], index=0, key="al")
        su = st.selectbox("Sugar (0-5)", [0, 1, 2, 3, 4, 5], index=0, key="su")
        bgr = st.number_input("Blood Glucose Random (mg/dl)", 50, 500, 120, key="bgr")

    with col2:
        st.markdown("**🩸 Serum Values**")
        bu = st.number_input("Blood Urea (mg/dl)", 1.0, 400.0, 40.0, key="bu")
        sc = st.number_input("Serum Creatinine (mg/dl)", 0.1, 30.0, 1.2, key="sc")
        sod = st.number_input("Sodium (mEq/L)", 100.0, 170.0, 140.0, key="sod")
        pot = st.number_input("Potassium (mEq/L)", 2.0, 10.0, 4.5, key="pot")
        hemo = st.number_input("Hemoglobin (gms)", 3.0, 20.0, 13.0, key="hemo")
        pcv = st.number_input("Packed Cell Volume", 10.0, 55.0, 40.0, key="pcv")

    with col3:
        st.markdown("**🧬 Cell Counts**")
        wc = st.number_input("White Blood Cell Count", 2000, 30000, 8000, key="wc")
        rc = st.number_input("Red Blood Cell Count (millions)", 2.0, 8.0, 5.0, key="rc")
        rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"], key="rbc")
        pc = st.selectbox("Pus Cell", ["normal", "abnormal"], key="pc")
        pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"], key="pcc")
        ba = st.selectbox("Bacteria", ["notpresent", "present"], key="ba")

    with col4:
        st.markdown("**🏥 Medical History**")
        htn = st.selectbox("Hypertension", ["no", "yes"], key="htn")
        dm = st.selectbox("Diabetes Mellitus", ["no", "yes"], key="dm")
        cad = st.selectbox("Coronary Artery Disease", ["no", "yes"], key="cad")
        appet = st.selectbox("Appetite", ["good", "poor"], key="appet")
        pe = st.selectbox("Pedal Edema", ["no", "yes"], key="pe")
        ane = st.selectbox("Anemia", ["no", "yes"], key="ane")

    st.markdown("---")

    # --- Predict ---
    if st.button("🔍 Predict CKD Status", width="stretch", type="primary"):
        # Build input dict
        input_data = {
            "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
            "rbc": rbc, "pc": pc, "pcc": pcc, "ba": ba,
            "bgr": bgr, "bu": bu, "sc": sc, "sod": sod, "pot": pot,
            "hemo": hemo, "pcv": pcv, "wc": wc, "rc": rc,
            "htn": htn, "dm": dm, "cad": cad, "appet": appet,
            "pe": pe, "ane": ane
        }

        # Encode categorical features
        for col, le in label_encoders.items():
            if col in input_data:
                input_data[col] = le.transform([input_data[col]])[0]

        # Create DataFrame in correct feature order
        input_df = pd.DataFrame([input_data])[feature_names]

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0] if hasattr(model, "predict_proba") else None

        # Display result
        st.markdown("### 📊 Prediction Result")

        if prediction == 1:
            confidence = proba[1] * 100 if proba is not None else 0
            st.markdown(f"""
            <div class="result-box ckd-positive">
                <h2>🔴 CKD DETECTED</h2>
                <p style="font-size: 1.3rem;">The patient is likely to have Chronic Kidney Disease</p>
                <p style="font-size: 1.1rem;">Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            confidence = proba[0] * 100 if proba is not None else 0
            st.markdown(f"""
            <div class="result-box ckd-negative">
                <h2>🟢 NO CKD DETECTED</h2>
                <p style="font-size: 1.3rem;">The patient is unlikely to have Chronic Kidney Disease</p>
                <p style="font-size: 1.1rem;">Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        # Show probabilities
        if proba is not None:
            col_a, col_b = st.columns(2)
            col_a.metric("Not CKD Probability", f"{proba[0]*100:.1f}%")
            col_b.metric("CKD Probability", f"{proba[1]*100:.1f}%")

    # --- Analysis Section ---
    st.markdown("---")
    st.markdown("### 📈 Model Analysis & Visualisations")

    tabs = st.tabs([
        "📊 Metrics Comparison",
        "🔄 Confusion Matrices",
        "📈 ROC Curves",
        "🎯 Feature Importance",
        "📉 CV Scores",
        "🗂️ Data Insights"
    ])

    plot_files = {
        0: "11_metrics_comparison.png",
        1: "09_confusion_matrices.png",
        2: "10_roc_curves.png",
        3: "08_feature_importance.png",
        4: "12_cv_boxplot.png",
    }

    for i, tab in enumerate(tabs[:5]):
        with tab:
            plot_path = os.path.join(PLOTS_DIR, plot_files[i])
            if os.path.exists(plot_path):
                st.image(plot_path, width="stretch")
            else:
                st.warning("Plot not found. Run `python src/train.py` to generate.")

    with tabs[5]:
        data_plots = [
            ("01_target_distribution.png", "Target Distribution"),
            ("05_correlation_heatmap.png", "Feature Correlations"),
            ("03_missing_bar.png", "Missing Values"),
            ("06_box_plots.png", "Box Plots by Class"),
        ]
        for fname, title in data_plots:
            path = os.path.join(PLOTS_DIR, fname)
            if os.path.exists(path):
                st.markdown(f"**{title}**")
                st.image(path, width="stretch")
                st.markdown("---")

    # --- Results table ---
    results_path = os.path.join(MODELS_DIR, "results_summary.csv")
    if os.path.exists(results_path):
        st.markdown("### 📋 Full Results Table")
        results = pd.read_csv(results_path)
        st.dataframe(results, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
