"""
MedInsight AI Clinical Decision Support Dashboard
===================================================
Integrates all 5 models into a single web interface using Streamlit.

Run locally:  streamlit run webapp/app.py
"""
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="MedInsight AI Dashboard",
    page_icon="🏥",
    layout="wide",
)

st.title("MedInsight AI Clinical Decision Support")
st.markdown("*Advancing Diabetes Care Excellence through AI*")

model_choice = st.sidebar.selectbox(
    "Choose a Model",
    [
        "Home",
        "Model 1: Readmission Risk (XGBoost)",
        "Model 2: Readmission Risk (Deep Learning)",
        "Model 3: Retinopathy Detection (CNN)",
        "Model 4: Drug Review Analysis (NLP)",
        "Model 5: Length of Stay Prediction",
    ],
)


# ── Model Loading (cached) ──

@st.cache_resource
def load_model1():
    import joblib
    model_dir = PROJECT_ROOT / "models" / "model1_traditional_ml" / "saved_model"
    model = joblib.load(model_dir / "model.joblib")
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    feature_cols = joblib.load(model_dir / "feature_cols.joblib")
    return model, preprocessor, feature_cols


@st.cache_resource
def load_model2():
    import joblib
    import onnxruntime as ort
    model_dir = PROJECT_ROOT / "models" / "model2_deep_learning" / "saved_model"
    session = ort.InferenceSession(str(model_dir / "model.onnx"))
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    feature_cols = joblib.load(model_dir / "feature_cols.joblib")
    return session, preprocessor, feature_cols


@st.cache_resource
def load_model3():
    import onnxruntime as ort
    model_dir = PROJECT_ROOT / "models" / "model3_cnn" / "saved_model"
    return ort.InferenceSession(str(model_dir / "model.onnx"))


@st.cache_resource
def load_model4():
    import joblib
    model_dir = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model"
    model = joblib.load(model_dir / "model.joblib")
    vectorizer = joblib.load(model_dir / "vectorizer.joblib")
    return model, vectorizer


@st.cache_resource
def load_model5():
    import joblib
    model_dir = PROJECT_ROOT / "models" / "model5_innovation" / "saved_model"
    model = joblib.load(model_dir / "model.joblib")
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    feature_cols = joblib.load(model_dir / "feature_cols.joblib")
    label_encoder = joblib.load(model_dir / "label_encoder.joblib")
    return model, preprocessor, feature_cols, label_encoder


# ── Helper Functions ──

def preprocess_single_encounter(inputs):
    """Build a single-row DataFrame from user inputs for encounter models."""
    from pipelines.data_pipeline import clean_encounters

    row = {
        "encounter_id": 0, "patient_nbr": 0,
        "race": inputs["race"], "gender": inputs["gender"],
        "age": inputs["age"], "weight": "?",
        "admission_type_id": inputs["admission_type_id"],
        "discharge_disposition_id": inputs["discharge_disposition_id"],
        "admission_source_id": inputs["admission_source_id"],
        "time_in_hospital": inputs["time_in_hospital"],
        "payer_code": "?", "medical_specialty": inputs.get("medical_specialty", "?"),
        "num_lab_procedures": inputs["num_lab_procedures"],
        "num_procedures": inputs["num_procedures"],
        "num_medications": inputs["num_medications"],
        "number_outpatient": inputs["number_outpatient"],
        "number_emergency": inputs["number_emergency"],
        "number_inpatient": inputs["number_inpatient"],
        "diag_1": inputs.get("diag_1", "250"),
        "diag_2": inputs.get("diag_2", "?"),
        "diag_3": inputs.get("diag_3", "?"),
        "number_diagnoses": inputs["number_diagnoses"],
        "max_glu_serum": inputs["max_glu_serum"],
        "A1Cresult": inputs["A1Cresult"],
        "change": inputs["change"],
        "diabetesMed": inputs["diabetesMed"],
        "insulin": inputs["insulin"],
        "readmitted": "NO",
    }
    # Add medication columns as "No"
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
        "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]
    for col in med_cols:
        if col not in row:
            row[col] = "No"

    df = pd.DataFrame([row])
    df = clean_encounters(df, filter_deceased=False, create_target=False)
    return df


def encounter_input_form():
    """Shared input form for encounter-based models."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        age = st.selectbox("Age Group", [
            "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
            "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
        ], index=6)
        gender = st.selectbox("Gender", ["Male", "Female"])
        race = st.selectbox("Race", [
            "Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"
        ])

    with col2:
        st.subheader("Admission Details")
        admission_type = st.selectbox("Admission Type", [
            (1, "Emergency"), (2, "Urgent"), (3, "Elective"), (7, "Trauma")
        ], format_func=lambda x: x[1])
        discharge_disp = st.selectbox("Discharge Disposition", [
            (1, "Home"), (3, "SNF"), (6, "Home with Health Service"),
            (2, "Other Hospital"), (22, "Rehab")
        ], format_func=lambda x: x[1])
        admission_source = st.selectbox("Admission Source", [
            (7, "Emergency Room"), (1, "Physician Referral"),
            (4, "Hospital Transfer"), (2, "Clinic Referral")
        ], format_func=lambda x: x[1])
        time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 4)

    with col3:
        st.subheader("Clinical Metrics")
        num_lab = st.slider("Lab Procedures", 0, 100, 40)
        num_procedures = st.slider("Procedures", 0, 10, 1)
        num_medications = st.slider("Medications", 0, 50, 15)
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 7)

    col4, col5 = st.columns(2)
    with col4:
        st.subheader("Prior History")
        n_outpatient = st.number_input("Prior Outpatient Visits", 0, 50, 0)
        n_emergency = st.number_input("Prior Emergency Visits", 0, 20, 0)
        n_inpatient = st.number_input("Prior Inpatient Stays", 0, 20, 0)

    with col5:
        st.subheader("Diabetes Metrics")
        a1c = st.selectbox("HbA1c Result", ["None", "Norm", ">7", ">8"])
        glucose = st.selectbox("Max Glucose Serum", ["None", "Norm", ">200", ">300"])
        insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
        change = st.selectbox("Medication Changed", ["No", "Ch"])
        diabetes_med = st.selectbox("On Diabetes Medication", ["Yes", "No"])

    return {
        "age": age, "gender": gender, "race": race,
        "admission_type_id": admission_type[0],
        "discharge_disposition_id": discharge_disp[0],
        "admission_source_id": admission_source[0],
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": n_outpatient,
        "number_emergency": n_emergency,
        "number_inpatient": n_inpatient,
        "number_diagnoses": number_diagnoses,
        "max_glu_serum": glucose,
        "A1Cresult": a1c,
        "insulin": insulin,
        "change": change,
        "diabetesMed": diabetes_med,
    }


# ── Pages ──

if model_choice == "Home":
    st.markdown("""
    ### Welcome to the MedInsight AI Clinical Decision Support Dashboard

    This platform integrates five AI models to support clinical decision-making
    for diabetic patient care:

    | Model | Purpose | Key Metric |
    |-------|---------|------------|
    | **Model 1** | Readmission Risk (XGBoost) | AUC-ROC: 0.693 |
    | **Model 2** | Readmission Risk (DNN) | AUC-ROC: 0.689 |
    | **Model 3** | Retinopathy Detection (CNN) | Accuracy: 92.6%, Sensitivity: 89.3% |
    | **Model 4** | Drug Review Classification (NLP) | Accuracy: 76.8%, F1: 0.760 |
    | **Model 5** | Length of Stay Prediction | Accuracy: 64.3%, F1: 0.636 |

    Use the sidebar to navigate between models and make real-time predictions.
    """)

    st.info("Select a model from the sidebar to begin.")


elif model_choice == "Model 1: Readmission Risk (XGBoost)":
    st.header("Model 1: Readmission Risk — XGBoost")
    st.markdown("*Predicts whether a diabetic patient will be readmitted within 30 days.*")

    inputs = encounter_input_form()

    if st.button("Predict Readmission Risk", type="primary"):
        from pipelines.data_pipeline import preprocess_encounters_for_prediction
        model, preprocessor, feature_cols = load_model1()
        df = preprocess_single_encounter(inputs)
        X, _ = preprocess_encounters_for_prediction(
            pd.DataFrame([{**inputs, "encounter_id": 0, "patient_nbr": 0,
                          "weight": "?", "payer_code": "?", "medical_specialty": "?",
                          "diag_1": "250", "diag_2": "?", "diag_3": "?",
                          "readmitted": "NO",
                          **{m: "No" for m in ["metformin","repaglinide","nateglinide",
                             "chlorpropamide","glimepiride","acetohexamide","glipizide",
                             "glyburide","tolbutamide","pioglitazone","rosiglitazone",
                             "acarbose","miglitol","troglitazone","tolazamide",
                             "examide","citoglipton","glyburide-metformin",
                             "glipizide-metformin","glimepiride-pioglitazone",
                             "metformin-rosiglitazone","metformin-pioglitazone"]},
                          "insulin": inputs["insulin"]}]),
            preprocessor, feature_cols,
        )
        proba = model.predict_proba(X)[0, 1]
        pred = int(proba >= 0.5)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if pred == 1:
                st.error(f"HIGH RISK — Readmission Probability: {proba:.1%}")
            else:
                st.success(f"LOW RISK — Readmission Probability: {proba:.1%}")
        with col_r2:
            st.metric("Confidence", f"{max(proba, 1-proba):.1%}")

    # Show SHAP plot if available
    shap_path = PROJECT_ROOT / "models" / "model1_traditional_ml" / "saved_model" / "shap_importance.png"
    if shap_path.exists():
        with st.expander("View Feature Importance (SHAP)"):
            st.image(str(shap_path), caption="SHAP Feature Importance")


elif model_choice == "Model 2: Readmission Risk (Deep Learning)":
    st.header("Model 2: Readmission Risk — Deep Neural Network")
    st.markdown("*Same task as Model 1 but using a deep learning approach.*")

    inputs = encounter_input_form()

    if st.button("Predict Readmission Risk (DNN)", type="primary"):
        from pipelines.data_pipeline import preprocess_encounters_for_prediction
        session, preprocessor, feature_cols = load_model2()
        X, _ = preprocess_encounters_for_prediction(
            pd.DataFrame([{**inputs, "encounter_id": 0, "patient_nbr": 0,
                          "weight": "?", "payer_code": "?", "medical_specialty": "?",
                          "diag_1": "250", "diag_2": "?", "diag_3": "?",
                          "readmitted": "NO",
                          **{m: "No" for m in ["metformin","repaglinide","nateglinide",
                             "chlorpropamide","glimepiride","acetohexamide","glipizide",
                             "glyburide","tolbutamide","pioglitazone","rosiglitazone",
                             "acarbose","miglitol","troglitazone","tolazamide",
                             "examide","citoglipton","glyburide-metformin",
                             "glipizide-metformin","glimepiride-pioglitazone",
                             "metformin-rosiglitazone","metformin-pioglitazone"]},
                          "insulin": inputs["insulin"]}]),
            preprocessor, feature_cols,
        )
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: np.array(X, dtype=np.float32)})
        proba = float(result[0].flatten()[0])
        pred = int(proba >= 0.5)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if pred == 1:
                st.error(f"HIGH RISK — Readmission Probability: {proba:.1%}")
            else:
                st.success(f"LOW RISK — Readmission Probability: {proba:.1%}")
        with col_r2:
            st.metric("Confidence", f"{max(proba, 1-proba):.1%}")

    # Show training curves if available
    curves_path = PROJECT_ROOT / "models" / "model2_deep_learning" / "saved_model" / "training_curves.png"
    if curves_path.exists():
        with st.expander("View Training Curves"):
            st.image(str(curves_path), caption="DNN Training Curves")


elif model_choice == "Model 3: Retinopathy Detection (CNN)":
    st.header("Model 3: Diabetic Retinopathy Detection — CNN")
    st.markdown("*Upload a retinal fundus image to screen for diabetic retinopathy.*")

    uploaded_file = st.file_uploader(
        "Upload a retinal scan image", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        from PIL import Image

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Retinal Scan", width=400)

        if st.button("Analyze Image", type="primary"):
            session = load_model3()
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            input_name = session.get_inputs()[0].name
            result = session.run(None, {input_name: img_batch})
            proba = float(result[0].flatten()[0])
            pred = int(proba >= 0.5)
            confidence = max(proba, 1 - proba)

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if pred == 1:
                    st.error(f"DIABETIC RETINOPATHY DETECTED")
                    st.markdown("*Recommend ophthalmology referral.*")
                else:
                    st.success(f"NO DIABETIC RETINOPATHY DETECTED")
                    st.markdown("*Continue routine screening schedule.*")
            with col_r2:
                st.metric("Confidence", f"{confidence:.1%}")
                st.metric("DR Probability", f"{proba:.1%}")


elif model_choice == "Model 4: Drug Review Analysis (NLP)":
    st.header("Model 4: Drug Review Effectiveness — NLP")
    st.markdown("*Classify patient drug reviews into effectiveness categories.*")

    user_text = st.text_area(
        "Enter a patient drug review:",
        height=150,
        placeholder="e.g., 'This medication helped my blood sugar levels significantly. "
                    "I noticed improvement within the first week. Minor side effects included "
                    "some nausea but it went away after a few days.'"
    )

    if st.button("Classify Review", type="primary") and user_text:
        from pipelines.data_pipeline import clean_review_text
        model, vectorizer = load_model4()

        cleaned = clean_review_text(user_text)
        X = vectorizer.transform([cleaned])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        confidence = float(np.max(proba))

        classes = model.classes_
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            color_map = {
                "Highly Effective": "success",
                "Somewhat Effective": "warning",
                "Ineffective": "error",
            }
            getattr(st, color_map.get(pred, "info"))(
                f"Predicted: **{pred}** (Confidence: {confidence:.1%})"
            )
        with col_r2:
            st.markdown("**Class Probabilities:**")
            for cls, p in zip(classes, proba):
                st.progress(float(p), text=f"{cls}: {p:.1%}")


elif model_choice == "Model 5: Length of Stay Prediction":
    st.header("Model 5: Length of Stay Prediction")
    st.markdown("""
    *Predicts hospital length of stay category to optimize bed management and discharge planning.*

    **Clinical Value:** Enables proactive resource allocation for 180,000+ diabetic patients.
    Estimated annual savings of $62M through 10% LOS optimization.
    """)

    inputs = encounter_input_form()

    if st.button("Predict Length of Stay", type="primary"):
        from pipelines.data_pipeline import preprocess_encounters_for_prediction
        model, preprocessor, feature_cols, label_encoder = load_model5()
        X, _ = preprocess_encounters_for_prediction(
            pd.DataFrame([{**inputs, "encounter_id": 0, "patient_nbr": 0,
                          "weight": "?", "payer_code": "?", "medical_specialty": "?",
                          "diag_1": "250", "diag_2": "?", "diag_3": "?",
                          "readmitted": "NO",
                          **{m: "No" for m in ["metformin","repaglinide","nateglinide",
                             "chlorpropamide","glimepiride","acetohexamide","glipizide",
                             "glyburide","tolbutamide","pioglitazone","rosiglitazone",
                             "acarbose","miglitol","troglitazone","tolazamide",
                             "examide","citoglipton","glyburide-metformin",
                             "glipizide-metformin","glimepiride-pioglitazone",
                             "metformin-rosiglitazone","metformin-pioglitazone"]},
                          "insulin": inputs["insulin"]}]),
            preprocessor, feature_cols,
        )
        proba = model.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        pred = label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx])

        los_labels = {"short_stay": "1-3 days", "medium_stay": "4-7 days", "long_stay": "8+ days"}
        los_colors = {"short_stay": "success", "medium_stay": "warning", "long_stay": "error"}

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            getattr(st, los_colors.get(pred, "info"))(
                f"Predicted: **{los_labels.get(pred, pred)}** (Confidence: {confidence:.1%})"
            )
        with col_r2:
            st.markdown("**Category Probabilities:**")
            for cls_idx, cls in enumerate(label_encoder.classes_):
                st.progress(float(proba[cls_idx]),
                           text=f"{los_labels.get(cls, cls)}: {proba[cls_idx]:.1%}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("*MedInsight Healthcare AI Platform*")
st.sidebar.markdown("*Advancing diabetes care through AI innovation*")
