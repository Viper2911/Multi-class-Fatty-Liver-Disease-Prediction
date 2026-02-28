import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="FLD Diagnostic System", layout="wide")

@st.cache_resource
def load_models():
    model = joblib.load('fld_stacking_model.pkl')
    scaler = joblib.load('fld_scaler.pkl')
    imputer = joblib.load('fld_imputer.pkl')
    return model, scaler, imputer

try:
    model, scaler, imputer = load_models()
except Exception as e:
    st.error(f"Failed to load models. The exact technical error is: {e}")
    st.warning("Please copy the error message above and paste it in our chat so we can fix the library versions!")
    st.stop()

st.title("Clinical Decision Support: Multiclass FLD Prediction")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex_input = st.selectbox("Sex", options=["Male", "Female"])
    sex = 1 if sex_input == "Male" else 0
    alb = st.number_input("Albumin (ALB)", value=35.0)
    alp = st.number_input("Alkaline Phosphatase (ALP)", value=80.0)

with col2:
    alt = st.number_input("Alanine Aminotransferase (ALT)", value=40.0)
    ast = st.number_input("Aspartate Aminotransferase (AST)", value=40.0)
    bil = st.number_input("Bilirubin (BIL)", value=1.0)
    che = st.number_input("Cholinesterase (CHE)", value=8.0)

with col3:
    chol = st.number_input("Cholesterol (CHOL)", value=5.0)
    crea = st.number_input("Creatinine (CREA)", value=80.0)
    ggt = st.number_input("Gamma-Glutamyl Transferase (GGT)", value=30.0)
    prot = st.number_input("Total Protein (PROT)", value=70.0)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Generate Diagnosis", type="primary", use_container_width=True):
    patient_data = pd.DataFrame([[age, sex, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot]],
                                columns=['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT'])
    
    patient_imputed = pd.DataFrame(imputer.transform(patient_data), columns=patient_data.columns)
    patient_scaled = pd.DataFrame(scaler.transform(patient_imputed), columns=patient_imputed.columns)
    
    prediction = model.predict(patient_scaled)[0]
    probabilities = model.predict_proba(patient_scaled)[0]
    
    stage_dict = {
        0: "Healthy (No Disease)",
        1: "Stage 1 (Early Disease / Hepatitis)",
        2: "Stage 2 (Fibrosis / Liver Scarring)",
        3: "Stage 3 (Cirrhosis / Severe Damage)"
    }
    
    st.markdown("### Diagnostic Results")
    st.success(f"**Predicted Diagnosis:** {stage_dict[prediction]}")
    
    st.markdown("#### Confidence Metrics")
    for i, prob in enumerate(probabilities):
        st.progress(float(prob), text=f"{stage_dict[i]}: {prob * 100:.2f}%")
