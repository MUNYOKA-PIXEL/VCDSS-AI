import streamlit as st
import joblib
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime

# 1. Page Configuration
st.set_page_config(
    page_title="VCDSS - Clinical Decision Support System",
    page_icon="🐄",
    layout="wide"
)

# 2. Load Stacking Ensemble Model and Features
@st.cache_resource
def load_vcdss_model():
    model_filename = 'vcdss_stacking_ensemble_model.pkl'
    model = joblib.load(model_filename)
    # Extract feature names from the underlying estimator
    features = model.named_estimators_['rf'].feature_names_in_
    classes = model.classes_
    return model, features, classes

try:
    loaded_model, expected_features, target_classes = load_vcdss_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Ensure 'vcdss_stacking_ensemble_model.pkl' is in the same directory.")
    st.stop()

# 3. Sidebar Navigation Menu
st.sidebar.title("🐄 VCDSS Menu")
menu = ["Prediction", "Model Explanation", "Case History"]
choice = st.sidebar.selectbox("Navigate To:", menu)

# History File Setup
HISTORY_FILE = "case_history.csv"

# ==============================================================================
# TAB 1: PREDICTION
# ==============================================================================
if choice == "Prediction":
    st.title("🐄 Veterinary Diagnosis Prediction")
    st.markdown("Enter patient vitals and check observed clinical signs to run an ensemble model prediction.")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Patient Vitals")
        selected_species = st.selectbox("Animal Species", ["cow", "sheep", "goat"])
        age_input = st.number_input("Age (Years)", min_value=0.1, max_value=20.0, value=3.0, step=0.5)
        temp_input = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=45.0, value=40.0, step=0.1)

    with col2:
        st.subheader("Observed Clinical Symptoms")
        st.caption("Select all symptoms currently present:")
        
        # Extract individual symptom names from model binary features
        symptom_cols = [c for c in expected_features if "symptom" in c.lower()]
        
        # Deduplicate symptoms across Symptom 1, 2, 3 slots for UI display
        unique_symptom_names = sorted(list(set(
            c.split('_', 1)[1] if '_' in c else c for c in symptom_cols
        )))

        # Display symptom checkboxes across 2 sub-columns
        selected_symptoms = []
        sym_col1, sym_col2 = st.columns(2)
        for i, sym_name in enumerate(unique_symptom_names):
            target_col = sym_col1 if i % 2 == 0 else sym_col2
            if target_col.checkbox(sym_name.capitalize(), key=f"chk_{sym_name}"):
                selected_symptoms.append(sym_name)

    st.markdown("---")

    # Predict Button Logic
    if st.button("Run Diagnostic Prediction", type="primary", use_container_width=True):
        # Build zeroed input DataFrame with all expected model features
        input_df = pd.DataFrame(0, index=[0], columns=expected_features)
        
        # Set Vitals & Species
        if 'Age' in input_df.columns:
            input_df['Age'] = age_input
        if 'Temperature' in input_df.columns:
            input_df['Temperature'] = temp_input
        
        species_col = f"Animal_{selected_species}"
        if species_col in input_df.columns:
            input_df[species_col] = 1

        # Turn on selected symptoms across all symptom slot columns (Symptom 1, 2, 3)
        for sym_name in selected_symptoms:
            matching_cols = [c for c in symptom_cols if sym_name.lower() in c.lower()]
            for mc in matching_cols:
                input_df[mc] = 1

        # Run Ensemble Model Prediction
        prediction = loaded_model.predict(input_df)[0]
        pred_probs = loaded_model.predict_proba(input_df)[0]
        confidence = np.max(pred_probs) * 100

        # Display Results
        st.subheader("Diagnostic Results")
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if confidence > 80:
                st.success(f"### Diagnosis: **{str(prediction).upper()}**\n**Confidence Score:** {confidence:.2f}%")
            elif confidence > 50:
                st.warning(f"### Diagnosis: **{str(prediction).upper()}**\n**Confidence Score:** {confidence:.2f}% (Moderate)")
            else:
                st.error(f"### Diagnosis: **{str(prediction).upper()}**\n**Confidence Score:** {confidence:.2f}% (Low Confidence)")

        with res_col2:
            st.markdown("**Confidence Distribution Across All Classes:**")
            prob_df = pd.DataFrame({
                "Condition": target_classes,
                "Probability (%)": [p * 100 for p in pred_probs]
            }).set_index("Condition")
            st.bar_chart(prob_df)

        # Log case history to CSV
        case_data = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Species": selected_species,
            "Age": age_input,
            "Temperature_C": temp_input,
            "Symptoms_Observed": ", ".join(selected_symptoms) if selected_symptoms else "None",
            "Diagnosis": prediction,
            "Confidence": f"{confidence:.2f}%"
        }
        case_df = pd.DataFrame([case_data])

        if os.path.exists(HISTORY_FILE):
            history = pd.read_csv(HISTORY_FILE)
            history = pd.concat([history, case_df], ignore_index=True)
        else:
            history = case_df
        history.to_csv(HISTORY_FILE, index=False)
        st.toast("Case recorded in history!", icon="💾")

# ==============================================================================
# TAB 2: MODEL EXPLANATION
# ==============================================================================
elif choice == "Model Explanation":
    st.title("🧠 Stacking Model Explanation")
    
    st.write("""
    ### How the Stacking Ensemble Model Works
    This system uses a **Stacking Classifier Ensemble** containing multiple base classifiers (e.g., Random Forest, Extra Trees, Gradient Boosting) combined via a Meta-Estimator:
    - **Feature Evaluation:** Checks vital flags (Temperature, Age) alongside species-specific symptom profiles.
    - **Consensus Predictions:** Each sub-model evaluates the case and assigns probability scores.
    - **Final Classification:** The stacking meta-learner blends these outputs to produce a high-confidence diagnostic decision.
    """)

    st.markdown("---")
    st.subheader("Random Forest Base Estimator Feature Importance")

    # Extract feature importance from the underlying RF estimator
    try:
        rf_model = loaded_model.named_estimators_['rf']
        importances = rf_model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": expected_features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        # Display Top 15 Features
        top_importance = importance_df.head(15).set_index("Feature")
        
        st.write("Top 15 decision-making features identified by the Random Forest estimator:")
        st.bar_chart(top_importance)

        st.subheader("Full Feature Importance Data Table")
        st.dataframe(importance_df, use_container_width=True)

        # Download Feature Importance Excel Export
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            importance_df.to_excel(writer, sheet_name="Feature Importance", index=False)
            
        st.download_button(
            label="Download Feature Importance (Excel)",
            data=excel_buffer.getvalue(),
            file_name="feature_importance.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.info("Feature importance view is only available when the Stacking Classifier exposes named base estimators.")

# ==============================================================================
# TAB 3: CASE HISTORY
# ==============================================================================
elif choice == "Case History":
    st.title("📋 Recorded Case History")

    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE)

        # Sort Controls
        sort_col1, sort_col2 = st.columns([2, 1])
        with sort_col1:
            sort_by = st.selectbox("Sort by column:", history.columns.tolist())
        with sort_col2:
            sort_order = st.radio("Order:", ["Descending", "Ascending"])

        history_sorted = history.sort_values(
            by=sort_by, 
            ascending=(sort_order == "Ascending")
        )
        st.dataframe(history_sorted, use_container_width=True)

        st.markdown("---")
        st.subheader("Diagnosis Analytics")
        
        # Diagnosis Distribution Bar Chart
        diag_counts = history["Diagnosis"].value_counts().reindex(target_classes, fill_value=0)
        st.bar_chart(diag_counts)

        # Action Buttons & Downloads
        st.markdown("---")
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

        # CSV Download
        csv_buffer = io.BytesIO()
        history_sorted.to_csv(csv_buffer, index=False)
        btn_col1.download_button(
            label="Download History (CSV)",
            data=csv_buffer.getvalue(),
            file_name="case_history.csv",
            mime="text/csv"
        )

        # Excel Download
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            history_sorted.to_excel(writer, sheet_name="Case History", index=False)
        btn_col2.download_button(
            label="Download History (Excel)",
            data=excel_buffer.getvalue(),
            file_name="case_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Clear History Option
        if btn_col3.button("Clear History Log", type="primary"):
            os.remove(HISTORY_FILE)
            st.warning("Case history log has been cleared.")
            st.rerun()
            
    else:
        st.info("No saved cases found. Run a prediction under the 'Prediction' menu tab to begin logging records.")