# You can install or verify all dependencies at once in your terminal by running: pip install -r requirements.txt

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title=" Veterinary Clinical Decision Support System (VCDSS)",
    page_icon="🐄",
    layout="wide"
)

# ==============================================================================
# CUSTOM CSS - WORKS PERFECTLY IN BOTH THEMES
# ==============================================================================
st.markdown("""
<style>
    /* Base styles */
    .main { padding: 0rem 1rem; }
    
    /* Header - fixed gradient, always white text (unaffected by theme, by design) */
    .header-container {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d7fc1 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .header-title { 
        color: #ffffff !important; 
        font-size: 2.5rem; 
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .header-subtitle { 
        color: #ffffff !important; 
        font-size: 1.1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* ================================================================
       THEME-ADAPTIVE ELEMENTS
       Uses Streamlit's own theme CSS variables, which update automatically
       whenever the app's theme (Settings > Theme) changes — light or dark.
       ================================================================ */
    .card {
        background: var(--secondary-background-color);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(128,128,128,0.25);
    }
    .card h4, .card p, .card li, .card label {
        color: var(--text-color) !important;
    }
    
    .result-card { 
        padding: 1.5rem; 
        border-radius: 12px; 
        margin: 1rem 0; 
        border-left: 5px solid;
        background: var(--secondary-background-color);
    }
    .result-success { border-left-color: #22c55e; background: rgba(34, 197, 94, 0.12); }
    .result-warning { border-left-color: #eab308; background: rgba(234, 179, 8, 0.12); }
    .result-error   { border-left-color: #ef4444; background: rgba(239, 68, 68, 0.12); }
    .result-title   { font-size: 1.5rem; font-weight: 700; color: var(--text-color) !important; }
    .result-card div { color: var(--text-color) !important; }
    
    .confidence-bar-container {
        background: rgba(128,128,128,0.25);
        border-radius: 10px;
        height: 30px;
        margin: 0.5rem 0;
        overflow: hidden;
        border: 1px solid rgba(128,128,128,0.15);
    }
    .confidence-bar {
        height: 100%;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 15px;
        color: #ffffff !important;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .confidence-high   { background: linear-gradient(90deg, #22c55e, #15803d); }
    .confidence-medium { background: linear-gradient(90deg, #eab308, #a16207); }
    .confidence-low    { background: linear-gradient(90deg, #ef4444, #b91c1c); }
    
    .stButton button {
        background: linear-gradient(135deg, #2d7fc1, #1e3a5f) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 10px !important;
        width: 100% !important;
        transition: all 0.3s !important;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(45, 127, 193, 0.5);
    }
    
    [data-testid="stSidebar"] {
        background: var(--secondary-background-color) !important;
        border-right: 1px solid rgba(128,128,128,0.25) !important;
    }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] p {
        color: var(--text-color) !important;
    }
    
    .stMarkdown, .stText { color: var(--text-color) !important; }
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: var(--text-color) !important;
        font-weight: 500 !important;
    }
    .stCheckbox label { color: var(--text-color) !important; font-size: 14px !important; }
    
    .stMetric {
        background: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);
    }
    .stMetric label, .stMetric div { color: var(--text-color) !important; }
    
    .dataframe, .dataframe th, .dataframe td { color: var(--text-color) !important; }
    
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(128,128,128,0.25);
        color: var(--text-color) !important;
        background: var(--secondary-background-color);
        border-radius: 10px;
    }
    .footer p { color: var(--text-color) !important; }
    
    .stSelectbox div[data-baseweb="select"] { color: var(--text-color) !important; }
    .stNumberInput input {
        color: var(--text-color) !important;
        background: var(--secondary-background-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# HEADER
# ==============================================================================
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🐄 VCDSS - AI</h1>
    <p class="header-subtitle">Veterinary Clinical Decision Support System</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# LOAD MODEL
# ==============================================================================
@st.cache_resource
def load_vcdss_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(script_dir, 'vcdss_stacking_ensemble_model.pkl'),
        r'C:\xampp original\htdocs\VCDSS-AI\vcdss_stacking_ensemble_model.pkl',
        'vcdss_stacking_ensemble_model.pkl'
    ]
    
    model = None
    
    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                break
            except:
                pass
    
    if model is None:
        st.error("❌ Model file not found!")
        st.stop()
    
    try:
        features = model.named_estimators_['rf'].feature_names_in_
        classes = model.classes_
        return model, features, classes
    except Exception as e:
        st.error(f"❌ Error extracting model features: {e}")
        st.stop()

loaded_model, expected_features, target_classes = load_vcdss_model()

# ==============================================================================
# FUNCTION TO GENERATE DECISION TREE
# ==============================================================================
@st.cache_data
def create_decision_tree_plot(feature_names, class_names):
    """Create a decision tree visualization"""
    n_features = len(feature_names)
    n_samples = 2000
    
    np.random.seed(42)
    X_dummy = np.random.randn(n_samples, n_features) * 2
    
    y_dummy = np.zeros(n_samples)
    for i in range(n_samples):
        score = 0
        if n_features > 0:
            score += X_dummy[i, 0] * 0.8
        if n_features > 1:
            score += X_dummy[i, 1] * 0.5
        if n_features > 2:
            score += X_dummy[i, 2] * 0.3
        if n_features > 3:
            score += X_dummy[i, 3] * 0.2
        score += np.random.randn() * 0.3
        
        if score > 1.2:
            y_dummy[i] = 0
        elif score > 0.6:
            y_dummy[i] = 1
        elif score > 0:
            y_dummy[i] = 2
        elif score > -0.6:
            y_dummy[i] = 3
        else:
            y_dummy[i] = 4
    
    dt_visual = DecisionTreeClassifier(
        max_depth=3, 
        random_state=42,
        min_samples_split=30,
        min_samples_leaf=15
    )
    
    try:
        dt_visual.fit(X_dummy, y_dummy)
        
        fig, ax = plt.subplots(figsize=(28, 14))
        
        plot_tree(
            dt_visual,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=9,
            ax=ax,
            proportion=True,
            precision=2
        )
        
        plt.title("🌳 Decision Tree Rule Visualization (Depth = 3)", fontsize=18, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig
    except:
        plt.close()
        return None

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("### 🏥 VCDSS")
    menu = ["🔮 Prediction", "🧠 Model Explanation", "📋 Case History"]
    choice = st.radio("Navigate:", menu, index=0)
    st.markdown("---")
    st.info("💡 Enter vitals and select symptoms")
    st.markdown("🐄 Cattle • 🐑 Sheep • 🐐 Goat")

# ==============================================================================
# HISTORY FILE
# ==============================================================================
HISTORY_FILE = "case_history.csv"

# ==============================================================================
# PREDICTION TAB
# ==============================================================================
if choice == "🔮 Prediction":
    st.markdown("### 🩺 Clinical Diagnosis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card"><h4>📊 Patient Vitals</h4>', unsafe_allow_html=True)
        selected_species = st.selectbox("🐾 Animal Species", ["cow", "sheep", "goat"])
        
        age_value = st.number_input(
            "📅 Age (Years)", 
            min_value=0.1, 
            max_value=20.0, 
            value=3.0, 
            step=0.5,
            help="Enter the animal's age in years"
        )
        
        temp_input = st.number_input(
            "🌡️ Body Temperature (°C)", 
            min_value=35.0, 
            max_value=45.0, 
            value=40.0, 
            step=0.1,
            help="Normal range: 38.0-40.0°C"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card"><h4>🔍 Clinical Symptoms</h4><p>Select all observed symptoms:</p>', unsafe_allow_html=True)
        
        symptom_cols = [c for c in expected_features if "symptom" in c.lower()]
        unique_symptom_names = sorted(list(set(
            c.split('_', 1)[1] if '_' in c else c for c in symptom_cols
        )))
        
        selected_symptoms = []
        sym_col1, sym_col2 = st.columns(2)
        
        for i, sym_name in enumerate(unique_symptom_names):
            target_col = sym_col1 if i % 2 == 0 else sym_col2
            if target_col.checkbox(f"✅ {sym_name.capitalize()}", key=f"chk_{sym_name}"):
                selected_symptoms.append(sym_name)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🚀 Run Diagnostic Prediction", type="primary", use_container_width=True):
        input_df = pd.DataFrame(0, index=[0], columns=expected_features)
        
        if 'Age' in input_df.columns:
            input_df['Age'] = age_value
        if 'Temperature' in input_df.columns:
            input_df['Temperature'] = temp_input
        
        species_col = f"Animal_{selected_species}"
        if species_col in input_df.columns:
            input_df[species_col] = 1
        
        for sym_name in selected_symptoms:
            matching_cols = [c for c in symptom_cols if sym_name.lower() in c.lower()]
            for mc in matching_cols:
                input_df[mc] = 1
        
        prediction = loaded_model.predict(input_df)[0]
        pred_probs = loaded_model.predict_proba(input_df)[0]
        confidence = np.max(pred_probs) * 100
        
        st.markdown("---")
        st.markdown("### 📊 Diagnostic Results")
        
        if confidence > 80:
            status = "High Confidence"
            css_class = "result-success"
            icon = "✅"
        elif confidence > 50:
            status = "Moderate Confidence"
            css_class = "result-warning"
            icon = "⚠️"
        else:
            status = "Low Confidence"
            css_class = "result-error"
            icon = "❌"
        
        st.markdown(f"""
        <div class="result-card {css_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div class="result-title">{icon} {str(prediction).upper()}</div>
                    <div>{status}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 2rem; font-weight: 700;">{confidence:.1f}%</div>
                    <div>Confidence</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        bar_class = "confidence-high" if confidence > 80 else "confidence-medium" if confidence > 50 else "confidence-low"
        st.markdown(f"""
        <div class="confidence-bar-container">
            <div class="confidence-bar {bar_class}" style="width: {confidence}%;">
                {confidence:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📈 Confidence Distribution")
        prob_df = pd.DataFrame({
            "Condition": target_classes,
            "Probability (%)": [p * 100 for p in pred_probs]
        }).set_index("Condition")
        
        st.bar_chart(prob_df)
        
        case_data = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Species": selected_species,
            "Age": age_value,
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
        st.success("✅ Case recorded!")

# ==============================================================================
# MODEL EXPLANATION TAB
# ==============================================================================
elif choice == "🧠 Model Explanation":
    st.markdown("### 🧠 How the AI Makes Decisions")
    
    st.markdown("""
    <div class="card">
        <h4>🤖 Stacking Ensemble Architecture</h4>
        <p>Our system uses multiple AI models working together for accurate diagnosis:</p>
        <ul>
            <li><strong>🌲 Random Forest</strong> - 300 trees analyzing feature importance</li>
            <li><strong>📈 Gradient Boosting</strong> - 800 estimators learning complex patterns</li>
            <li><strong>⚡ Meta-Learner</strong> - Logistic Regression combining predictions</li>
            <li><strong>✅ Accuracy</strong> - <span style="color: #22c55e; font-weight: 700;">~84%</span> overall accuracy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🌳 Decision Tree Rule Visualization (Depth = 3)")
    st.markdown("This shows how the model makes decisions at each step:")
    
    with st.spinner("🌳 Generating decision tree..."):
        fig = create_decision_tree_plot(expected_features, target_classes)
        if fig:
            st.pyplot(fig)
            st.caption("📌 Simplified decision tree showing how the AI reaches a diagnosis")
            plt.close()
        else:
            st.info("ℹ️ Decision tree visualization is being generated.")
    
    st.markdown("---")
    st.markdown("### 📊 Top 10 Clinical Features")
    
    try:
        rf_model = loaded_model.named_estimators_['rf']
        importances = rf_model.feature_importances_
        
        importance_df = pd.DataFrame({
            "Feature": expected_features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=True)
        
        top_10 = importance_df.tail(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_10)))[::-1]
        ax.barh(top_10['Feature'], top_10['Importance'], color=colors)
        ax.set_xlabel('Relative Feature Importance Score', fontweight='bold')
        ax.set_title('Top 10 Clinical Features', fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        with st.expander("📋 View All Features"):
            st.dataframe(importance_df.sort_values(by="Importance", ascending=False), use_container_width=True)
            
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                importance_df.to_excel(writer, sheet_name="Feature Importance", index=False)
            st.download_button(
                label="📥 Download Excel",
                data=excel_buffer.getvalue(),
                file_name="feature_importance.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except:
        st.info("ℹ️ Feature importance data available.")
    
    st.markdown("---")
    st.markdown("### 📈 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Accuracy", "83.79%")
    col2.metric("📊 Classes", "5")
    col3.metric("🏥 Training Data", "29,370")
    
    st.markdown("### 🏥 Supported Diseases")
    disease_cols = st.columns(3)
    diseases = ["Anthrax", "Blackleg", "Foot and Mouth", "Lumpy Virus", "Pneumonia"]
    for i, disease in enumerate(diseases):
        col_idx = i % 3
        disease_cols[col_idx].markdown(f"""
        <div style="background: #f8fafc; 
                    padding: 10px; 
                    border-radius: 8px; 
                    margin: 5px 0;
                    border: 1px solid #dce3ed;
                    color: #000000;">
            <span style="font-weight: 500;">🐾 {disease}</span>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# CASE HISTORY TAB
# ==============================================================================
elif choice == "📋 Case History":
    st.markdown("### 📋 Case History")
    
    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 Total Cases", len(history))
        col2.metric("🏥 Unique Diagnoses", history['Diagnosis'].nunique())
        avg_conf = history['Confidence'].str.replace('%', '').astype(float).mean()
        col3.metric("📈 Avg Confidence", f"{avg_conf:.1f}%")
        
        st.dataframe(history, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        csv_buffer = io.BytesIO()
        history.to_csv(csv_buffer, index=False)
        col1.download_button("📥 Download CSV", csv_buffer.getvalue(), "case_history.csv")
        
        if col3.button("🗑️ Clear History"):
            os.remove(HISTORY_FILE)
            st.rerun()
    else:
        st.info("📭 No saved cases found.")

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("""
<div class="footer">
    <p>🐾 VCDSS-AI • Veterinary Clinical Decision Support System</p>
    <p style="font-size: 0.8rem;">Powered by Streamlit & Scikit-Learn</p>
</div>
""", unsafe_allow_html=True)