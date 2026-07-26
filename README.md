# 🐄 Veterinary Clinical Decision Support System (VCDSS)

An AI-powered diagnostic support system designed to assist veterinarians and livestock managers in predicting disease conditions in cattle, sheep, and goats based on observed symptoms and animal vitals. 

The core predictive engine uses a **Stacking Classifier Ensemble** trained on clinical symptom profiles to deliver diagnostic predictions with probabilistic confidence breakdowns.

---

## 📌 Features

- **Multi-Species Support:** Tailored diagnostics for cattle, sheep, and goats.
- **Ensemble Intelligence:** Leverages a trained Stacking Classifier (`scikit-learn`) for reliable disease prediction across multiple conditions (e.g., *Foot and Mouth*, *Anthrax*, *Blackleg*, *Lumpy Skin Disease*, *Pneumonia*).
- **Interactive Web Interface:** Streamlit UI allowing rapid input of animal vitals (temperature, age) and observed clinical symptoms.
- **Detailed Diagnostic Metrics:** Provides both a primary diagnostic prediction and a complete probability breakdown across all target disease categories.

---

## 🛠️ Project Structure

```text
├── app.py                            # Streamlit interactive UI application
├── test_model.py                     # CLI multi-disease verification test script
├── vcdss_stacking_ensemble_model.pkl # Trained Stacking Classifier model
├── requirements.txt                  # Python dependency configuration
└── README.md                         # Project documentation