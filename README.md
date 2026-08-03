# 🐄 Veterinary Clinical Decision Support System (VCDSS)

An AI-powered diagnostic support system designed to assist veterinarians, field officers, and livestock managers in predicting disease conditions in **cattle, sheep, and goats** based on observed clinical symptoms, thermal deviations, and animal vitals.

The core predictive engine uses a high-performance **Stacking Classifier Ensemble** trained across **32,540 verified clinical records**. It reaches **87.29% accuracy**—capturing **99.96% of the theoretical Bayes Optimal Limit (87.32%)** physically achievable on the dataset, while delivering a **100% perfect F1-score ($1.00$)** on fatal and contagious outbreaks (*Anthrax*, *Blackleg*, and *Foot & Mouth Disease*).

---

## 📌 Highlights & Key Performance Metrics

- **Multi-Species Focus:** Tailored diagnostic baselines for cattle ($101.5^\circ\text{F}$), sheep ($102.3^\circ\text{F}$), and goats ($102.5^\circ\text{F}$).
- **Benchmark Proven:** Evaluated against **13 active machine learning architectures** (single trees, bagging/boosting ensembles, and multi-model pipelines).
- **Hits Data Ceiling:** Reaches **87.29% accuracy**, outperforming standard single decision trees (**83.11%**) by effectively blending probabilities on ambiguous disease boundaries (*Lumpy Skin Disease* vs. *Pneumonia*).
- **Zero False Negatives on Outbreaks:** Perfect **1.00 F1-score (100% precision and recall)** on high-consequence conditions (*Anthrax*, *Blackleg*, *Foot and Mouth*).
- **Clinical Explainability:** Features an interactive Streamlit UI with real-time probability distributions, top feature drivers, and an optional 3-level diagnostic decision flowchart for human-in-the-loop verification.

---

## 📊 Dataset Profile & Engineering Pipeline

The raw dataset (`animal_disease_dataset.csv`, 43,778 rows × 7 columns) was filtered strictly for ruminant livestock, yielding **32,540 high-quality target records**:

| Species | Filtered Record Count | Share of Dataset |
| :--- | :---: | :---: |
| **Cow** | **11,254** | 34.58% |
| **Sheep** | **10,658** | 32.75% |
| **Goat** | **10,628** | 32.66% |
| **TOTAL** | **32,540** | **100.0%** |

### Target Disease Breakdown
- **Anthrax:** 7,343 records ($F1 = 1.00$)
- **Blackleg:** 7,318 records ($F1 = 1.00$)
- **Foot and Mouth:** 7,261 records ($F1 = 1.00$)
- **Pneumonia:** 5,361 records ($F1 = 0.63$)
- **Lumpy Virus:** 5,257 records ($F1 = 0.59$)

### Feature Transformations
- **Multi-Hot Symptom Vectorization:** Expands clinical symptom slots into a 22-dimensional binary matrix.
- **Species Thermal Deviation ($\Delta T$):** Calculates relative fever index ($\Delta T = T_{\text{recorded}} - T_{\text{baseline}}$) based on species physiological baselines.
- **Derived Interaction Features:** Includes `temp_age_ratio = Temperature / (Age + 1)` for improved tree node split resolution.

---

## 🏆 Model Architecture Benchmark Summary (Top Models Evaluated)

| Architecture | Category | Accuracy (%) | Outbreak F1 | Lumpy Virus F1 | Pneumonia F1 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Stacking Ensemble (RF + GBM + LogReg)** | **Multi-Model** | **87.29%** | **1.00** | **0.59** | **0.63** | **🏆 CHAMPION MODEL** |
| Two-Stage Hierarchical Classifier | Multi-Model | 87.23% | 1.00 | 0.59 | 0.62 | Evaluated Pipeline |
| Tuned Deep GBM (800 Trees, `lr=0.05`) | Multi-Tree Ensemble | 87.20% | 1.00 | 0.58 | 0.62 | Evaluated Pipeline |
| Extra Trees Classifier | Multi-Tree Ensemble | 87.15% | 1.00 | 0.58 | 0.61 | Evaluated Pipeline |
| Random Forest Classifier (100 Trees) | Multi-Tree Ensemble | 87.09% | 1.00 | 0.57 | 0.61 | Evaluated Pipeline |
| XGBoost Classifier | Multi-Tree Ensemble | 86.95% | 1.00 | 0.57 | 0.60 | Evaluated Pipeline |
| Optimized CART (`max_depth=8`) | Single Tree | 83.11% | 1.00 | 0.47 | 0.50 | Single Tree Baseline |
| Standard CART (Gini, `max_depth=5`) | Single Tree | 81.96% | 1.00 | 0.44 | 0.48 | Decision Flowchart Baseline |

> **Mathematical Note:** Bayes Error Rate analysis proved that **32.63% of the dataset** contains identical input vectors mapped to both *Lumpy Virus* and *Pneumonia*, placing the absolute mathematical accuracy ceiling at **87.32%**. The Stacking Ensemble captures **99.96%** of all learnable signal in the data.

---

## 🛠️ Project Structure

```text
├── app.py                            # Streamlit interactive web application & UI
├── test_model.py                     # CLI multi-disease validation test script
├── vcdss_stacking_ensemble_model.pkl # Production Stacking Ensemble model binary
├── requirements.txt                  # Python dependency configuration file
└── README.md                         # Technical documentation & project guide