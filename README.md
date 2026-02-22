# ML‑Based Health Insurance Premium Prediction

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52.2-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## Live Application

* Deployed App: [https://ml-based-premium-prediction-v1.streamlit.app/](https://ml-based-premium-prediction-v1.streamlit.app/)

---

## Problem Statement
* Health insurers need accurate premium pricing to remain financially viable while staying competitive. Underpricing high-risk customers leads to losses; overpricing low-risk customers drives them away. This project builds a production-deployed ML system that predicts annual health insurance premiums from demographic, lifestyle, and medical risk factors, enabling data-driven, personalised pricing decisions.

---

## Key Results
| Segment | Model | Predictions within ±10% error |
|---------|-------|-------------------------------|
| Young customers (≤25) | Linear Regression + genetic risk features | 98% |
| All other customers (>25) | XGBoost | 99.98% |

> A dual-model strategy (separating young customers where genetic risk is a significant predictor) reduced overall prediction error by over 90% compared to a single combined model.

---

## Approach
* Data: 50,000 synthetic records with demographic, lifestyle, employment, and medical history features. Data cleaning addressed outliers (ages >100, incomes >103K via 4*IQR), invalid entries (negative dependent counts), and 26 missing records.
* Feature Engineering: Multicollinearity was assessed using Variance Inflation Factor (VIF) analysis. Income_Level was dropped (VIF=12.4) as it was redundant with Income_Thousands. Categorical variables were one-hot encoded; ordinal variables (insurance plan tier, income band, medical risk) were encoded to preserve natural ordering.
* Modelling: Seven algorithms were evaluated (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, AdaBoost, XGBoost). The key insight was that young customers have a distinct premium structure driven by genetic risk, a single model failed to capture this, producing 73% error rate for this segment. Splitting by age group and including genetic features for the younger cohort reduced this to 2%.
* Deployment: Production inference runs via a Streamlit app on Streamlit Cloud with CI/CD (auto-redeploy on push to main). Training code is fully separated from inference, only serialised model artifacts and scalers are loaded at runtime.

---

## Technical Stack
* Python 3.10 · Scikit-learn · XGBoost · Streamlit · Pandas · NumPy · Joblib

--- 

## Project Structure
```
ml-based-premium-prediction/
├── main.py                  # Streamlit app entry point
├── prediction_helper.py     # Preprocessing + inference logic
├── requirements.txt         # Pinned runtime dependencies
├── artifacts/               # Serialised models, scalers, feature names
└── notebooks/               # EDA, feature engineering, model training (reference only)
```
---

## Limitations & Version 2 (In Progress)
The current version has known areas for improvement being addressed in v2:
* MinMaxScaler was fit before train-test split in v1, introducing minor data leakage (v2 corrects this)
* Evaluation will expand beyond R^2 to include cross-validated metrics and stricter holdout validation
* Improved code maintainability and test coverage

---

## Data & Privacy
* All data is synthetic. No real patient information is used or stored. User inputs are computed in-memory and not logged or persisted.






