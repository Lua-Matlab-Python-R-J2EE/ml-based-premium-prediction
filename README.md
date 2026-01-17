# ML‑Based Health Insurance Premium Prediction

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52.2-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

* A machine‑learning (ML) powered Streamlit web application that predicts annual health insurance premiums based on demographic, lifestyle, and medical risk factors. The project demonstrates a complete ML lifecycle, from data cleaning, feature engineering and model training to production‑grade deployment using Streamlit Cloud.

---

## Live Application

* Deployed App: [https://ml-based-premium-prediction-v1.streamlit.app/](https://ml-based-premium-prediction-v1.streamlit.app/)

---

## Dataset Information

* Size: 50,000 records
* Type: Synthetic dataset
* Target Variable: Annual Health Insurance Premium (£)

### Original Features
* Age
* Gender
* Region
* Marital Status
* Number of Dependents
* BMI Category
* Smoking Status
* Employment Status
* Income Level (dropped after VIF analysis)
* Income Thousands
* Medical History
* Insurance Plan
* Annual Premium Amount
* Genetical Risk

---

## Assumptions

- Data Quality
  - The dataset of ~50,000 records is representative of the target population
  - Missing values (26 total records) are missing at random and their removal does not introduce bias
  - Negative dependents (-1, -3) were data entry errors; absolute values represent true counts
  - Ages >100 (58 records) and incomes >103K (10 records using 4*IQR) are genuine outliers to be excluded

- Business Domain
  - Risk scores for medical conditions reflect relative severity: None (0), Thyroid (5), Diabetes/High BP (6), Heart disease (8)
  - Multiple conditions have additive risk (e.g., Diabetes & Heart disease = 14)
  - Insurance plans follow natural hierarchy: Bronze (1) < Silver (2) < Gold (3)
  - Income levels are ordinal: <10K (1) < 10K-25K (2) < 25K-40K (3) < >40K (4)

- Modeling
  - 70/30 train-test split provides adequate data for training and validation
  - VIF threshold of 5 is appropriate for multicollinearity; removing income_level_encode (VIF=12.4) addresses this
  - MinMaxScaler normalization applied after train-test split would prevent data leakage (noted for v2 improvement)
  - R2 score is the primary evaluation metric, supplemented by residual error analysis
  - Performance targets are achievable: R2 >97% and residual errors <10% on 95% of predictions

- Known Limitations
  - Model shows significantly higher errors (>10%) for younger individuals (age ≤25), comprising ~30% of test set errors
  - This suggests potential need for age-based model segmentation or additional data collection for younger demographics

---

## Key Features

* Interactive Streamlit UI for real‑time predictions
* Dual‑model strategy for better accuracy:
  * Young customers (≤ 25 years) with genetic risk features
  * Rest of population (> 25 years) without genetic risk features
* Robust data cleaning and feature engineering
* One‑hot encoding for categorical variables
* Consistent preprocessing using saved scalers
* Production‑safe inference (no training code executed)

---

## Model Performance

### Performance Comparison

| Dataset Segment | Model | Features | Predictions with Error > ±10% |
|----------------|-------|----------|-------------------------------|
| Before Split | Combined | Standard | 30% |
| Before Split | Young-only (≤25) | Standard | 73% |
| After Split | Young-only (≤25) | Linear Regression (with genetic data) | 2% |
| After Split | Rest (>25) | XGBoost (without genetic data) | 0.02% |

### Key Observations

* The inclusion of genetic risk data significantly improved performance for the young-only group (73% --> 2% error rate)
* XGBoost achieved near-perfect accuracy (99.98%) on the non-young dataset
* The dual-model strategy reduced overall prediction errors by 90%+
* Model performance varies substantially based on age and feature set

### Evaluation Metric

* Primary Metric: R2 (Coefficient of Determination) - (not the best)
* Error Analysis: Percentage of predictions with absolute errors > ±10%

---

## Machine Learning Overview

### Models Evaluated

* Linear Regression (selected for young customers)
* Ridge Regression
* Lasso Regression
* Decision Tree
* Random Forest
* AdaBoost
* XGBoost (selected for rest of customers)

### Final Features After Preprocessing (18 total)

After one-hot encoding and feature engineering, the models use these 18 features:

1. Age
2. Number_Of_Dependants
3. Income_Thousands
4. Genetical_Risk (young customers only)
5. Insurance_Plan (ordinal encoded: Bronze=1, Silver=2, Gold=3)
6. Gender_Male
7. Marital_status_Unmarried
8. Region_Northwest
9. Region_Southeast
10. Region_Southwest
11. BMI_Category_Normal
12. BMI_Category_Obesity
13. BMI_Category_Overweight
14. Smoking_Status_Occasional
15. Smoking_Status_Regular
16. Employment_Status_Self-Employed
17. Employment_Status_Unemployed
18. Medical_History_No Disease

> Feature Selection: Finalized using Variance Inflation Factor (VIF) analysis to remove multicollinearity. Income_Level was dropped due to high VIF with Income_Thousands.

---

## Project Structure
```text
ml-based-premium-prediction/
│
├── main.py                     # Streamlit application entry point
├── prediction_helper.py        # Preprocessing + inference logic
├── requirements.txt            # Runtime dependencies
├── runtime.txt                 # Python version pin (3.10)
│
├── artifacts/                  # Trained models & scalers & feature names
│   ├── model_young.joblib
│   ├── model_rest.joblib
│   ├── scaler_young.joblib
│   ├── scaler_rest.joblib
│   └── feature_names.joblib
│
├── notebooks/                  # Training & EDA notebooks
│   ├── ml_premium_prediction_young_with_gr.ipynb
│   ├── ml_premium_prediction_rest_with_gr.ipynb
│   └── ml_premium_prediction.ipynb
│
├── .devcontainer/              # VS Code / Codespaces config
└── README.md
```

---

## Development & Training

### Notebooks

The `notebooks/` directory contains Jupyter notebooks used for:

* Data cleaning and preprocessing
* Feature engineering (VIF analysis, encoding, scaling)
* Exploratory Data Analysis (EDA)
* Model training and evaluation
* Hyperparameter tuning

These notebooks are not required for running the deployed app and are provided for reference and reproducibility.

> Note: No raw data is included in the repository.

---

## Deployment

This application is deployed using Streamlit Cloud with continuous deployment enabled.

### Deployment Details

* Platform: Streamlit Cloud
* Repository: [GitHub](https://github.com/Lua-Matlab-Python-R-J2EE/ml-based-premium-prediction)
* Branch: `main`
* Python Version: 3.10
* Entry Point: `main.py`
* Deployment Type: CI/CD (auto‑redeploy on push)

### How Deployment Works

1. Code is pushed to GitHub
2. Streamlit Cloud pulls the latest commit
3. Dependencies are installed from `requirements.txt`
4. App is launched via:
```bash
streamlit run main.py
```

> Note: Only inference‑time files are used. Training notebooks are not executed during deployment.

---

## Dependency Management

Runtime dependencies are intentionally kept minimal and pinned for reproducibility:
```text
joblib==1.5.3
numpy==2.2.6
pandas==2.3.3
python-dateutil==2.9.0.post0
scikit-learn==1.7.2
scipy==1.15.3
streamlit==1.52.2
tzdata==2025.3
xgboost==3.1.2
```

> Note: The environment used during training is not required for deployment.

---

## Model Artifacts

All trained models and scalers are exported using `joblib` and loaded at runtime:

* Models expect exact feature ordering
* Feature consistency is enforced during inference
* No retraining occurs in production
* Artifacts are version-controlled for reproducibility

---

## Development Environment

This repository includes a `.devcontainer/` configuration for:

* Visual Studio Code
* GitHub Codespaces

> Note: `.devcontainer/` is ignored by Streamlit Cloud and does not affect deployment.

---

## Data & Privacy

* No personal data is stored
* All predictions are computed in‑memory
* No user inputs are logged or persisted
* Synthetic data only—no real patient information

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## Author

**Lua-Matlab-Python-R-J2EE**
* GitHub: [@Lua-Matlab-Python-R-J2EE](https://github.com/Lua-Matlab-Python-R-J2EE)
* Project Link: [https://github.com/Lua-Matlab-Python-R-J2EE/ml-based-premium-prediction](https://github.com/Lua-Matlab-Python-R-J2EE/ml-based-premium-prediction)

---
