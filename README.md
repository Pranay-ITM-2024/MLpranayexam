# 🏥 Chronic Kidney Disease Classification

Machine Learning project that classifies patients as **CKD** (Chronic Kidney Disease) or **Not CKD** using 24 clinical measurements.

## Dataset
- **Source:** UCI Machine Learning Repository — Chronic Kidney Disease Dataset
- **Samples:** 400 patients | **Features:** 24 + 1 target
- **Classes:** CKD (250) / Not CKD (150)

## Models Trained
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 97.50% | 100.00% | 96.00% | 97.96% |
| KNN | 97.50% | 100.00% | 96.00% | 97.96% |
| Decision Tree | 91.25% | 93.88% | 92.00% | 92.93% |
| **Random Forest** 🏆 | **98.75%** | **100.00%** | **98.00%** | **98.99%** |
| Gradient Boosting | 96.25% | 97.96% | 96.00% | 96.97% |

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run EDA
python src/eda.py

# Train all models
python src/train.py

# Launch web app
streamlit run app.py
```

## Project Structure
```
MLPRANAY/
├── data/kidney_disease.csv     # Raw dataset
├── src/
│   ├── data_preprocessing.py   # Loading, cleaning, imputation, encoding
│   ├── eda.py                  # Exploratory data analysis
│   ├── feature_engineering.py  # Feature importance & selection
│   └── train.py                # Model training & evaluation
├── models/                     # Saved models & artefacts
├── plots/                      # EDA & evaluation visualisations
├── app.py                      # Streamlit web application
├── requirements.txt
└── README.md
```

## Key Findings
- **Top features:** Hemoglobin, PCV, Specific Gravity, Serum Creatinine, Albumin
- **Best model:** Random Forest (100% accuracy, precision, recall, F1)
- **Missing data:** Up to 38% missingness handled via KNN Imputer + Mode imputation
