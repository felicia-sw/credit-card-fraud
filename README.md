# 🔍 Credit Card Fraud Detection — KNN vs Logistic Regression

**Comparative Classification Analysis with SMOTE Oversampling**

Statistics & Probability — Final Research Project · Team Nekat Part 100 · May 2026

---

## Overview

This research compares two classification algorithms — **K-Nearest Neighbor (KNN)** and **Logistic Regression** — for detecting credit card fraud on a highly imbalanced dataset (~0.58% fraud rate). Each algorithm is evaluated under two conditions:

| Condition | Description |
|---|---|
| **Baseline** | No resampling — trained on original imbalanced data |
| **SMOTE** | Synthetic Minority Oversampling applied to training data only |

**Primary metrics:** F1-score and Recall on the fraud (minority) class.

## Dataset

- **Source:** [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) by Kartik Shenoy (Kaggle)
- **Generator:** [Sparkov Data Generation Tool](https://github.com/namebrandon/Sparkov_Data_Generation)
- **Size:** 1,852,394 transactions (subsampled to ~100K for analysis)
- **Period:** January 2019 – December 2020
- **Fraud rate:** ~0.58% (highly imbalanced)

## Variables

| Role | Variable | Description |
|---|---|---|
| **Y** | `is_fraud` | Binary: 1 = fraud, 0 = legitimate |
| X1 | `category` | Merchant category (14 categories, one-hot encoded) |
| X2 | `gender` | Cardholder gender (M/F, label encoded) |
| X3 | `age` | Cardholder age (derived from `dob`) |
| X4 | `city_pop` | Cardholder city population |
| X5 | `hour` | Transaction hour (0–23) |
| X6 | `day_of_week` | Transaction day (Mon–Sun, one-hot encoded) |
| X7 | `is_weekend` | Weekend indicator (1/0) |
| X8 | `distance` | Haversine distance cardholder–merchant (km) |
| X9 | `amt` | Transaction amount (USD) |

**Behavioral extension features** (computed leak-free on the full data, beyond the registered set): `amt_zscore_card` (amount vs the card's own spending history), `hours_since_prev` and `txns_24h` (transaction velocity).

## Project Structure

```
credit-card-fraud/
├── data/                  # Raw & cleaned datasets (not tracked in git)
├── notebooks/
│   ├── 01_eda.ipynb       # EDA & data preparation → saves data_clean.csv
│   └── 02_analysis.ipynb  # Classification analysis (KNN vs LR, SMOTE, rigor)
├── outputs/               # Generated plots & tables
├── create_notebook.py     # Generator script (builds both notebooks)
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/credit-card-fraud.git
cd credit-card-fraud

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset from Kaggle
kaggle datasets download -d kartik2112/fraud-detection -p ./data/ --unzip

# 5. Open the notebooks — run 01_eda first (it creates data_clean.csv), then 02_analysis
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_analysis.ipynb
```

## Methodology

1. **Preprocessing:** Feature engineering (hour, age, distance via haversine), one-hot encoding (category, day_of_week), label encoding (gender), z-score standardization
2. **Class imbalance:** Baseline (no resampling) vs SMOTE (training data only)
3. **Evaluation:** Stratified 80:20 split, 5-fold stratified CV for KNN k-selection, confusion matrix, accuracy, precision, recall, F1-score, ROC-AUC
4. **Comparison:** 4 model variants (KNN Baseline, KNN+SMOTE, LR Baseline, LR+SMOTE)

## Author

**Felicia Sword** · 0706012410012

## License

This project is for academic purposes. The dataset is licensed under CC0: Public Domain.
