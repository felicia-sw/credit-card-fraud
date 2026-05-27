# 🔍 Detecting and Forecasting Credit Card Fraud

**A Combined Classification and Time Series Approach Using Two Years of Transaction Data**

Statistics & Probability — Final Research Project · May 2026

---

## Overview

This research integrates three complementary statistical methods into a single analytical framework for credit card fraud detection:

| Method | Target | Purpose |
|---|---|---|
| **Multiple Linear Regression** | `amt` (transaction amount) | Identify determinants of transaction amounts |
| **K-Nearest Neighbor Classification** | `is_fraud` (binary) | Per-transaction fraud detection |
| **ARIMA / SARIMA** | Aggregated fraud counts | Temporal forecasting of fraud volume |

## Dataset

- **Source:** [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) by Kartik Shenoy (Kaggle)
- **Generator:** [Sparkov Data Generation Tool](https://github.com/namebrandon/Sparkov_Data_Generation)
- **Size:** 1,852,394 transactions (subsampled to ~150K–200K for analysis)
- **Period:** January 2019 – December 2020
- **Fraud rate:** ~0.52% (highly imbalanced)
- **License:** CC0: Public Domain

## Project Structure

```
credit-card-fraud/
├── data/                  # Raw & cleaned datasets (not tracked in git)
├── notebooks/
│   └── fraud_detection_analysis.ipynb   # Full analysis pipeline
├── outputs/               # Generated plots, tables, exports
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

# 5. Open the notebook
jupyter notebook notebooks/fraud_detection_analysis.ipynb
```

## Analytical Pipeline

| Stage | Method | Tools |
|---|---|---|
| 1 | Data Acquisition | `kaggle`, `pandas` |
| 2 | Stratified Subsampling | `sklearn.model_selection` |
| 3 | Feature Engineering | `pandas`, `numpy` |
| 4 | Exploratory Data Analysis | `pandas`, `seaborn` |
| 5 | Correlation Analysis (Pearson) | `scipy.stats`, `seaborn` |
| 6 | Multiple Linear Regression | `statsmodels` |
| 7 | MLR Diagnostics | `statsmodels`, `scipy` |
| 8 | KNN Classification | `sklearn.neighbors` |
| 9 | Classification Evaluation | `sklearn.metrics` |
| 10 | Temporal Aggregation | `pandas` |
| 11 | Time Series Decomposition | `statsmodels` |
| 12 | ARIMA / SARIMA Modeling | `statsmodels` |
| 13 | Forecast Evaluation | `sklearn.metrics`, `statsmodels` |

## Reproducibility

- All stochastic procedures use `random_state=42`
- Analysis runs in a single Jupyter Notebook
- Python 3.10+ required

## Author

**Felicia Sword** · 0706012410012

## License

This project is for academic purposes. The dataset is licensed under CC0: Public Domain.
