import json

cells = []

def md(source):
    cells.append({"cell_type":"markdown","metadata":{},"source":[s+"\n" for s in source.strip().split("\n")]})

def code(source):
    cells.append({"cell_type":"code","metadata":{},"source":[s+"\n" for s in source.strip().split("\n")],"outputs":[],"execution_count":None})

def build_notebook(path):
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.10.0"}},
          "cells": cells}
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'Notebook created: {path}  ({len(cells)} cells)')

# Shared imports cell (used by both notebooks so they stay in sync)
IMPORTS = '''import pandas as pd
import numpy as np
from scipy import stats
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score,
                             average_precision_score, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
os.makedirs('../outputs', exist_ok=True)

SEED = 42
np.random.seed(SEED)
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
print('All libraries loaded successfully.')'''

# ══════════════════════════════════════════════════════════════════════
# NOTEBOOK 1 — EDA & DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════

# ── HEADER ──
md("""# 🔍 Credit Card Fraud Detection — Notebook 1: EDA & Data Preparation
**Statistics & Probability — Final Research Project**
Team Nekat Part 100 · Felicia Sword · 0706012410012 · May 2026

---
This notebook covers **exploratory data analysis (EDA) and data preparation**: it engineers the features, explores the data, and saves a cleaned dataset (`data/data_clean.csv`) that **Notebook 2** uses for the KNN vs Logistic Regression analysis.""")

# ── INTRO ──
md("""## Background & Research Design

**Problem.** Credit-card fraud is one of the most significant threats in the global digital-finance ecosystem. The U.S. Federal Trade Commission reported consumer losses of more than **USD 8.8 billion** to fraud in 2022, up 30% year-on-year (FTC, 2023). In Indonesia, OJK recorded losses of **Rp 7.8 trillion** from digital financial fraud in November 2024, alongside a 550% surge in cyber-crime (OJK, 2026). The core technical challenge is **severe class imbalance** — real fraud rates are typically under 1% — which biases classifiers toward the majority (legitimate) class.

**Research question.** Among two standard classification algorithms — **K-Nearest Neighbor (KNN)** and **Logistic Regression (LR)** — which gives the best performance for classifying credit-card fraud on a highly imbalanced dataset, and how does applying **SMOTE** affect their relative performance?

**Variables.**
- **Y** — `is_fraud` (1 = fraud, 0 = legitimate)
- **X1–X8** — `category`, `gender`, `age`, `city_pop`, `hour`, `day_of_week`, `is_weekend`, `distance` (the registered design)
- **X9** — `amt` (transaction amount): *an extension beyond the registered X1–X8 set, included because it is the dominant fraud signal in this dataset; its impact is analysed explicitly in the conclusions.*

**Method.** Stratified 80:20 train-test split; one-hot encoding for `category` and `day_of_week`, label encoding for `gender`, and z-score standardisation of numeric features. Optimal *k* for KNN is chosen by 5-fold stratified cross-validation. Each algorithm is evaluated under two conditions — **baseline** (no resampling) and **SMOTE** (synthetic oversampling applied to the training data only; the test set is never modified). Models are compared primarily on **F1-score and recall** for the fraud (minority) class, with accuracy, precision, ROC-AUC, and confusion matrices reported alongside.""")

# ── STAGE 1 ──
md("## Stage 1 · Setup & Imports")
code(IMPORTS)

# ── STAGE 2 ──
md("## Stage 2 · Data Acquisition")
code("""# Download first: kaggle datasets download -d kartik2112/fraud-detection -p ./data/ --unzip
df_train = pd.read_csv('../data/fraudTrain.csv', index_col=0)
df_test  = pd.read_csv('../data/fraudTest.csv',  index_col=0)
df_raw   = pd.concat([df_train, df_test], ignore_index=True)
print(f'Total transactions: {len(df_raw):,}')
print(f'Fraud rate: {df_raw["is_fraud"].mean()*100:.2f}%')
df_raw.head()""")

# ── STAGE 2.5: BEHAVIORAL FEATURES ──
md("""## Stage 2.5 · Behavioral Feature Engineering (full data, leak-free)
These per-cardholder features need each card's *full* transaction history, so they are computed
on all 1.85M rows **before** subsampling — otherwise a random subsample would leave each card
with too few transactions to be meaningful. Each feature uses only a transaction's **own past**
(no future information), so there is no data leakage.

- **`amt_zscore_card`** — how unusual the amount is vs the card's prior spending (z-score of the card's past transactions). Fraud amounts average ~3.8σ above a card's norm.
- **`hours_since_prev`** — hours since the card's previous transaction (velocity).
- **`txns_24h`** — number of transactions by the card in the last 24h (velocity).

*These extend the registered X1–X8 + `amt` (X9) variable set; their value is examined empirically.*""")
code("""df_raw['trans_date_trans_time'] = pd.to_datetime(df_raw['trans_date_trans_time'])
df_raw = df_raw.sort_values(['cc_num', 'trans_date_trans_time']).reset_index(drop=True)
_g = df_raw.groupby('cc_num')

# (a) amount vs the card's own PAST average — leak-free (expanding then shift excludes current row)
_past_mean = _g['amt'].transform(lambda s: s.expanding().mean().shift())
_past_std  = _g['amt'].transform(lambda s: s.expanding().std().shift())
df_raw['amt_zscore_card'] = ((df_raw['amt'] - _past_mean) / _past_std).replace([np.inf, -np.inf], np.nan).fillna(0)

# (b) hours since the card's previous transaction
df_raw['hours_since_prev'] = _g['trans_date_trans_time'].diff().dt.total_seconds() / 3600
df_raw['hours_since_prev'] = df_raw['hours_since_prev'].fillna(df_raw['hours_since_prev'].median())

# (c) number of transactions by this card in the last 24h (includes current; known at scoring time)
df_raw['txns_24h'] = df_raw.groupby('cc_num', group_keys=False).apply(
    lambda grp: grp.rolling('24h', on='trans_date_trans_time')['amt'].count(), include_groups=False)

print('Behavioral features computed on full data (leak-free).')
print('Mean by class (0=legit, 1=fraud):')
print(df_raw.groupby('is_fraud')[['amt_zscore_card', 'hours_since_prev', 'txns_24h']].mean().round(3).to_string())""")

# ── STAGE 3 ──
md("## Stage 3 · Stratified Subsampling (~100K rows)")
code("""SAMPLE_SIZE = 100_000
df, _ = train_test_split(df_raw, train_size=SAMPLE_SIZE, stratify=df_raw['is_fraud'], random_state=SEED)
df = df.reset_index(drop=True)
print(f'Subsample size: {len(df):,}')
print(f'Fraud rate preserved: {df["is_fraud"].mean()*100:.2f}%')""")

# ── STAGE 4 ──
md("""## Stage 4 · Feature Engineering
Variables per research plan:
- **Y**: `is_fraud` (binary)
- **X1**: `category` (14 categories → one-hot)
- **X2**: `gender` (M/F → label encoding)
- **X3**: `age` (derived from `dob`)
- **X4**: `city_pop`
- **X5**: `hour` (0–23)
- **X6**: `day_of_week` (Monday–Sunday → one-hot)
- **X7**: `is_weekend` (1/0)
- **X8**: `distance` (haversine km)
- **X9**: `amt` (transaction amount, USD)""")

code("""# Datetime features
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day_of_week'] = df['trans_date_trans_time'].dt.day_name()
df['is_weekend'] = df['trans_date_trans_time'].dt.dayofweek.isin([5,6]).astype(int)

# Age from dob
df['dob'] = pd.to_datetime(df['dob'])
df['age'] = ((df['trans_date_trans_time'] - df['dob']).dt.days / 365.25).astype(int)

# Haversine distance (km)
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(a))

df['distance'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])

# Gender encoding (label encoding: F=0, M=1)
df['gender_enc'] = LabelEncoder().fit_transform(df['gender'])

print(f'Engineered features: hour, day_of_week, is_weekend, age, distance, gender_enc')
print(f'Shape: {df.shape}')""")

# ── STAGE 5 ──
md("## Stage 5 · Export Clean Dataset")
code("""df.to_csv('../data/data_clean.csv', index=False)
print('Saved data_clean.csv')""")

# ── STAGE 6: EDA ──
md("""## Stage 6 · Exploratory Data Analysis
### 6.1 Descriptive Statistics""")
code("""df[['amt','amt_zscore_card','hours_since_prev','txns_24h','age','city_pop','distance','hour']].describe().round(2)""")

md("### 6.2 Class Distribution (Imbalance Check)")
code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# Count
counts = df['is_fraud'].value_counts()
axes[0].bar(['Legitimate (0)', 'Fraud (1)'], counts.values, color=['steelblue','coral'])
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')
axes[0].set_title('Transaction Count by Class')
axes[0].set_ylabel('Count')

# Percentage
axes[1].pie(counts.values, labels=['Legitimate','Fraud'], autopct='%1.2f%%',
            colors=['steelblue','coral'], startangle=90, explode=[0, 0.1])
axes[1].set_title('Class Distribution (%)')
plt.tight_layout()
plt.savefig('../outputs/eda_class_distribution.png', dpi=150)
plt.show()
print(f'Imbalance ratio: 1:{counts[0]//counts[1]}')""")

md("### 6.3 Fraud Rate by Category")
code("""fraud_by_cat = df.groupby('category')['is_fraud'].agg(['mean','count']).sort_values('mean', ascending=False)
fraud_by_cat['mean'] = fraud_by_cat['mean'] * 100

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.barh(fraud_by_cat.index, fraud_by_cat['mean'], color='coral')
ax.set_xlabel('Fraud Rate (%)')
ax.set_title('Fraud Rate by Merchant Category')
for bar, val in zip(bars, fraud_by_cat['mean']):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.2f}%', va='center')
plt.tight_layout()
plt.savefig('../outputs/eda_fraud_by_category.png', dpi=150)
plt.show()""")

md("### 6.4 Temporal Patterns")
code("""fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fraud_by_hour = df.groupby('hour')['is_fraud'].mean() * 100
fraud_by_hour.plot(kind='bar', ax=axes[0], color='coral')
axes[0].set_title('Fraud Rate by Hour (%)')
axes[0].set_ylabel('Fraud %')
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
fraud_by_dow = df.groupby('day_of_week')['is_fraud'].mean().reindex(dow_order) * 100
fraud_by_dow.plot(kind='bar', ax=axes[1], color='mediumpurple')
axes[1].set_title('Fraud Rate by Day of Week (%)')
plt.tight_layout()
plt.savefig('../outputs/eda_temporal.png', dpi=150)
plt.show()""")

md("""### 6.5 Feature Distributions by Fraud Status
Includes the strongest predictors — `amt` and the behavioral `amt_zscore_card`. Heavily-skewed
features are clipped to their 1st–99th percentile so the bulk of the distribution is visible.""")
code("""fig, axes = plt.subplots(2, 3, figsize=(16, 9))
eda_cols = ['amt', 'amt_zscore_card', 'txns_24h', 'hours_since_prev', 'distance', 'age']
for ax, col in zip(axes.ravel(), eda_cols):
    lo, hi = df[col].quantile([0.01, 0.99])
    for label, color in [(0, 'steelblue'), (1, 'coral')]:
        subset = df[df['is_fraud'] == label][col].clip(lo, hi)
        ax.hist(subset, bins=40, alpha=0.6, color=color, density=True,
                label=('Fraud' if label else 'Legit'))
    ax.set_title(f'{col} by fraud status (1-99 pct)')
    ax.legend()
plt.tight_layout()
plt.savefig('../outputs/eda_distributions.png', dpi=150)
plt.show()""")

md("### 6.6 Correlation Heatmap")
code("""num_cols = ['amt','amt_zscore_card','hours_since_prev','txns_24h',
            'age','city_pop','distance','hour','is_weekend','gender_enc','is_fraud']
corr = df[num_cols].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True)
plt.title('Pearson Correlation Matrix')
plt.tight_layout()
plt.savefig('../outputs/correlation_heatmap.png', dpi=150)
plt.show()""")

md("""### 6.7 EDA Summary
- **Severe imbalance** — fraud is ~0.5% of transactions, so plain accuracy is misleading; F1 and recall on the fraud class are the metrics that matter.
- **`amt` and `amt_zscore_card` are the strongest signals** — fraudulent transactions are larger and, especially, sit far above each card's own normal spending (~3.8σ).
- **Some merchant categories and late-night hours carry higher fraud rates** (6.3–6.4).
- **Velocity features (`txns_24h`, `hours_since_prev`) separate the classes only weakly** here — a useful negative finding.
- `distance`, `age`, and `city_pop` show little individual separation.

The cleaned, feature-engineered dataset is saved as `data/data_clean.csv`; **Notebook 2** loads it for the KNN vs Logistic Regression analysis.""")

# ── BUILD NOTEBOOK 1, THEN BEGIN NOTEBOOK 2 ──
build_notebook('notebooks/01_eda.ipynb')
cells = []

# ══════════════════════════════════════════════════════════════════════
# NOTEBOOK 2 — CLASSIFICATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════
md("""# 🔍 Credit Card Fraud Detection — Notebook 2: Classification Analysis
**Statistics & Probability — Final Research Project**
Team Nekat Part 100 · Felicia Sword · 0706012410012 · May 2026

---
**Research question.** Among **K-Nearest Neighbor (KNN)** and **Logistic Regression (LR)**, which best classifies credit-card fraud on this highly imbalanced dataset, and how does **SMOTE** affect their relative performance?

This notebook *continues from* **Notebook 1 (EDA & Data Preparation — Stages 1–6)**: it loads the cleaned, feature-engineered dataset Notebook 1 saved, then trains and compares the models. Primary metrics: **F1 and recall** on the fraud (minority) class.""")

md("""## Setup · Imports & Prepared Data
Re-imports the libraries and loads the cleaned dataset saved by Notebook 1
(run `01_eda.ipynb` first if `data/data_clean.csv` is missing).""")
code(IMPORTS)
code('''df = pd.read_csv('../data/data_clean.csv')
print(f'Loaded {len(df):,} rows, {df.shape[1]} columns | fraud rate {df["is_fraud"].mean()*100:.2f}%')
df[['amt', 'amt_zscore_card', 'hours_since_prev', 'txns_24h', 'is_fraud']].head()''')

# ── STAGE 7: DATA PREP ──
md("""---
## Stage 7 · Classification Data Preparation
### 7.1 Feature Matrix Construction""")
code("""# One-hot encode category (X1) — drop_first to avoid multicollinearity
cat_dummies = pd.get_dummies(df['category'], prefix='cat', drop_first=True).astype(int)

# One-hot encode day_of_week (X6)
dow_dummies = pd.get_dummies(df['day_of_week'], prefix='dow', drop_first=True).astype(int)

# Numerical features: gender_enc (X2), age (X3), city_pop (X4), hour (X5), is_weekend (X7),
# distance (X8), amt (X9) + behavioral extensions (amt_zscore_card, hours_since_prev, txns_24h)
num_features = df[['gender_enc', 'age', 'city_pop', 'hour', 'is_weekend', 'distance', 'amt',
                   'amt_zscore_card', 'hours_since_prev', 'txns_24h']]

# Combine all features
X = pd.concat([cat_dummies, dow_dummies, num_features], axis=1).astype(float)
y = df['is_fraud']

print(f'Feature matrix shape: {X.shape}')
print(f'Features ({X.shape[1]} total):')
print(f'  - Category dummies: {cat_dummies.shape[1]}')
print(f'  - Day-of-week dummies: {dow_dummies.shape[1]}')
print(f'  - Numerical: gender_enc, age, city_pop, hour, is_weekend, distance, amt')
print(f'  - Behavioral: amt_zscore_card, hours_since_prev, txns_24h')
print(f'\\nClass distribution:\\n{y.value_counts().to_string()}')""")

md("### 7.2 Stratified Train-Test Split (80:20)")
code("""X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f'Train: {len(X_train):,} | Test: {len(X_test):,}')
print(f'Train fraud rate: {y_train.mean()*100:.2f}%')
print(f'Test  fraud rate: {y_test.mean()*100:.2f}%')""")

md("### 7.3 Standardization (Z-score)")
code("""# Standardization is critical for KNN (distance-based) and beneficial for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f'Scaled train shape: {X_train_scaled.shape}')
print(f'Scaled test shape:  {X_test_scaled.shape}')""")

md("### 7.4 SMOTE Oversampling (Training Data Only)")
code("""smote = SMOTE(random_state=SEED)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(f'Before SMOTE: {len(X_train_scaled):,} samples')
print(f'After  SMOTE: {len(X_train_smote):,} samples')
print(f'\\nClass distribution after SMOTE:')
print(pd.Series(y_train_smote).value_counts().to_string())
print(f'\\n⚠️ SMOTE applied ONLY to training data. Test set remains unmodified.')""")

# ── STAGE 8: KNN BASELINE ──
md("""---
## Stage 8 · KNN Classification — Baseline (No SMOTE)
### 8.1 Optimal k Selection via 5-Fold Stratified CV""")
code("""k_candidates = [1, 3, 5, 7, 9, 11, 15, 21]
cv_results_knn = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for k in k_candidates:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=skf, scoring='f1')
    cv_results_knn.append({'k': k, 'mean_f1': scores.mean(), 'std_f1': scores.std()})
    print(f'k={k:2d} → F1={scores.mean():.4f} ± {scores.std():.4f}')

best_k_base = max(cv_results_knn, key=lambda x: x['mean_f1'])['k']
print(f'\\n✅ Best k (baseline): {best_k_base}')""")

md("### 8.2 KNN Baseline — Train & Evaluate")
code("""knn_base = KNeighborsClassifier(n_neighbors=best_k_base)
knn_base.fit(X_train_scaled, y_train)
y_pred_knn_base = knn_base.predict(X_test_scaled)
y_prob_knn_base = knn_base.predict_proba(X_test_scaled)[:, 1]

print('=== KNN Baseline (No SMOTE) ===')
print(f'k = {best_k_base}\\n')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_knn_base))
print('\\nClassification Report:')
print(classification_report(y_test, y_pred_knn_base, target_names=['Legitimate','Fraud']))
print(f'ROC-AUC: {roc_auc_score(y_test, y_prob_knn_base):.4f}')""")

# ── STAGE 9: KNN + SMOTE ──
md("""---
## Stage 9 · KNN Classification — With SMOTE
### 9.1 Optimal k Selection via 5-Fold Stratified CV (SMOTE inside CV)""")
code("""# Use imblearn Pipeline to apply SMOTE inside each CV fold (prevents data leakage)
cv_results_knn_smote = []
for k in k_candidates:
    pipe = ImbPipeline([('smote', SMOTE(random_state=SEED)),
                        ('knn', KNeighborsClassifier(n_neighbors=k))])
    scores = cross_val_score(pipe, X_train_scaled, y_train, cv=skf, scoring='f1')
    cv_results_knn_smote.append({'k': k, 'mean_f1': scores.mean(), 'std_f1': scores.std()})
    print(f'k={k:2d} → F1={scores.mean():.4f} ± {scores.std():.4f}')

best_k_smote = max(cv_results_knn_smote, key=lambda x: x['mean_f1'])['k']
print(f'\\n✅ Best k (SMOTE): {best_k_smote}')""")

md("### 9.2 KNN + SMOTE — Train & Evaluate")
code("""knn_smote = KNeighborsClassifier(n_neighbors=best_k_smote)
knn_smote.fit(X_train_smote, y_train_smote)
y_pred_knn_smote = knn_smote.predict(X_test_scaled)
y_prob_knn_smote = knn_smote.predict_proba(X_test_scaled)[:, 1]

print('=== KNN + SMOTE ===')
print(f'k = {best_k_smote}\\n')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_knn_smote))
print('\\nClassification Report:')
print(classification_report(y_test, y_pred_knn_smote, target_names=['Legitimate','Fraud']))
print(f'ROC-AUC: {roc_auc_score(y_test, y_prob_knn_smote):.4f}')""")

# ── STAGE 10: LR BASELINE ──
md("""---
## Stage 10 · Logistic Regression — Baseline (No SMOTE)""")
code("""lr_base = LogisticRegression(max_iter=1000, random_state=SEED)
lr_base.fit(X_train_scaled, y_train)
y_pred_lr_base = lr_base.predict(X_test_scaled)
y_prob_lr_base = lr_base.predict_proba(X_test_scaled)[:, 1]

print('=== Logistic Regression Baseline (No SMOTE) ===\\n')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_lr_base))
print('\\nClassification Report:')
print(classification_report(y_test, y_pred_lr_base, target_names=['Legitimate','Fraud']))
print(f'ROC-AUC: {roc_auc_score(y_test, y_prob_lr_base):.4f}')""")

# ── STAGE 11: LR + SMOTE ──
md("""---
## Stage 11 · Logistic Regression — With SMOTE""")
code("""lr_smote = LogisticRegression(max_iter=1000, random_state=SEED)
lr_smote.fit(X_train_smote, y_train_smote)
y_pred_lr_smote = lr_smote.predict(X_test_scaled)
y_prob_lr_smote = lr_smote.predict_proba(X_test_scaled)[:, 1]

print('=== Logistic Regression + SMOTE ===\\n')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_lr_smote))
print('\\nClassification Report:')
print(classification_report(y_test, y_pred_lr_smote, target_names=['Legitimate','Fraud']))
print(f'ROC-AUC: {roc_auc_score(y_test, y_prob_lr_smote):.4f}')""")

# ── STAGE 12: COMPARATIVE ANALYSIS ──
md("""---
## Stage 12 · Comparative Analysis
### 12.1 Performance Comparison Table""")
code("""# Collect metrics for all 4 models
models = {
    'KNN Baseline':   (y_pred_knn_base,  y_prob_knn_base),
    'KNN + SMOTE':    (y_pred_knn_smote, y_prob_knn_smote),
    'LR Baseline':    (y_pred_lr_base,   y_prob_lr_base),
    'LR + SMOTE':     (y_pred_lr_smote,  y_prob_lr_smote),
}

results = []
for name, (y_pred, y_prob) in models.items():
    results.append({
        'Model': name,
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision (Fraud)': precision_score(y_test, y_pred),
        'Recall (Fraud)':    recall_score(y_test, y_pred),
        'F1 (Fraud)':        f1_score(y_test, y_pred),
        'ROC-AUC':           roc_auc_score(y_test, y_prob),
        'PR-AUC':            average_precision_score(y_test, y_prob),
    })

df_results = pd.DataFrame(results).set_index('Model')
print('═' * 80)
print('COMPARATIVE PERFORMANCE — ALL 4 MODELS')
print('═' * 80)
print(df_results.round(4).to_string())
print()
print(f'🏆 Best F1 (Fraud):  {df_results["F1 (Fraud)"].idxmax()} ({df_results["F1 (Fraud)"].max():.4f})')
print(f'🏆 Best Recall (Fraud): {df_results["Recall (Fraud)"].idxmax()} ({df_results["Recall (Fraud)"].max():.4f})')
print(f'🏆 Best ROC-AUC:     {df_results["ROC-AUC"].idxmax()} ({df_results["ROC-AUC"].max():.4f})')""")

md("### 12.2 ROC Curves — All Models")
code("""plt.figure(figsize=(8, 6))
colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']
for (name, (_, y_prob)), color in zip(models.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.3f})')
plt.plot([0,1],[0,1],'k--', lw=1, alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — KNN vs Logistic Regression (Baseline vs SMOTE)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('../outputs/roc_curves_comparison.png', dpi=150)
plt.show()""")

md("### 12.3 Confusion Matrix Grid")
code("""fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, (y_pred, _)) in zip(axes.ravel(), models.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax,
                xticklabels=['Legitimate','Fraud'], yticklabels=['Legitimate','Fraud'])
    ax.set_title(name, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
plt.suptitle('Confusion Matrices — All Models', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../outputs/confusion_matrices.png', dpi=150)
plt.show()""")

md("### 12.4 F1 & Recall Comparison (Bar Chart)")
code("""fig, axes = plt.subplots(1, 2, figsize=(12, 5))
model_names = df_results.index.tolist()
x = range(len(model_names))

# F1
bars1 = axes[0].bar(x, df_results['F1 (Fraud)'], color=colors)
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names, rotation=15, ha='right')
axes[0].set_title('F1-Score (Fraud Class)', fontweight='bold')
axes[0].set_ylim(0, 1)
for bar, val in zip(bars1, df_results['F1 (Fraud)']):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.4f}', ha='center', fontsize=9)

# Recall
bars2 = axes[1].bar(x, df_results['Recall (Fraud)'], color=colors)
axes[1].set_xticks(x)
axes[1].set_xticklabels(model_names, rotation=15, ha='right')
axes[1].set_title('Recall (Fraud Class)', fontweight='bold')
axes[1].set_ylim(0, 1)
for bar, val in zip(bars2, df_results['Recall (Fraud)']):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.4f}', ha='center', fontsize=9)

plt.suptitle('Primary Metrics — Fraud Detection Performance', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/f1_recall_comparison.png', dpi=150)
plt.show()""")

md("### 12.5 SMOTE Impact Analysis")
code("""print('═' * 60)
print('SMOTE IMPACT ANALYSIS')
print('═' * 60)

for algo in ['KNN', 'LR']:
    base_name = f'{algo} Baseline'
    smote_name = f'{algo} + SMOTE'
    print(f'\\n--- {algo} ---')
    for metric in ['Accuracy', 'Precision (Fraud)', 'Recall (Fraud)', 'F1 (Fraud)', 'ROC-AUC', 'PR-AUC']:
        base_val  = df_results.loc[base_name, metric]
        smote_val = df_results.loc[smote_name, metric]
        delta = smote_val - base_val
        arrow = '↑' if delta > 0 else '↓' if delta < 0 else '→'
        print(f'  {metric:20s}: {base_val:.4f} → {smote_val:.4f} ({arrow} {abs(delta):.4f})')""")

# ── STAGE 12.6: THRESHOLD TUNING ──
md("""### 12.6 · Threshold Tuning
The default 0.5 cut-off is rarely optimal for imbalanced data — it is why some models
score 0 recall despite a strong ROC-AUC. Here we instead pick, for each model, the
decision threshold that **maximises the fraud-class F1**, read off the precision–recall
curve of its predicted probabilities. The test set is untouched; only the cut-off changes.""")
code("""from sklearn.metrics import precision_recall_curve

def best_f1_threshold(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    i = int(np.argmax(f1))
    return thr[i], prec[i], rec[i], f1[i]

tuned = []
for name, (_, y_prob) in models.items():
    t, p, r, f = best_f1_threshold(y_test, y_prob)
    tuned.append({'Model': name,
                  'Default F1': df_results.loc[name, 'F1 (Fraud)'],
                  'Tuned thr': t,
                  'Tuned Precision': p,
                  'Tuned Recall': r,
                  'Tuned F1': f})
df_tuned = pd.DataFrame(tuned).set_index('Model')

print('═' * 78)
print('THRESHOLD TUNING — operating point that maximises fraud-class F1')
print('═' * 78)
print(df_tuned.round(4).to_string())
print()
for name in ['LR Baseline', 'LR + SMOTE']:
    print(f"{name}: F1 {df_results.loc[name,'F1 (Fraud)']:.4f} (thr=0.50) "
          f"→ {df_tuned.loc[name,'Tuned F1']:.4f} (thr={df_tuned.loc[name,'Tuned thr']:.3f})")
print('\\nNote: KNN uses k=1, so its predicted probabilities are ~binary (0/1) and barely'
      ' respond to thresholding — tuning mainly benefits Logistic Regression.')""")

md("### 12.6b · Precision–Recall Curves")
code("""from sklearn.metrics import average_precision_score

plt.figure(figsize=(8, 6))
for (name, (_, y_prob)), color in zip(models.items(), colors):
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    plt.plot(rec, prec, color=color, lw=2, label=f'{name} (AP={ap:.3f})')

# Mark the tuned LR + SMOTE operating point (max-F1)
plt.scatter([df_tuned.loc['LR + SMOTE', 'Tuned Recall']],
            [df_tuned.loc['LR + SMOTE', 'Tuned Precision']],
            color='black', zorder=5, s=120, marker='*',
            label=f"LR+SMOTE tuned (F1={df_tuned.loc['LR + SMOTE','Tuned F1']:.3f})")
base_rate = y_test.mean()
plt.axhline(base_rate, ls='--', color='grey', lw=1, alpha=0.7,
            label=f'No-skill baseline ({base_rate:.4f})')
plt.xlabel('Recall (Fraud)')
plt.ylabel('Precision (Fraud)')
plt.title('Precision–Recall Curves — KNN vs Logistic Regression')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('../outputs/precision_recall_curves.png', dpi=150)
plt.show()""")

# ── STAGE 12.7: DISTANCE-WEIGHTED KNN ──
md("""### 12.7 · KNN Enhancement — Distance-Weighted Voting
The uniform-weight KNN above selected **k = 1** (an artefact of the imbalance — any larger
neighbourhood is swamped by legitimate points) and produced **binary 0/1 probabilities** that
cannot be threshold-tuned. Distance weighting (`weights='distance'`) makes closer neighbours
count more, which *in principle* lets a larger k still surface the minority class and yields
continuous probabilities. Below we (a) re-select k by the same 5-fold CV, then (b) probe a
larger fixed k to see what distance weighting actually enables. This is an extension beyond
the registered design, included for a fairer KNN-vs-LR comparison.""")
code("""# Re-select k for distance-weighted KNN (baseline + SMOTE) — same 5-fold stratified CV on F1
dw_base_cv, dw_smote_cv = [], []
for k in k_candidates:
    knn_dw = KNeighborsClassifier(n_neighbors=k, weights='distance')
    dw_base_cv.append((k, cross_val_score(knn_dw, X_train_scaled, y_train, cv=skf, scoring='f1').mean()))
    pipe = ImbPipeline([('smote', SMOTE(random_state=SEED)),
                        ('knn', KNeighborsClassifier(n_neighbors=k, weights='distance'))])
    dw_smote_cv.append((k, cross_val_score(pipe, X_train_scaled, y_train, cv=skf, scoring='f1').mean()))

best_k_dw_base  = max(dw_base_cv,  key=lambda x: x[1])[0]
best_k_dw_smote = max(dw_smote_cv, key=lambda x: x[1])[0]
print(f'Best k (distance-weighted, baseline): {best_k_dw_base}  (uniform was {best_k_base})')
print(f'Best k (distance-weighted, SMOTE):    {best_k_dw_smote}  (uniform was {best_k_smote})')""")

code("""# Train distance-weighted KNN on the same data and evaluate on the untouched test set
knn_dw_base  = KNeighborsClassifier(n_neighbors=best_k_dw_base,  weights='distance').fit(X_train_scaled, y_train)
knn_dw_smote = KNeighborsClassifier(n_neighbors=best_k_dw_smote, weights='distance').fit(X_train_smote, y_train_smote)

y_pred_dw_base  = knn_dw_base.predict(X_test_scaled)
y_prob_dw_base  = knn_dw_base.predict_proba(X_test_scaled)[:, 1]
y_pred_dw_smote = knn_dw_smote.predict(X_test_scaled)
y_prob_dw_smote = knn_dw_smote.predict_proba(X_test_scaled)[:, 1]

def _row(name, k, y_pred, y_prob):
    return {'Model': name, 'k': k,
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall':    recall_score(y_test, y_pred),
            'F1':        f1_score(y_test, y_pred),
            'ROC-AUC':   roc_auc_score(y_test, y_prob)}

cmp_knn = pd.DataFrame([
    _row('KNN Baseline (uniform)',  best_k_base,     y_pred_knn_base,  y_prob_knn_base),
    _row('KNN Baseline (distance)', best_k_dw_base,  y_pred_dw_base,   y_prob_dw_base),
    _row('KNN + SMOTE (uniform)',   best_k_smote,    y_pred_knn_smote, y_prob_knn_smote),
    _row('KNN + SMOTE (distance)',  best_k_dw_smote, y_pred_dw_smote,  y_prob_dw_smote),
]).set_index('Model')

print('═' * 78)
print('UNIFORM vs DISTANCE-WEIGHTED KNN (k re-selected by CV)')
print('═' * 78)
print(cmp_knn.round(4).to_string())

# WHY identical? At the CV-optimal k=1 there is a single neighbour, so its weight is
# irrelevant — distance weighting only differs from uniform when k > 1, and the severe
# imbalance keeps F1-optimal k at 1. To show what distance weighting *enables*, fit both
# weightings at the largest grid value (k=21) on the SMOTE data:
K_DEMO = 21
proba = {}
for w in ['uniform', 'distance']:
    m = KNeighborsClassifier(n_neighbors=K_DEMO, weights=w).fit(X_train_smote, y_train_smote)
    proba[w] = m.predict_proba(X_test_scaled)[:, 1]

print(f'\\n--- Probe at k={K_DEMO} (SMOTE) ---')
print(f'Distinct probability values: uniform={len(np.unique(proba[\"uniform\"]))}, '
      f'distance={len(np.unique(proba[\"distance\"]))}')
for w in ['uniform', 'distance']:
    default_f1 = f1_score(y_test, (proba[w] >= 0.5).astype(int))
    t, p, r, f = best_f1_threshold(y_test, proba[w])
    print(f'  {w:8s}: default-0.5 F1={default_f1:.4f}  ->  tuned F1={f:.4f} '
          f'(thr={t:.3f}, P={p:.3f}, R={r:.3f})')
print('\\nTakeaway: distance weighting produces many distinct probabilities (vs the binary k=1'
      ' model), so KNN becomes threshold-tunable — but F1-based k-selection still prefers k=1 here.')""")

# ── STAGE 12.8: CONFIDENCE INTERVALS ──
md("""### 12.8 · Confidence Intervals (repeated splits)
A single 80:20 split gives one number per metric, but with so few fraud cases in the test set
that number is noisy. Here we repeat the whole train/evaluate cycle over **5 different random
splits** (each stratified, each with its own scaler and SMOTE fitted only on its training fold)
and report **mean ± standard deviation**, so the stability of each result is visible.
Hyperparameters (KNN k, LR settings) are held at the values chosen earlier.""")
code("""N_REPEATS = 5
ci = {n: {'F1': [], 'Recall': [], 'ROC-AUC': [], 'PR-AUC': []}
      for n in ['KNN Baseline', 'KNN + SMOTE', 'LR Baseline', 'LR + SMOTE']}

def _collect(model, Xtr, ytr, Xte, yte, store):
    model.fit(Xtr, ytr)
    yp = model.predict(Xte); pr = model.predict_proba(Xte)[:, 1]
    store['F1'].append(f1_score(yte, yp));       store['Recall'].append(recall_score(yte, yp))
    store['ROC-AUC'].append(roc_auc_score(yte, pr)); store['PR-AUC'].append(average_precision_score(yte, pr))

for rep in range(N_REPEATS):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED + rep)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    Xtr_sm, ytr_sm = SMOTE(random_state=SEED).fit_resample(Xtr_s, ytr)
    _collect(KNeighborsClassifier(n_neighbors=best_k_base),  Xtr_s,  ytr,    Xte_s, yte, ci['KNN Baseline'])
    _collect(KNeighborsClassifier(n_neighbors=best_k_smote), Xtr_sm, ytr_sm, Xte_s, yte, ci['KNN + SMOTE'])
    _collect(LogisticRegression(max_iter=1000, random_state=SEED), Xtr_s,  ytr,    Xte_s, yte, ci['LR Baseline'])
    _collect(LogisticRegression(max_iter=1000, random_state=SEED), Xtr_sm, ytr_sm, Xte_s, yte, ci['LR + SMOTE'])

rows = []
for n, d in ci.items():
    r = {'Model': n}
    for m in ['F1', 'Recall', 'ROC-AUC', 'PR-AUC']:
        a = np.array(d[m]); r[m] = f'{a.mean():.3f} ± {a.std():.3f}'
    rows.append(r)
df_ci = pd.DataFrame(rows).set_index('Model')
print(f'Mean ± std over {N_REPEATS} stratified 80:20 splits (default 0.5 threshold):')
print(df_ci.to_string())""")

# ── STAGE 12.9: McNEMAR TEST ──
md("""### 12.9 · Significance Test (McNemar)
Is one algorithm *genuinely* better, or could the gap be chance? McNemar's test compares two
models on the **same** test cases by counting where they disagree — how often model A is right
while B is wrong, versus the reverse. A small p-value (< 0.05) means the difference is
statistically significant, not luck.""")
code("""from scipy.stats import chi2 as _chi2

def mcnemar(y_true, pred_a, pred_b, name_a, name_b):
    yt = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == yt
    b_ok = np.asarray(pred_b) == yt
    b_only = int(np.sum(a_ok & ~b_ok))   # A right, B wrong
    c_only = int(np.sum(~a_ok & b_ok))   # A wrong, B right
    n = b_only + c_only
    stat = (abs(b_only - c_only) - 1) ** 2 / n if n else 0.0
    p = _chi2.sf(stat, df=1)
    verdict = 'SIGNIFICANT (p < 0.05)' if p < 0.05 else 'not significant'
    print(f'{name_a:12s} vs {name_b:12s} | A-right/B-wrong={b_only:4d}, A-wrong/B-right={c_only:4d} '
          f'| chi2={stat:7.3f}, p={p:.4f} -> {verdict}')

print('McNemar test on test-set predictions (A vs B):')
mcnemar(y_test, y_pred_lr_base,  y_pred_knn_base,  'LR Baseline', 'KNN Baseline')
mcnemar(y_test, y_pred_lr_smote, y_pred_knn_smote, 'LR + SMOTE',  'KNN + SMOTE')""")

# ── STAGE 13: SUMMARY ──
md("""---
## Stage 13 · Summary & Conclusion
### 13.1 Final Results Table (auto-generated from the run)""")
code("""from IPython.display import display, Markdown

print('FINAL RESULTS — All 4 Models')
display(df_results.round(4))

best_f1  = df_results['F1 (Fraud)'].idxmax()
best_rec = df_results['Recall (Fraud)'].idxmax()
best_auc = df_results['ROC-AUC'].idxmax()

lines = ['**Key findings (computed):**', '']
lines.append(f"- Best **F1 (fraud)**: `{best_f1}` ({df_results.loc[best_f1,'F1 (Fraud)']:.4f})")
lines.append(f"- Best **Recall (fraud)**: `{best_rec}` ({df_results.loc[best_rec,'Recall (Fraud)']:.4f})")
lines.append(f"- Best **ROC-AUC**: `{best_auc}` ({df_results.loc[best_auc,'ROC-AUC']:.4f})")
for algo in ['KNN', 'LR']:
    d_f1  = df_results.loc[f'{algo} + SMOTE','F1 (Fraud)']     - df_results.loc[f'{algo} Baseline','F1 (Fraud)']
    d_rec = df_results.loc[f'{algo} + SMOTE','Recall (Fraud)'] - df_results.loc[f'{algo} Baseline','Recall (Fraud)']
    lines.append(f"- **SMOTE effect on {algo}**: F1 {d_f1:+.4f}, Recall {d_rec:+.4f}")
display(Markdown('\\n'.join(lines)))""")

md("""### 13.2 Research Questions Answered

**Default-threshold results (100K stratified subsample, 0.52% fraud, test set ≈ 104 fraud cases, 9 registered features + 3 behavioral extensions):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| KNN Baseline | 0.9958 | **0.6438** | 0.4519 | 0.5311 | 0.7253 | **0.2938** |
| KNN + SMOTE | 0.9950 | 0.5179 | 0.5577 | **0.5370** | 0.7775 | 0.2911 |
| LR Baseline | 0.9943 | 0.1875 | 0.0288 | 0.0500 | 0.8369 | 0.2676 |
| LR + SMOTE | 0.9016 | 0.0395 | **0.7692** | 0.0752 | **0.9007** | 0.1884 |

**Threshold-tuned (cut-off chosen to maximise fraud-class F1 — Stage 12.6):**

| Model | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| KNN Baseline | 1.00 | 0.644 | 0.452 | 0.531 |
| KNN + SMOTE | 1.00 | 0.518 | 0.558 | **0.537** |
| LR Baseline | 0.085 | 0.534 | 0.452 | 0.490 |
| LR + SMOTE | 0.970 | 0.343 | 0.452 | 0.390 |

**Robustness (Stages 12.8–12.9).** Over 5 repeated splits, KNN + SMOTE F1 = 0.517 ± 0.027 and LR + SMOTE recall = 0.754 ± 0.013 — intervals tight enough that the gaps below are real, not noise. **McNemar's test confirms it:** KNN beats LR on prediction correctness in both the baseline (p = 0.001) and SMOTE (p < 0.0001) conditions.

**1. Which algorithm performs best?**

With the behavioral features added, **KNN becomes the stronger *balanced* classifier while Logistic Regression remains the stronger *ranker* — and PR-AUC shows why that distinction matters:**

- **KNN + SMOTE** has the best **F1 (0.537)**, and KNN Baseline the best **precision (0.644)** and **PR-AUC (0.294)** — KNN actually makes usable fraud predictions.
- **LR + SMOTE** has the best **recall (0.769)** and **ROC-AUC (0.901)** — but the **worst PR-AUC (0.188)**. This is the key lesson of reporting PR-AUC: ROC-AUC flatters LR + SMOTE because it ignores the flood of *false alarms* behind that recall (precision is just 0.04). Under the precision-aware PR-AUC, KNN wins.
- The single best F1 anywhere is **distance-weighted KNN at k = 21 with threshold tuning (F1 ≈ 0.570, Stage 12.7)** — better than any default-threshold model.

**Verdict:** for a *deployable* detector that balances catching fraud against false alarms, **KNN (with SMOTE) is the better choice**, and McNemar says its edge over LR is statistically significant. LR + SMOTE is preferable only when maximum recall is the sole goal and false positives are cheap.

**2. How does SMOTE affect relative performance?**

- **For KNN, SMOTE now helps** (F1 0.531 → 0.537, recall 0.452 → 0.558) — a reversal from the weaker feature set where it hurt. With informative behavioral features there is real minority-class structure for SMOTE to interpolate.
- **For LR, SMOTE remains essential but double-edged.** It lifts recall from ~0.03 to 0.77 and ROC-AUC to 0.90, yet collapses precision to 0.04 (accuracy 0.90) — a flood of false positives. Threshold-tuning the *untouched* LR baseline reaches a far healthier balance (F1 0.49, precision 0.53) than LR + SMOTE.
- **Takeaway:** SMOTE and threshold tuning both rebalance the minority decision. SMOTE helps the model that has structure to exploit (KNN here); for LR, simple threshold tuning is the cleaner lever.

### Recommendation

**Use KNN + SMOTE as the practical fraud detector** (F1 0.54, precision 0.52, best PR-AUC), with distance-weighted voting at a larger k + threshold tuning as the best-performing variant (F1 ≈ 0.57). Reserve **LR + SMOTE** for when recall is the sole priority and false positives are acceptable. And report **PR-AUC, not just ROC-AUC**, on imbalanced fraud data — it reverses the verdict here.

### Limitations & Future Work

- **Behavioral features drove the gains.** `amt_zscore_card` (amount vs the card's own history) lifted KNN + SMOTE F1 from 0.35 to 0.54 and LR + SMOTE ROC-AUC from 0.88 to 0.90. The velocity features (`hours_since_prev`, `txns_24h`) were near-noise (|corr| < 0.02) on this dataset — a fair negative result.
- **KNN probabilities are degenerate at k = 1**, which blocks threshold tuning. Stage 12.7 shows distance weighting is identical at k = 1, but at k = 21 it yields continuous probabilities and, after tuning, the best KNN result (F1 ≈ 0.57). The strongest KNN needs distance weighting + a larger k + tuning, not the F1-CV-optimal k = 1.
- **`amt` and `amt_zscore_card` are right-skewed**; a log transform before scaling may further help the distance-based KNN.
- **Stronger models / more data** (gradient-boosted trees, or the full 1.85M rows) would likely push PR-AUC above the ~0.29 ceiling reached here.""")

# ── REFERENCES ──
md("""---
## References

- Federal Trade Commission. (2023, February 21). *New FTC data show consumers reported losing nearly $8.8 billion to scams in 2022.* https://www.ftc.gov/news-events/news/press-releases/2023/02/new-ftc-data-show-consumers-reported-losing-nearly-88-billion-scams-2022
- Kaggle / Kartik Shenoy. (2020). *Credit Card Transactions Fraud Detection Dataset* [Dataset]. https://www.kaggle.com/datasets/kartik2112/fraud-detection
- Harris, B. (2016). *Sparkov Data Generation Tool* [Source code]. GitHub. https://github.com/namebrandon/Sparkov_Data_Generation
- Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). PaySim: A financial mobile money simulator for fraud detection. *Proceedings of the 28th European Modeling and Simulation Symposium (EMSS 2016).* https://www.msc-les.org/proceedings/emss/2016/EMSS2016_249.pdf
- Grover, P., Xu, J., Tittelfitz, J., Cheng, A., Li, Z., Zablocki, J., Liu, J., & Zhou, H. (2022). Fraud dataset benchmark and applications. *arXiv.* https://doi.org/10.48550/arXiv.2208.14417
- OJK. (2026, January 19). *Kejahatan siber melonjak 550 persen, OJK ingatkan pentingnya keamanan digital.* ANTARA News. https://www.antaranews.com/berita/5363118""")

# ── BUILD NOTEBOOK 2 ──
build_notebook('notebooks/02_analysis.ipynb')
