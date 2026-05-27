import json

cells = []

def md(source):
    cells.append({"cell_type":"markdown","metadata":{},"source":[s+"\n" for s in source.strip().split("\n")]})

def code(source):
    cells.append({"cell_type":"code","metadata":{},"source":[s+"\n" for s in source.strip().split("\n")],"outputs":[],"execution_count":None})

# ── HEADER ──
md("""# 🔍 Credit Card Fraud Detection & Forecasting
**Statistics & Probability — Final Research Project**
Felicia Sword · 0706012410012 · May 2026

---
Pipeline: MLR → KNN Classification → ARIMA/SARIMA Time Series""")

# ── STAGE 1: SETUP & IMPORTS ──
md("## Stage 1 · Setup & Imports")
code("""import pandas as pd
import numpy as np
from scipy import stats
from math import radians, cos, sin, asin, sqrt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, mean_squared_error,
                             mean_absolute_error, mean_absolute_percentage_error)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
print('All libraries loaded successfully.')""")

# ── STAGE 2: DATA ACQUISITION ──
md("## Stage 2 · Data Acquisition")
code("""# Load raw CSVs (download from Kaggle first)
# kaggle datasets download -d kartik2112/fraud-detection -p ./data/ --unzip
df_train = pd.read_csv('../data/fraudTrain.csv', index_col=0)
df_test = pd.read_csv('../data/fraudTest.csv', index_col=0)
df_raw = pd.concat([df_train, df_test], ignore_index=True)
print(f'Total transactions: {len(df_raw):,}')
print(f'Fraud rate: {df_raw["is_fraud"].mean()*100:.2f}%')
df_raw.head()""")

# ── STAGE 3: STRATIFIED SUBSAMPLING ──
md("## Stage 3 · Stratified Subsampling (~200K rows)")
code("""SAMPLE_SIZE = 200_000
df, _ = train_test_split(df_raw, train_size=SAMPLE_SIZE, stratify=df_raw['is_fraud'], random_state=SEED)
df = df.reset_index(drop=True)
print(f'Subsample size: {len(df):,}')
print(f'Fraud rate preserved: {df["is_fraud"].mean()*100:.2f}%')""")

# ── STAGE 4: FEATURE ENGINEERING ──
md("## Stage 4 · Feature Engineering")
code("""# Datetime features
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day_of_week'] = df['trans_date_trans_time'].dt.day_name()
df['month'] = df['trans_date_trans_time'].dt.month
df['is_weekend'] = df['trans_date_trans_time'].dt.dayofweek.isin([5,6]).astype(int)

# Age
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

# Gender encoding
df['gender_enc'] = LabelEncoder().fit_transform(df['gender'])

# Log-transformed amount
df['log_amt'] = np.log1p(df['amt'])

print(f'Engineered features: hour, day_of_week, month, is_weekend, age, distance, gender_enc, log_amt')
print(f'Shape: {df.shape}')""")

# ── STAGE 5: SAVE CLEAN DATA ──
md("## Stage 5 · Export Clean Dataset")
code("""df.to_csv('../data/data_clean.csv', index=False)
print('Saved data_clean.csv')""")

# ── STAGE 6: EDA ──
md("""## Stage 6 · Exploratory Data Analysis
### 6.1 Descriptive Statistics""")
code("""df[['amt','age','city_pop','distance','hour']].describe().round(2)""")

md("### 6.2 Fraud Distribution")
code("""fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.countplot(x='is_fraud', data=df, ax=axes[0], hue='is_fraud', palette='coolwarm', legend=False)
axes[0].set_title('Fraud vs Legitimate')
sns.boxplot(x='is_fraud', y='amt', data=df, ax=axes[1], hue='is_fraud', palette='coolwarm', legend=False)
axes[1].set_title('Amount by Fraud Status')
sns.histplot(df['log_amt'], bins=50, kde=True, ax=axes[2], color='steelblue')
axes[2].set_title('Log(Amount) Distribution')
plt.tight_layout()
plt.savefig('../outputs/eda_fraud_overview.png', dpi=150)
plt.show()""")

md("### 6.3 Temporal Patterns")
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

# ── STAGE 7: CORRELATION ──
md("### 6.4 Correlation Heatmap (Pearson)")
code("""num_cols = ['amt','log_amt','age','city_pop','distance','hour','is_weekend','gender_enc','is_fraud']
corr = df[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True)
plt.title('Pearson Correlation Matrix')
plt.tight_layout()
plt.savefig('../outputs/correlation_heatmap.png', dpi=150)
plt.show()""")

# ── STAGE 8: MLR ──
md("""---
## Stage 7 · Multiple Linear Regression
### 7.1 Model Specification
`ln(amt) = β₀ + β₁·Category + β₂·Gender + β₃·Age + β₄·Hour + β₅·DayOfWeek + β₆·Distance + β₇·CityPop + ε`""")
code("""# Prepare features
cat_dummies = pd.get_dummies(df['category'], prefix='cat', drop_first=True)
dow_dummies = pd.get_dummies(df['day_of_week'], prefix='dow', drop_first=True)
X_mlr = pd.concat([cat_dummies, dow_dummies, df[['gender_enc','age','hour','distance','city_pop']]], axis=1).astype(float)
y_mlr = df['log_amt']

X_mlr_const = sm.add_constant(X_mlr)
ols_model = sm.OLS(y_mlr, X_mlr_const).fit()
print(ols_model.summary())""")

md("### 7.2 MLR Diagnostics — VIF")
code("""vif_data = pd.DataFrame({'Feature': X_mlr.columns,
                         'VIF': [variance_inflation_factor(X_mlr.values, i) for i in range(X_mlr.shape[1])]})
print(vif_data.sort_values('VIF', ascending=False).head(15).to_string(index=False))""")

md("### 7.3 MLR Diagnostics — Residual Plots & Tests")
code("""residuals = ols_model.resid
fitted = ols_model.fittedvalues

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].scatter(fitted, residuals, alpha=0.1, s=5, color='steelblue')
axes[0].axhline(0, color='red', ls='--')
axes[0].set_title('Residuals vs Fitted')
axes[0].set_xlabel('Fitted'); axes[0].set_ylabel('Residuals')

stats.probplot(residuals, plot=axes[1])
axes[1].set_title('Q-Q Plot')

axes[2].hist(residuals, bins=50, color='steelblue', edgecolor='white')
axes[2].set_title('Residual Distribution')
plt.tight_layout()
plt.savefig('../outputs/mlr_diagnostics.png', dpi=150)
plt.show()

bp_test = het_breuschpagan(residuals, X_mlr_const)
print(f'Breusch-Pagan LM stat: {bp_test[0]:.2f}, p-value: {bp_test[1]:.4e}')
shapiro_sample = residuals.sample(5000, random_state=SEED)
sw_stat, sw_p = stats.shapiro(shapiro_sample)
print(f'Shapiro-Wilk (n=5000 sample): stat={sw_stat:.4f}, p={sw_p:.4e}')""")

md("### 7.4 MLR Performance Metrics")
code("""y_pred_mlr = ols_model.predict(X_mlr_const)
r2 = ols_model.rsquared
adj_r2 = ols_model.rsquared_adj
mae = mean_absolute_error(y_mlr, y_pred_mlr)
mse = mean_squared_error(y_mlr, y_pred_mlr)
rmse = np.sqrt(mse)
print(f'R²: {r2:.4f} | Adj R²: {adj_r2:.4f}')
print(f'MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f}')""")

# ── STAGE 9: KNN ──
md("""---
## Stage 8 · KNN Classification
### 8.1 Data Preparation""")
code("""feature_cols = ['amt','age','city_pop','distance','hour','is_weekend','gender_enc']
X_knn = df[feature_cols].copy()
y_knn = df['is_fraud'].copy()

scaler = StandardScaler()
X_knn_scaled = scaler.fit_transform(X_knn)

X_train, X_test, y_train, y_test = train_test_split(
    X_knn_scaled, y_knn, test_size=0.2, stratify=y_knn, random_state=SEED)
print(f'Train: {len(X_train):,} | Test: {len(X_test):,}')
print(f'Train fraud rate: {y_train.mean()*100:.2f}% | Test fraud rate: {y_test.mean()*100:.2f}%')""")

md("### 8.2 Hyperparameter Tuning (5-Fold CV)")
code("""k_candidates = [1, 3, 5, 7, 9, 11, 15, 21]
cv_results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for k in k_candidates:
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')
    scores = cross_val_score(knn, X_train, y_train, cv=skf, scoring='f1')
    cv_results.append({'k': k, 'mean_f1': scores.mean(), 'std_f1': scores.std()})
    print(f'k={k:2d} → F1={scores.mean():.4f} ± {scores.std():.4f}')

best_k = max(cv_results, key=lambda x: x['mean_f1'])['k']
print(f'\\nBest k: {best_k}')""")

md("### 8.3 Final KNN Evaluation")
code("""knn_final = KNeighborsClassifier(n_neighbors=best_k, algorithm='kd_tree')
knn_final.fit(X_train, y_train)
y_pred_knn = knn_final.predict(X_test)
y_prob_knn = knn_final.predict_proba(X_test)[:, 1]

print('=== Confusion Matrix ===')
print(confusion_matrix(y_test, y_pred_knn))
print('\\n=== Classification Report ===')
print(classification_report(y_test, y_pred_knn, target_names=['Legitimate','Fraud']))
print(f'ROC-AUC: {roc_auc_score(y_test, y_prob_knn):.4f}')""")

md("### 8.4 ROC Curve")
code("""fpr, tpr, _ = roc_curve(y_test, y_prob_knn)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='coral', lw=2, label=f'KNN (AUC={roc_auc_score(y_test, y_prob_knn):.3f})')
plt.plot([0,1],[0,1],'k--', lw=1)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve — KNN Fraud Classification')
plt.legend()
plt.tight_layout()
plt.savefig('../outputs/knn_roc.png', dpi=150)
plt.show()""")

# ── STAGE 10: TIME SERIES ──
md("""---
## Stage 9 · Time Series Analysis (ARIMA/SARIMA)
### 9.1 Temporal Aggregation""")
code("""df_ts = df.set_index('trans_date_trans_time').sort_index()
daily_fraud = df_ts['is_fraud'].resample('D').sum()
daily_fraud = daily_fraud.asfreq('D', fill_value=0)
print(f'Daily fraud series length: {len(daily_fraud)} days')
daily_fraud.plot(figsize=(14, 3), title='Daily Fraud Count', color='coral')
plt.ylabel('Fraud Count')
plt.tight_layout()
plt.savefig('../outputs/ts_daily_fraud.png', dpi=150)
plt.show()""")

md("### 9.2 Time Series Decomposition")
code("""decomp = seasonal_decompose(daily_fraud, model='additive', period=7)
fig = decomp.plot()
fig.set_size_inches(14, 8)
plt.tight_layout()
plt.savefig('../outputs/ts_decomposition.png', dpi=150)
plt.show()""")

md("### 9.3 Stationarity Test (ADF)")
code("""adf_result = adfuller(daily_fraud.dropna())
print(f'ADF Statistic: {adf_result[0]:.4f}')
print(f'p-value: {adf_result[1]:.4e}')
for key, val in adf_result[4].items():
    print(f'  Critical Value ({key}): {val:.4f}')
print(f'\\nStationary: {"Yes" if adf_result[1] < 0.05 else "No — differencing needed"}')""")

md("### 9.4 ACF & PACF")
code("""fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(daily_fraud.dropna(), ax=axes[0], lags=40)
plot_pacf(daily_fraud.dropna(), ax=axes[1], lags=40)
plt.tight_layout()
plt.savefig('../outputs/ts_acf_pacf.png', dpi=150)
plt.show()""")

md("### 9.5 ARIMA Model Fitting & Forecast")
code("""# Chronological train/test split (80/20)
split_idx = int(len(daily_fraud) * 0.8)
train_ts = daily_fraud[:split_idx]
test_ts = daily_fraud[split_idx:]
print(f'Train: {len(train_ts)} days | Test: {len(test_ts)} days')

# Fit ARIMA — adjust (p,d,q) based on ACF/PACF above
model = ARIMA(train_ts, order=(2, 1, 2))
model_fit = model.fit()
print(model_fit.summary())""")

code("""forecast = model_fit.forecast(steps=len(test_ts))
mse_ts = mean_squared_error(test_ts, forecast)
mape_ts = mean_absolute_percentage_error(test_ts[test_ts > 0], forecast[test_ts > 0]) * 100

plt.figure(figsize=(14, 4))
plt.plot(train_ts.index, train_ts, label='Train', color='steelblue')
plt.plot(test_ts.index, test_ts, label='Actual', color='coral')
plt.plot(test_ts.index, forecast, label='Forecast', color='green', ls='--')
plt.title('ARIMA Forecast vs Actual')
plt.ylabel('Daily Fraud Count')
plt.legend()
plt.tight_layout()
plt.savefig('../outputs/ts_forecast.png', dpi=150)
plt.show()
print(f'MSE: {mse_ts:.4f} | MAPE: {mape_ts:.2f}%')
print(f'MAPE < 10%: {"Highly Accurate ✓" if mape_ts < 10 else "Needs improvement"}')""")

# ── SUMMARY ──
md("""---
## Stage 10 · Summary & Next Steps

| Analysis | Key Metric | Result |
|---|---|---|
| MLR | Adj R² | *fill after run* |
| KNN | F1 (Fraud) | *fill after run* |
| ARIMA | MAPE | *fill after run* |

### Next Steps
- [ ] Interpret MLR coefficients for actionable insights
- [ ] Try SARIMA with weekly seasonality (s=7)
- [ ] Write Discussion section connecting all three analyses""")

# Build notebook JSON
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

with open('notebooks/fraud_detection_analysis.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Notebook created: notebooks/fraud_detection_analysis.ipynb')
print(f'Total cells: {len(cells)}')
