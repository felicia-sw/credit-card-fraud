import json

cells = []

def md(source):
    cells.append({"cell_type":"markdown","metadata":{},"source":[s+"\n" for s in source.strip().split("\n")]})

def code(source):
    cells.append({"cell_type":"code","metadata":{},"source":[s+"\n" for s in source.strip().split("\n")],"outputs":[],"execution_count":None})

# ── HEADER ──
md("""# 🔍 Credit Card Fraud Detection — KNN vs Logistic Regression
**Statistics & Probability — Final Research Project**
Team Nekat Part 100 · Felicia Sword · 0706012410012 · May 2026

---
**Objective:** Compare K-Nearest Neighbor and Logistic Regression for fraud classification under baseline and SMOTE conditions.
**Dataset:** Sparkov Credit Card Transactions (1.85M transactions, ~0.58% fraud rate)""")

# ── STAGE 1 ──
md("## Stage 1 · Setup & Imports")
code("""import pandas as pd
import numpy as np
from scipy import stats
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)
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
print('All libraries loaded successfully.')""")

# ── STAGE 2 ──
md("## Stage 2 · Data Acquisition")
code("""# Download first: kaggle datasets download -d kartik2112/fraud-detection -p ./data/ --unzip
df_train = pd.read_csv('../data/fraudTrain.csv', index_col=0)
df_test  = pd.read_csv('../data/fraudTest.csv',  index_col=0)
df_raw   = pd.concat([df_train, df_test], ignore_index=True)
print(f'Total transactions: {len(df_raw):,}')
print(f'Fraud rate: {df_raw["is_fraud"].mean()*100:.2f}%')
df_raw.head()""")

# ── STAGE 3 ──
md("## Stage 3 · Stratified Subsampling (~50K rows)")
code("""SAMPLE_SIZE = 50_000
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
code("""df[['amt','age','city_pop','distance','hour']].describe().round(2)""")

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

md("### 6.5 Numerical Feature Distributions by Fraud Status")
code("""fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, col in zip(axes.ravel(), ['age','city_pop','distance','hour']):
    for label, color in [(0,'steelblue'),(1,'coral')]:
        subset = df[df['is_fraud']==label][col]
        ax.hist(subset, bins=40, alpha=0.6, color=color, label=f'{"Fraud" if label else "Legit"}', density=True)
    ax.set_title(f'{col} Distribution by Fraud Status')
    ax.legend()
plt.tight_layout()
plt.savefig('../outputs/eda_distributions.png', dpi=150)
plt.show()""")

md("### 6.6 Correlation Heatmap")
code("""num_cols = ['age','city_pop','distance','hour','is_weekend','gender_enc','is_fraud']
corr = df[num_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0, square=True)
plt.title('Pearson Correlation Matrix')
plt.tight_layout()
plt.savefig('../outputs/correlation_heatmap.png', dpi=150)
plt.show()""")

# ── STAGE 7: DATA PREP ──
md("""---
## Stage 7 · Classification Data Preparation
### 7.1 Feature Matrix Construction""")
code("""# One-hot encode category (X1) — drop_first to avoid multicollinearity
cat_dummies = pd.get_dummies(df['category'], prefix='cat', drop_first=True).astype(int)

# One-hot encode day_of_week (X6)
dow_dummies = pd.get_dummies(df['day_of_week'], prefix='dow', drop_first=True).astype(int)

# Numerical features: gender_enc (X2), age (X3), city_pop (X4), hour (X5), is_weekend (X7), distance (X8), amt (X9)
num_features = df[['gender_enc', 'age', 'city_pop', 'hour', 'is_weekend', 'distance', 'amt']]

# Combine all features
X = pd.concat([cat_dummies, dow_dummies, num_features], axis=1).astype(float)
y = df['is_fraud']

print(f'Feature matrix shape: {X.shape}')
print(f'Features ({X.shape[1]} total):')
print(f'  - Category dummies: {cat_dummies.shape[1]}')
print(f'  - Day-of-week dummies: {dow_dummies.shape[1]}')
print(f'  - Numerical: gender_enc, age, city_pop, hour, is_weekend, distance, amt')
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
    for metric in ['Accuracy', 'Precision (Fraud)', 'Recall (Fraud)', 'F1 (Fraud)', 'ROC-AUC']:
        base_val  = df_results.loc[base_name, metric]
        smote_val = df_results.loc[smote_name, metric]
        delta = smote_val - base_val
        arrow = '↑' if delta > 0 else '↓' if delta < 0 else '→'
        print(f'  {metric:20s}: {base_val:.4f} → {smote_val:.4f} ({arrow} {abs(delta):.4f})')""")

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

**Results (50K stratified subsample, 0.52% fraud, test set ≈ 52 fraud cases, 9 features incl. `amt`):**

| Model | Accuracy | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | ROC-AUC |
|---|---|---|---|---|---|
| KNN Baseline | 0.9946 | **0.4750** | 0.3654 | **0.4130** | 0.6816 |
| KNN + SMOTE | 0.9922 | 0.3088 | 0.4038 | 0.3500 | 0.6996 |
| LR Baseline | 0.9948 | 0.0000 | 0.0000 | 0.0000 | 0.7617 |
| LR + SMOTE | 0.8516 | 0.0252 | **0.7308** | 0.0487 | **0.8834** |

**1. Which algorithm performs best?**

It depends on the operating goal — the two algorithms win on different metrics:

- **KNN Baseline** gives the best **F1 (0.413)** and the best **precision (0.475)** — at the default 0.5 threshold it is the most *balanced* detector, raising relatively few false alarms.
- **LR + SMOTE** gives the best **recall (0.731)** and the best **ROC-AUC (0.884)** — it catches ~73% of fraud and, by the threshold-independent ROC-AUC, has the strongest underlying ability to rank fraud above legitimate transactions (LR 0.76–0.88 vs KNN 0.68–0.70).

Because in fraud detection a missed fraud (false negative) is usually far costlier than a false alarm, **recall and ranking ability are the priority → Logistic Regression + SMOTE is the recommended algorithm**, with the threshold then tuned to recover precision. If instead a balanced precision/recall at the default threshold is wanted, KNN Baseline is the better single choice.

**2. How does SMOTE affect relative performance?**

SMOTE's effect **differs by algorithm** — it is not universally beneficial here:

- **Logistic Regression — SMOTE is essential.** LR Baseline predicts **zero** frauds (recall 0.000) yet still scores 99.5% accuracy — the classic *accuracy paradox*. SMOTE transforms it: recall **0.000 → 0.731** and ROC-AUC **0.762 → 0.883**. The cost is precision (0.025) and accuracy (down to 0.85) from many false positives.
- **KNN — SMOTE slightly hurts the balance.** With `amt` available, KNN Baseline already detects fraud reasonably (precision 0.475, F1 0.413). SMOTE nudges recall up (0.365 → 0.404) but drops precision more (0.475 → 0.309), so **F1 falls (0.413 → 0.350)**; ROC-AUC barely moves (0.682 → 0.700).
- **Takeaway:** SMOTE is critical for the linear model that otherwise ignores the minority class, but counter-productive (for F1) for the distance-based model that already captures local fraud structure.

### Recommendation

**Use Logistic Regression + SMOTE for fraud screening.** Its ROC-AUC of 0.883 shows the strongest discriminative power of all four variants, and it recovers ~73% of fraudulent transactions. Its low precision is a *threshold* artefact, not a ranking weakness — tuning the decision cut-off (or applying a cost-sensitive threshold) on the predicted probabilities is the natural next step to trade some recall for usable precision. Where false-alarm volume must stay low and no threshold tuning is done, **KNN Baseline** (F1 0.413, precision 0.475) is the better off-the-shelf option. Apply SMOTE to LR but **not** to KNN.

### Limitations & Future Work

- **Adding `amt` (X9) was decisive.** Versus the 8-feature version, KNN's ROC-AUC rose 0.51 → 0.68 and its F1 0.04 → 0.41, and LR + SMOTE's ROC-AUC rose 0.75 → 0.88 — confirming transaction amount is the dominant fraud signal in this dataset.
- **Low precision for the SMOTE models** means many false positives at the default threshold; threshold optimisation / cost-sensitive cut-offs were not applied and are the clearest next improvement.
- **`amt` is highly right-skewed**, which can let outliers dominate KNN's Euclidean distances even after z-scoring; a log transform before scaling may further help KNN.
- **Moderate fraud count.** At 50K the test set holds ~52 fraud cases — steadier than 30K but still moderate; 100K–200K would tighten the estimates further (at materially higher KNN cross-validation cost).""")

# ── BUILD ──
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

print(f'Notebook created: notebooks/fraud_detection_analysis.ipynb')
print(f'Total cells: {len(cells)}')
