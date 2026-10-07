"""
generate_paper_figures.py - Generates publication-ready figures for the research paper.
Creates high-resolution 300 DPI figures for:
1. Dataset 1 vs Dataset 2 Cross-Benchmark Comparison (Grouped Bar Chart)
2. Split-Variance & Protocol Reliability (Single Split vs 10x Repeated Holdout error bars)
3. Dataset 2 Feature Distribution by Crop (ANOVA Boxplots)
4. Confusion Matrix Heatmaps for Dataset 2 (Visualizing Class Collapse)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Set global publication styling
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})
sns.set_style('whitegrid')

os.makedirs('results/figures', exist_ok=True)

# -------------------------------------------------------------------------
# Figure 1: Dataset 1 vs Dataset 2 Accuracy Comparison (5-Fold CV)
# -------------------------------------------------------------------------
print("Generating Figure 1: Cross-Dataset Comparison Bar Chart...", flush=True)
models = ['Random Forest', 'XGBoost', 'SVM (RBF)', 'KNN', 'Gaussian NB']
d1_acc = [0.9955, 0.9941, 0.9859, 0.9809, 0.9945]
d1_std = [0.0032, 0.0044, 0.0044, 0.0107, 0.0020]

d2_acc = [0.2260, 0.2240, 0.2200, 0.2120, 0.2000]
d2_std = [0.0594, 0.0351, 0.0561, 0.0239, 0.0235]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
rects1 = ax.bar(x - width/2, [a * 100 for a in d1_acc], width, yerr=[s * 100 for s in d1_std],
                label='Dataset 1 (Benchmark)', capsize=5, color='#2b5c8f', edgecolor='black', alpha=0.9)
rects2 = ax.bar(x + width/2, [a * 100 for a in d2_acc], width, yerr=[s * 100 for s in d2_std],
                label='Dataset 2 (Noisy Sensor)', capsize=5, color='#d95f02', edgecolor='black', alpha=0.9)

ax.axhline(22.20, color='red', linestyle='--', linewidth=1.5, label='Dataset 2 Majority Baseline (22.2%)')

ax.set_ylabel('Mean Classification Accuracy (%)', fontweight='bold')
ax.set_title('Cross-Dataset Generalization Breakdown Across 5 Machine Learning Algorithms (5-Fold CV)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight='bold')
ax.set_ylim(0, 115)
ax.legend(frameon=True, loc='upper right')

# Value labels on top of bars
for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/fig1_cross_dataset_comparison.png', dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 2: Split-Variance and Protocol Comparison (Single Split vs 10x Repeated)
# -------------------------------------------------------------------------
print("Generating Figure 2: Protocol Split Variance Analysis...", flush=True)
d2_single = [0.2700, 0.2100, 0.1700, 0.1700, 0.2200]
d2_rep = [0.2330, 0.1920, 0.2080, 0.2100, 0.1790]
d2_rep_std = [0.0356, 0.0297, 0.0290, 0.0333, 0.0251]

fig, ax = plt.subplots(figsize=(9, 5))
rects_single = ax.bar(x - width/2, [s * 100 for s in d2_single], width,
                      label='Single 80/20 Holdout (Unstratified)', color='#7570b3', edgecolor='black')
rects_rep = ax.bar(x + width/2, [r * 100 for r in d2_rep], width, yerr=[s * 100 for s in d2_rep_std],
                   label='10x Repeated Holdout (Mean +/- Std)', capsize=5, color='#1b9e77', edgecolor='black')

ax.axhline(22.20, color='red', linestyle='--', linewidth=1.5, label='Majority-Class Baseline (22.2%)')
ax.set_ylabel('Accuracy on Dataset 2 (%)', fontweight='bold')
ax.set_title('Quantifying the "Lucky Split" Fallacy: Single Split vs. 10x Repeated Holdout on Dataset 2', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight='bold')
ax.set_ylim(0, 35)
ax.legend(frameon=True, loc='upper right')

for rect in rects_single:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for rect in rects_rep:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/fig2_split_variance_bias.png', dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 3: ANOVA Feature Distribution Boxplots (Dataset 2)
# -------------------------------------------------------------------------
print("Generating Figure 3: ANOVA Feature Distributions by Crop...", flush=True)
df2 = pd.read_csv('smart_farming_data.csv')
features_to_plot = ['soil_pH', 'temperature_C', 'rainfall_mm', 'soil_moisture_%']
titles = ['Soil pH (ANOVA p = 0.96)', 'Temperature (°C) (ANOVA p = 0.99)',
          'Rainfall (mm) (ANOVA p = 0.43)', 'Soil Moisture (%) (ANOVA p = 0.15)']

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()

for idx, feat in enumerate(features_to_plot):
    sns.boxplot(x='crop_type', y=feat, data=df2, ax=axes[idx], palette='Set2')
    axes[idx].set_title(titles[idx], fontweight='bold')
    axes[idx].set_xlabel('Crop Class', fontweight='bold')
    axes[idx].set_ylabel(feat, fontweight='bold')

plt.suptitle('Distribution of Environmental Attributes Across Crop Classes in Dataset 2\n(Demonstrating Lack of Agronomic Class Separability)',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('results/figures/fig3_anova_feature_distributions.png', dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 4: Confusion Matrices on Dataset 2 (Random Forest & XGBoost)
# -------------------------------------------------------------------------
print("Generating Figure 4: Dataset 2 Out-Of-Fold Confusion Matrices...", flush=True)
df2_proc = df2.copy()
df2_proc['irrigation_type'] = df2_proc['irrigation_type'].fillna('missing')
for col in ['irrigation_type', 'fertilizer_type']:
    df2_proc[col] = LabelEncoder().fit_transform(df2_proc[col].astype(str))

features = ['soil_moisture_%', 'soil_pH', 'temperature_C', 'rainfall_mm',
            'humidity_%', 'sunlight_hours', 'NDVI_index', 'irrigation_type', 'fertilizer_type']
le_target = LabelEncoder()
y2 = le_target.fit_transform(df2_proc['crop_type'])
X2 = df2_proc[features].values
classes = list(le_target.classes_)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_preds = np.zeros(len(y2), dtype=int)
xgb_preds = np.zeros(len(y2), dtype=int)

for train_idx, test_idx in skf.split(X2, y2):
    X_tr, X_te = X2[train_idx], X2[test_idx]
    y_tr, y_te = y2[train_idx], y2[test_idx]

    rf = RandomForestClassifier(n_estimators=200, min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_preds[test_idx] = rf.predict(X_te)

    xgb = XGBClassifier(n_estimators=200, learning_rate=0.2, max_depth=7, eval_metric='mlogloss', random_state=42, n_jobs=-1)
    xgb.fit(X_tr, y_tr)
    xgb_preds[test_idx] = xgb.predict(X_te)

cm_rf = confusion_matrix(y2, rf_preds)
cm_xgb = confusion_matrix(y2, xgb_preds)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax1, cbar=False)
ax1.set_title('Random Forest (Out-of-Fold 5-Fold CV)\nAccuracy: 22.6%', fontweight='bold')
ax1.set_xlabel('Predicted Label', fontweight='bold')
ax1.set_ylabel('True Label', fontweight='bold')

sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', xticklabels=classes, yticklabels=classes, ax=ax2, cbar=False)
ax2.set_title('XGBoost (Out-of-Fold 5-Fold CV)\nAccuracy: 22.4%', fontweight='bold')
ax2.set_xlabel('Predicted Label', fontweight='bold')
ax2.set_ylabel('True Label', fontweight='bold')

plt.suptitle('Out-of-Fold Confusion Matrices on Dataset 2 Illustrating Stochastic Prediction Dispersion', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('results/figures/fig4_confusion_matrices_dataset2.png', dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 5: Dataset 2 Collapse with Majority Baseline Bar Chart (Reviewer Recommended)
# -------------------------------------------------------------------------
print("Generating Figure 5: Dataset 2 5-Fold CV vs Majority Baseline Bar Chart...", flush=True)
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(models))

bars = ax.bar(x, [m * 100 for m in d2_acc], yerr=[s * 100 for s in d2_std],
              capsize=6, color='#d95f02', edgecolor='black', alpha=0.85, width=0.55,
              label='5-Fold CV Mean (+/- 1 Std Dev)')

line = ax.axhline(22.20, color='red', linestyle='--', linewidth=2,
                  label='Zero-Rule Majority Baseline (Maize: 22.20%)')

ax.set_ylabel('Classification Accuracy (%)', fontweight='bold', fontsize=12)
ax.set_title('Dataset 2 Performance Collapse: 5-Fold CV Mean vs. Majority Baseline', fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight='bold', fontsize=10)
ax.set_ylim(0, 35)
ax.legend(frameon=True, loc='upper right', fontsize=10)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('results/figures/fig5_dataset2_collapse_with_baseline.png', dpi=300)
plt.close()

print("All 5 publication figures generated successfully in 'results/figures/'!", flush=True)

