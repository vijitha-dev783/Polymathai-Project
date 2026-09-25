import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

print("=== Starting Milestone 2: Hyperparameter Tuning & Freezing ===")

# 1. Load Dataset 1
data_path = 'Crop_recommendation.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found!")

df = pd.read_csv(data_path)
print(f"Loaded {data_path} with shape: {df.shape}")

feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
target_col = 'label'

X = df[feature_cols]
y = df[target_col]

# 2. Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = list(le.classes_)
print(f"Total crop classes: {len(class_names)}")

# 3. Train/Test Split (80/20, Stratified, Seed=42)
# Crucial requirement: Tuning MUST be performed ONLY on the training split of Dataset 1
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)
print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

# 4. Define GridSearchCV candidate grids for the 5 algorithms
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grids = {
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'scale': False
    },
    'XGBoost': {
        'model': XGBClassifier(eval_metric='mlogloss', random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'scale': False
    },
    'SVM (RBF)': {
        'model': SVC(kernel='rbf', probability=True, random_state=42),
        'params': {
            'C': [0.1, 1.0, 10.0, 50.0],
            'gamma': ['scale', 'auto', 0.01, 0.1]
        },
        'scale': True
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        },
        'scale': True
    },
    'Gaussian Naive Bayes': {
        'model': GaussianNB(),
        'params': {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        },
        'scale': False
    }
}

best_hyperparams = {}
tuning_summary = {}

for name, cfg in grids.items():
    print(f"\n---> Tuning {name}...")
    
    if cfg['scale']:
        # Apply StandardScaler specifically for distance/margin-based models (SVM & KNN)
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed = scaler.transform(X_test)
    else:
        X_train_processed = X_train.values
        X_test_processed = X_test.values

    grid_search = GridSearchCV(
        estimator=cfg['model'],
        param_grid=cfg['params'],
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train_processed, y_train)
    
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    test_score = grid_search.score(X_test_processed, y_test)
    
    print(f"  Best CV Accuracy: {best_cv_score*100:.2f}%")
    print(f"  Held-out Test Accuracy: {test_score*100:.2f}%")
    print(f"  Optimal Parameters: {best_params}")
    
    best_hyperparams[name] = {
        'best_params': best_params,
        'requires_scaling': cfg['scale'],
        'train_cv_accuracy': round(best_cv_score, 4),
        'test_accuracy': round(test_score, 4)
    }

# 5. Build configs/frozen_params.json
os.makedirs('configs', exist_ok=True)
frozen_config = {
    "project_id": "F04-I1",
    "description": "Frozen hyperparameters tuned strictly once on Dataset 1 training split via 5-fold GridSearchCV",
    "tuning_split": "Dataset 1 (Crop_recommendation.csv) 80/20 train split (n=1760), random_state=42",
    "preprocessing_rules": {
        "label_encoding": "sklearn.preprocessing.LabelEncoder (22 classes)",
        "scaling_policy": "StandardScaler applied strictly to distance/margin-based models: ['SVM (RBF)', 'KNN']",
        "dataset2_missing_value_rule": "Fill null values in categorical irrigation_type with sentinel token 'missing'"
    },
    "models": best_hyperparams
}

output_path = os.path.join('configs', 'frozen_params.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(frozen_config, f, indent=4)

print(f"\n=======================================================")
print(f"SUCCESS: Frozen configuration saved to: {output_path}")
print(f"=======================================================")
