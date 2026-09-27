"""
run_protocol.py - Testing Engine for Milestone 3
Evaluates 5 frozen classifiers across three experimental protocols on Dataset 1 (Crop_recommendation.csv).

Protocols:
1. Single Split (The "Lucky Split" test):
   train_test_split(X, y, test_size=0.2, random_state=0, stratify=None)
2. 5-Fold Cross-Validation:
   StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
3. 10x Repeated Holdout:
   10 stratified 80/20 splits across seeds 0 through 9:
   train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

Preprocessing Rules:
- Label Encoding: For crop labels.
- Scaling: StandardScaler strictly for distance/margin models (SVM, KNN).
  Tree-based and probabilistic models (RF, XGBoost, GaussianNB) use raw features.
  StandardScaler is fit strictly on training data of each split/fold to prevent leakage.

Results:
Saved to results/grid.csv.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


def load_data():
    """Load Dataset 1 (Crop_recommendation.csv) and encode target labels."""
    data_path = 'Crop_recommendation.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find dataset at '{data_path}'.")

    df = pd.read_csv(data_path)
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target_col = 'label'

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"Loaded {data_path}: {df.shape[0]} samples, {df.shape[1]} columns, {len(le.classes_)} classes.", flush=True)
    return X, y_encoded, le


def load_frozen_config():
    """Load frozen hyperparameters and model specifications from configs/frozen_params.json."""
    config_path = os.path.join('configs', 'frozen_params.json')
    if not os.path.exists(config_path):
        config_path = 'frozen_params.json'

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Frozen configuration file '{config_path}' not found.")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print(f"Loaded frozen parameters from '{config_path}'.", flush=True)
    return config


def create_model_instance(model_name, best_params):
    """
    Instantiate a fresh model instance with the frozen best_params.
    Fixed random_state=42 is applied for reproducible stochastic learners.
    """
    if model_name == 'Random Forest':
        return RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    elif model_name == 'XGBoost':
        return XGBClassifier(**best_params, eval_metric='mlogloss', random_state=42, n_jobs=-1)
    elif model_name == 'SVM (RBF)':
        return SVC(**best_params, kernel='rbf', random_state=42)
    elif model_name == 'KNN':
        return KNeighborsClassifier(**best_params, n_jobs=-1)
    elif model_name == 'Gaussian Naive Bayes':
        return GaussianNB(**best_params)
    else:
        raise ValueError(f"Unknown model name: '{model_name}'")


def preprocess_data(X_train, X_test, requires_scaling):
    """
    Apply preprocessing rules:
    - StandardScaler strictly for distance/margin models (SVM, KNN).
    - Raw features for tree-based and probabilistic models (RF, XGBoost, GaussianNB).
    StandardScaler is fit ONLY on training fold/split to avoid data leakage.
    """
    if requires_scaling:
        scaler = StandardScaler()
        X_train_proc = scaler.fit_transform(X_train)
        X_test_proc = scaler.transform(X_test)
    else:
        X_train_proc = np.array(X_train)
        X_test_proc = np.array(X_test)

    return X_train_proc, X_test_proc


def evaluate_split(model_name, best_params, requires_scaling, X_train, y_train, X_test, y_test):
    """Train fresh model on X_train/y_train and compute accuracy and macro-F1 on X_test/y_test."""
    X_tr_proc, X_te_proc = preprocess_data(X_train, X_test, requires_scaling)

    model = create_model_instance(model_name, best_params)
    model.fit(X_tr_proc, y_train)

    y_pred = model.predict(X_te_proc)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    return acc, macro_f1


def main():
    print("=" * 65, flush=True)
    print(" Milestone 3: Testing Engine (run_protocol.py) ", flush=True)
    print("=" * 65, flush=True)

    # 1. Load Data and Frozen Parameters
    X, y, le = load_data()
    frozen_cfg = load_frozen_config()
    models_cfg = frozen_cfg['models']

    os.makedirs('results', exist_ok=True)
    results = []

    # Iterate over each frozen model
    for model_name, info in models_cfg.items():
        best_params = info['best_params']
        requires_scaling = info.get('requires_scaling', False)
        print(f"\n---> Evaluating [{model_name}] (Scaling: {requires_scaling})", flush=True)

        # -------------------------------------------------------------
        # Protocol 1: Single Split (The "Lucky Split" test)
        # train_test_split(X, y, test_size=0.2, random_state=0, stratify=None)
        # -------------------------------------------------------------
        print("  [Protocol 1: Single Split] (random_state=0, stratify=None)...", flush=True)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=None
        )
        acc, f1 = evaluate_split(model_name, best_params, requires_scaling, X_tr, y_tr, X_te, y_te)
        results.append({
            'model': model_name,
            'protocol': 'Single Split',
            'seed_or_fold': 0,
            'accuracy': acc,
            'macro_f1': f1
        })
        print(f"    Single Split -> Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}", flush=True)

        # -------------------------------------------------------------
        # Protocol 2: 5-Fold Cross-Validation
        # StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # -------------------------------------------------------------
        print("  [Protocol 2: 5-Fold Cross-Validation] (random_state=42)...", flush=True)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_accs = []
        cv_f1s = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            acc, f1 = evaluate_split(model_name, best_params, requires_scaling, X_tr, y_tr, X_te, y_te)
            cv_accs.append(acc)
            cv_f1s.append(f1)
            results.append({
                'model': model_name,
                'protocol': '5-Fold CV',
                'seed_or_fold': f'Fold {fold_idx}',
                'accuracy': acc,
                'macro_f1': f1
            })
            print(f"    Fold {fold_idx}: Accuracy={acc:.4f}, Macro-F1={f1:.4f}", flush=True)
        print(f"    CV Mean -> Accuracy: {np.mean(cv_accs):.4f} +/- {np.std(cv_accs):.4f} | "
              f"Macro-F1: {np.mean(cv_f1s):.4f} +/- {np.std(cv_f1s):.4f}", flush=True)

        # -------------------------------------------------------------
        # Protocol 3: 10x Repeated Holdout
        # 10 stratified 80/20 splits across seeds 0 through 9
        # -------------------------------------------------------------
        print("  [Protocol 3: 10x Repeated Holdout] (seeds 0 through 9)...", flush=True)
        rep_accs = []
        rep_f1s = []
        for seed in range(10):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y
            )
            acc, f1 = evaluate_split(model_name, best_params, requires_scaling, X_tr, y_tr, X_te, y_te)
            rep_accs.append(acc)
            rep_f1s.append(f1)
            results.append({
                'model': model_name,
                'protocol': '10x Repeated',
                'seed_or_fold': f'Seed {seed}',
                'accuracy': acc,
                'macro_f1': f1
            })
            print(f"    Seed {seed}: Accuracy={acc:.4f}, Macro-F1={f1:.4f}", flush=True)
        print(f"    Repeated Holdout Mean -> Accuracy: {np.mean(rep_accs):.4f} +/- {np.std(rep_accs):.4f} | "
              f"Macro-F1: {np.mean(rep_f1s):.4f} +/- {np.std(rep_f1s):.4f}", flush=True)

    # 4. Save to results/grid.csv
    out_df = pd.DataFrame(results)
    out_csv = os.path.join('results', 'grid.csv')
    out_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 65, flush=True)
    print(f"SUCCESS: Saved {len(out_df)} records to '{out_csv}'.", flush=True)
    print("=" * 65, flush=True)

    # Display preview summary table
    print("\nProtocol Performance Summary (Mean +/- Std):", flush=True)
    summary = out_df.groupby(['model', 'protocol']).agg(
        mean_accuracy=('accuracy', 'mean'),
        std_accuracy=('accuracy', 'std'),
        mean_macro_f1=('macro_f1', 'mean'),
        std_macro_f1=('macro_f1', 'std')
    ).round(4)
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
