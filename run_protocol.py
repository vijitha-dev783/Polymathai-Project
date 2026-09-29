"""
run_protocol.py - Full Grid Testing Engine for Milestone 5
Evaluates 5 frozen classifiers across three experimental protocols on BOTH:
- Dataset 1: Crop_recommendation.csv (22 classes, 7 features, well-clustered benchmark)
- Dataset 2: smart_farming_data.csv (5 classes, 9 features, real-world/synthetic farm data)

Protocols:
1. Single Split (The "Lucky Split" test):
   train_test_split(X, y, test_size=0.2, random_state=0, stratify=None)
2. 5-Fold Cross-Validation:
   StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
3. 10x Repeated Holdout:
   10 stratified 80/20 splits across seeds 0 through 9:
   train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

Preprocessing Rules:
- Label Encoding: Applied to target labels (and categorical features on Dataset 2).
- Dataset 2 Preprocessing:
  * Fill irrigation_type missing values with sentinel token 'missing'.
  * Label encode categorical features: 'irrigation_type' and 'fertilizer_type'.
  * Feature set: ['soil_moisture_%', 'soil_pH', 'temperature_C', 'rainfall_mm',
                  'humidity_%', 'sunlight_hours', 'NDVI_index', 'irrigation_type', 'fertilizer_type']
- Scaling: StandardScaler strictly for distance/margin models (SVM, KNN).
  Tree-based and probabilistic models (RF, XGBoost, GaussianNB) use raw features.
  StandardScaler is fit strictly on training partition of each split/fold to prevent leakage.

Total Runs:
5 models x 2 datasets x (1 + 5 + 10) = 160 runs total.

Results:
Saved to results/full_grid.csv and results/grid.csv.
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


def load_data(dataset_id):
    """
    Load and preprocess dataset:
    - dataset1: Crop_recommendation.csv
    - dataset2: smart_farming_data.csv
    """
    if dataset_id == 'dataset1':
        data_path = 'Crop_recommendation.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Could not find dataset at '{data_path}'.")

        df = pd.read_csv(data_path)
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        target = 'label'

        X = df[features].copy()
        le_target = LabelEncoder()
        y = le_target.fit_transform(df[target])

    elif dataset_id == 'dataset2':
        data_path = 'smart_farming_data.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Could not find dataset at '{data_path}'.")

        df = pd.read_csv(data_path)

        # Preprocess Dataset 2 specific issues
        df['irrigation_type'] = df['irrigation_type'].fillna('missing')
        for col in ['irrigation_type', 'fertilizer_type']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        features = [
            'soil_moisture_%', 'soil_pH', 'temperature_C', 'rainfall_mm',
            'humidity_%', 'sunlight_hours', 'NDVI_index',
            'irrigation_type', 'fertilizer_type'
        ]
        target = 'crop_type'

        X = df[features].copy()
        le_target = LabelEncoder()
        y = le_target.fit_transform(df[target])

    else:
        raise ValueError(f"Unknown dataset_id: '{dataset_id}'. Choose 'dataset1' or 'dataset2'.")

    print(f"Loaded {dataset_id} ({data_path}): {df.shape[0]} samples, {X.shape[1]} features, {len(le_target.classes_)} classes.", flush=True)
    return X, y


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
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    return acc, macro_f1


def main():
    print("=" * 70, flush=True)
    print(" Milestone 5: Full Grid Protocol Evaluation (run_protocol.py) ", flush=True)
    print(" 5 Methods x 2 Datasets x 3 Protocols = 160 Total Runs ", flush=True)
    print("=" * 70, flush=True)

    # 1. Load Frozen Parameters
    frozen_cfg = load_frozen_config()
    models_cfg = frozen_cfg['models']

    os.makedirs('results', exist_ok=True)
    results = []

    # 2. Iterate over both datasets
    for ds_id in ['dataset1', 'dataset2']:
        print("\n" + "=" * 70, flush=True)
        print(f" DATASET: {ds_id.upper()} ", flush=True)
        print("=" * 70, flush=True)

        X, y = load_data(ds_id)

        # 3. Iterate over each frozen model
        for model_name, info in models_cfg.items():
            best_params = info['best_params']
            requires_scaling = info.get('requires_scaling', False)
            print(f"\n---> [{ds_id}] Evaluating [{model_name}] (Scaling: {requires_scaling})", flush=True)

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
                'dataset': ds_id,
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
                    'dataset': ds_id,
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
                    'dataset': ds_id,
                    'model': model_name,
                    'protocol': '10x Repeated',
                    'seed_or_fold': f'Seed {seed}',
                    'accuracy': acc,
                    'macro_f1': f1
                })
                print(f"    Seed {seed}: Accuracy={acc:.4f}, Macro-F1={f1:.4f}", flush=True)
            print(f"    Repeated Holdout Mean -> Accuracy: {np.mean(rep_accs):.4f} +/- {np.std(rep_accs):.4f} | "
                  f"Macro-F1: {np.mean(rep_f1s):.4f} +/- {np.std(rep_f1s):.4f}", flush=True)

    # 4. Save results to results/full_grid.csv and results/grid.csv
    out_df = pd.DataFrame(results)
    out_full_csv = os.path.join('results', 'full_grid.csv')
    out_grid_csv = os.path.join('results', 'grid.csv')

    out_df.to_csv(out_full_csv, index=False)
    out_df.to_csv(out_grid_csv, index=False)

    print("\n" + "=" * 70, flush=True)
    print(f"SUCCESS: Saved {len(out_df)} records to '{out_full_csv}' and '{out_grid_csv}'.", flush=True)
    print("=" * 70, flush=True)

    # 5. Display comprehensive performance summary table
    print("\nFull Grid Performance Summary (Mean +/- Std across multi-seed/fold runs):", flush=True)
    summary = out_df.groupby(['dataset', 'model', 'protocol']).agg(
        mean_accuracy=('accuracy', 'mean'),
        std_accuracy=('accuracy', 'std'),
        mean_macro_f1=('macro_f1', 'mean'),
        std_macro_f1=('macro_f1', 'std')
    ).round(4)
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
