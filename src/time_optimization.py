"""
time_optimization.py
=====================
Runs model hyperparameter optimization for 1 seed on a single dataset
and reports the time taken per model and per sampler.

Usage:
    python3 src/time_optimization.py --dataset glass4.csv --seed 0

Output:
    A table showing optimization time (seconds) per model × sampler.
"""

import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from data_setup import prepare_datasets
from experiments import tune_sampler_for_dataset

warnings.filterwarnings("ignore")

cpus = int(os.getenv('SLURM_CPUS_PER_TASK', 1))

param_grids = {
    "Decision Tree": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
    },
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "max_features": ["sqrt", "log2"],
    },
    "Bagging": {
        "n_estimators": [50, 100, 200],
        "max_samples": [0.5, 0.8, 1.0],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1, 0.2],
    },
    "MLP": {
        "hidden_layer_sizes": [(50,), (100,), (100, 50)],
        "activation": ["relu"],
        "alpha": [0.0001],
    },
    "TabNet": {
        "n_d": [8, 16],
        "n_a": [8, 16],
        "n_steps": [3, 5],
        "gamma": [1.3, 1.5],
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf"],
    },
    "LightGBM": {
        "n_estimators": [200, 500],
        "num_leaves": [31,64],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [-1, 10,20],
    },
}


def build_models(seed):
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(random_state=seed, n_jobs=1),
        "Bagging":       BaggingClassifier(estimator=DecisionTreeClassifier(),
                                           random_state=seed, n_jobs=1),
        "XGBoost":       XGBClassifier(eval_metric='logloss', random_state=seed,
                                        n_jobs=1),
        "MLP":           MLPClassifier(max_iter=500, random_state=seed),
        "TabNet":        TabNetClassifier(verbose=0, seed=seed),
        "SVM":           SVC(probability=True, random_state=seed),
        "LightGBM":      LGBMClassifier(random_state=seed, verbosity=-1,
                                         n_jobs=1),
    }


def time_optimize(X, y, models, seed):
    """Run grid search per model and return a dict of {model_name: seconds}."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    times = {}

    for model_name, model in models.items():
        grid_dict = param_grids.get(model_name, {})
        n_combos = int(np.prod([len(v) for v in grid_dict.values()])) if grid_dict else 0

        if n_combos <= 1:
            times[model_name] = 0.0
            continue

        X_fit = X.values if hasattr(X, "values") else X
        y_fit = y.values if hasattr(y, "values") else y

        if model_name == "SVM" and len(X_fit) > 2000:
            X_fit, _, y_fit, _ = train_test_split(
                X_fit, y_fit, train_size=2000, stratify=y_fit, random_state=seed)

        X_fit = np.array(X_fit, dtype=np.float32, copy=True)
        y_fit = np.array(y_fit, dtype=np.int64, copy=True)

        grid = GridSearchCV(model, grid_dict, cv=cv, scoring="f1_macro",
                            n_jobs=cpus, verbose=0, error_score=np.nan)

        print(f"   ⏱  {model_name} ({n_combos} combos × 3 folds)...", end=" ", flush=True)
        t0 = time.time()
        grid.fit(X_fit, y_fit)
        elapsed = time.time() - t0
        times[model_name] = elapsed
        print(f"{elapsed:.1f}s")

    return times


def fmt(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset filename, e.g. glass4.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samplers", nargs="+",
                        default=["original", "smote", "borderline", "adasyn", "corsmote"])
    args = parser.parse_args()

    print(f"\nLoading {args.dataset} (seed={args.seed})...")
    prepared = prepare_datasets(dataset_names=[args.dataset], seed=args.seed)
    data = prepared[args.dataset]

    X_tr = data["X_train"].astype(float)
    y_tr = data["y_train"].astype(int)

    print(f"Train size: {X_tr.shape[0]} rows × {X_tr.shape[1]} features\n")

    all_times = {}

    for sampler_type in args.samplers:
        print(f"\n{'─'*50}")
        print(f" Sampler: {sampler_type.upper()}")
        print(f"{'─'*50}")

        if sampler_type == "original":
            X_fit, y_fit = X_tr.copy(), y_tr.copy()
        else:
            # Tune sampler first
            print(f"  Tuning sampler params...")
            sampler_params = tune_sampler_for_dataset(
                X_tr.to_numpy(), y_tr.to_numpy(),
                smote_type=sampler_type, seed=args.seed
            )
            if sampler_params is None:
                print(f"  Sampler tuning failed, skipping.")
                continue

            from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
            from cor_smote import CorSMOTE
            sp = sampler_params['best_params']

            if sampler_type == "smote":
                s = SMOTE(k_neighbors=sp['sampler__k_neighbors'], random_state=args.seed)
            elif sampler_type == "corsmote":
                s = CorSMOTE(k_neighbors=sp['sampler__k_neighbors'],
                             top_k_features=sp['sampler__top_k_features'],
                             random_state=args.seed)
            elif sampler_type == "adasyn":
                s = ADASYN(n_neighbors=sp['sampler__n_neighbors'], random_state=args.seed)
            elif sampler_type == "borderline":
                s = BorderlineSMOTE(k_neighbors=sp['sampler__k_neighbors'],
                                    kind=sp['sampler__kind'], random_state=args.seed)

            X_fit, y_fit = s.fit_resample(X_tr, y_tr)

        models = build_models(args.seed)
        times = time_optimize(X_fit, y_fit, models, args.seed)
        all_times[sampler_type] = times
        total = sum(times.values())
        print(f"  → Sampler total: {fmt(total)}")

    # ── Summary table ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(" OPTIMIZATION TIME SUMMARY")
    print(f"{'='*60}")

    df = pd.DataFrame(all_times).T  # rows=samplers, cols=models
    df["TOTAL"] = df.sum(axis=1)

    # Format for display
    df_fmt = df.applymap(fmt)
    print(df_fmt.to_string())

    print(f"\nGrand total: {fmt(df['TOTAL'].sum())}")
    print(f"(This is 1 seed. Full run = ×30 seeds, but parallelised via SLURM array.)\n")

    # Save raw seconds to CSV for reference
    out = f"timing_{args.dataset.replace('.csv','')}_{args.seed}.csv"
    df.to_csv(out)
    print(f"Raw times saved to {out}")
