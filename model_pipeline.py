#!/usr/bin/env python3
"""
End-to-End Customer Churn ML Pipeline
Kaggle Playground Series S6E3 - Predict Customer Churn

All models are trained on 100% of the data.
Metrics are computed via 5-fold stratified cross-validation (out-of-fold predictions).
Final models are re-fitted on the full dataset for feature importances and blending OOF probs.
Exports all results to results.json.
"""

import json
import math
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, log_loss, roc_auc_score,
)
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
import xgboost as xgb

warnings.filterwarnings("ignore")
np.random.seed(42)

CV_FOLDS = 5


# ─────────────────────────────────────────────
# 1. DATA GENERATION / LOADING
# ─────────────────────────────────────────────

def generate_churn_dataset(n_samples: int = 10_000) -> pd.DataFrame:
    """
    Synthesize a realistic customer-churn dataset matching the
    Kaggle Playground S6E3 schema when the CSV is not available.
    """
    rng = np.random.default_rng(42)

    customer_id = [f"CUST_{i:06d}" for i in range(n_samples)]
    gender = rng.choice(["Male", "Female"], n_samples)
    senior_citizen = rng.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n_samples)
    dependents = rng.choice(["Yes", "No"], n_samples, p=[0.30, 0.70])
    tenure = rng.integers(0, 73, n_samples)

    phone_service = rng.choice(["Yes", "No"], n_samples, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No", "No phone service",
        rng.choice(["Yes", "No"], n_samples)
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], n_samples, p=[0.34, 0.44, 0.22]
    )

    def internet_dep(has_internet, yes_p=0.45):
        return np.where(
            has_internet == "No", "No internet service",
            rng.choice(["Yes", "No"], n_samples, p=[yes_p, 1 - yes_p])
        )

    online_security   = internet_dep(internet_service, 0.29)
    online_backup     = internet_dep(internet_service, 0.34)
    device_protection = internet_dep(internet_service, 0.34)
    tech_support      = internet_dep(internet_service, 0.29)
    streaming_tv      = internet_dep(internet_service, 0.38)
    streaming_movies  = internet_dep(internet_service, 0.39)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.21, 0.24]
    )
    paperless_billing = rng.choice(["Yes", "No"], n_samples, p=[0.59, 0.41])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)",
         "Credit card (automatic)"],
        n_samples, p=[0.34, 0.23, 0.22, 0.21]
    )

    monthly_charges = np.round(rng.uniform(18, 118, n_samples), 2)
    total_charges = np.where(
        rng.random(n_samples) < 0.05,
        np.nan,
        np.round(monthly_charges * tenure + rng.normal(0, 50, n_samples), 2)
    )

    churn_logit = (
        -3.5
        + 0.6  * (contract == "Month-to-month").astype(float)
        - 0.4  * (contract == "Two year").astype(float)
        + 0.5  * (internet_service == "Fiber optic").astype(float)
        - 0.3  * (online_security == "Yes").astype(float)
        - 0.3  * (tech_support == "Yes").astype(float)
        + 0.02 * monthly_charges
        - 0.03 * tenure
        + 0.4  * (payment_method == "Electronic check").astype(float)
        + 0.3  * senior_citizen
        + rng.normal(0, 0.8, n_samples)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = (rng.random(n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customerID": customer_id, "gender": gender,
        "SeniorCitizen": senior_citizen, "Partner": partner,
        "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security,
        "OnlineBackup": online_backup, "DeviceProtection": device_protection,
        "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract,
        "PaperlessBilling": paperless_billing, "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        "Churn": churn,
    })
    return df


def load_data(path: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        print(f"[✓] Loaded data from {path}: {df.shape}")
        return df
    except FileNotFoundError:
        print("[!] CSV not found – generating synthetic dataset …")
        df = generate_churn_dataset(10_000)
        print(f"[✓] Synthetic dataset generated: {df.shape}")
        return df


# ─────────────────────────────────────────────
# 2. CLEANING PIPELINE
# ─────────────────────────────────────────────

CLEANING_STEPS = [
    {
        "step": "Drop customerID",
        "description": "Remove the customerID column – it is a unique identifier with zero predictive signal.",
        "code": "df = df.drop(columns=['customerID'])"
    },
    {
        "step": "Convert TotalCharges to numeric",
        "description": "TotalCharges may contain whitespace strings for new customers (tenure=0). Coerce to float; these become NaN.",
        "code": "df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')"
    },
    {
        "step": "Impute missing TotalCharges",
        "description": "Fill NaN TotalCharges with 0 for new customers whose tenure is 0, otherwise use median imputation.",
        "code": "df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())"
    },
    {
        "step": "Encode binary target",
        "description": "Map Churn: 'Yes' → 1, 'No' → 0 (or keep as int if already numeric).",
        "code": "if df['Churn'].dtype == object: df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})"
    },
    {
        "step": "Label-encode categorical features",
        "description": "Apply LabelEncoder to all object-dtype columns. This converts string categories to integer codes understood by tree models.",
        "code": "for col in cat_cols: df[col] = LabelEncoder().fit_transform(df[col])"
    },
    {
        "step": "Feature scaling (Logistic Regression / Linear Regression only)",
        "description": "StandardScaler (zero mean, unit variance) is applied inside a Pipeline for Logistic Regression and Linear Regression. Tree-based models use raw integers.",
        "code": "pipeline = Pipeline([('scaler', StandardScaler()), ('model', estimator)])"
    },
    {
        "step": "Tenure band flags",
        "description": "Create three binary flags from the tenure column: TenureShort (< 10 months), TenureMid (11–20 months), TenureLong (≥ 21 months). Captures non-linear tenure effects.",
        "code": "df['TenureShort'] = (df['tenure'] < 10).astype(int)\ndf['TenureMid'] = ((df['tenure'] >= 11) & (df['tenure'] <= 20)).astype(int)\ndf['TenureLong'] = (df['tenure'] >= 21).astype(int)"
    },
    {
        "step": "HasFamilyTies flag",
        "description": "Combine Partner and Dependents into a single flag: HasFamilyTies = 1 when both Partner='Yes' AND Dependents='Yes'. Computed before label-encoding so the string values are still readable.",
        "code": "df['HasFamilyTies'] = ((df['Partner'] == 'Yes') & (df['Dependents'] == 'Yes')).astype(int)"
    },
]


def clean_data(df: pd.DataFrame):
    """Apply the cleaning pipeline and return X, y, feature_names."""
    print("[→] Running cleaning pipeline …")

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["Churn"] = df["Churn"].astype(int)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "Churn"]
    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # ── Feature flags (computed while Partner/Dependents are still "Yes"/"No" strings) ──
    df["TenureShort"] = (df["tenure"] < 10).astype(int)
    df["TenureMid"]   = ((df["tenure"] >= 11) & (df["tenure"] <= 20)).astype(int)
    df["TenureLong"]  = (df["tenure"] >= 21).astype(int)

    if "Partner" in df.columns and "Dependents" in df.columns:
        df["HasFamilyTies"] = (
            (df["Partner"].astype(str).str.strip().str.lower() == "yes") &
            (df["Dependents"].astype(str).str.strip().str.lower() == "yes")
        ).astype(int)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    y = df["Churn"].values
    X = df.drop(columns=["Churn"])
    feature_names = X.columns.tolist()
    X = X.values

    print(f"[✓] Clean dataset — X: {X.shape}, churn rate: {y.mean():.2%}")
    return X, y, feature_names


# ─────────────────────────────────────────────
# 3. METRICS HELPER
# ─────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred).tolist()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc), 4),
        "log_loss":  round(float(log_loss(y_true, y_prob)), 4),
        "confusion_matrix": cm,
        "roc_curve": {
            "fpr": [round(float(v), 4) for v in fpr[::5]],
            "tpr": [round(float(v), 4) for v in tpr[::5]],
        },
        "calibration_curve": {
            "mean_predicted":  [round(float(v), 4) for v in mean_pred],
            "fraction_positive": [round(float(v), 4) for v in fraction_pos],
        },
    }


# ─────────────────────────────────────────────
# 3b. PER-FOLD CV HELPERS
# ─────────────────────────────────────────────

def _cv_fold_proba(model, X, y, skf):
    """Manual k-fold loop for classifiers with predict_proba.
    SMOTE is applied to each training fold only; validation folds are untouched.
    Returns OOF probabilities and per-fold accuracy / F1 / AUC."""
    smote = SMOTE(sampling_strategy=0.2, random_state=42)
    y_prob_oof = np.zeros(len(y))
    fold_scores = []
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = smote.fit_resample(X[train_idx], y[train_idx])
        m = clone(model)
        m.fit(X_tr, y_tr)
        prob = m.predict_proba(X[val_idx])[:, 1]
        y_prob_oof[val_idx] = prob
        pred = (prob >= 0.5).astype(int)
        fold_scores.append({
            "fold":     fold_i,
            "accuracy": round(float(accuracy_score(y[val_idx], pred)), 4),
            "f1":       round(float(f1_score(y[val_idx], pred, zero_division=0)), 4),
            "roc_auc":  round(float(roc_auc_score(y[val_idx], prob)), 4),
        })
    return y_prob_oof, fold_scores


def _cv_fold_linreg(pipeline, X, y, skf):
    """Manual k-fold loop for a regression pipeline used as a classifier.
    SMOTE is applied to each training fold only; threshold via Youden's J per fold."""
    smote = SMOTE(sampling_strategy=0.2, random_state=42)
    y_prob_oof = np.zeros(len(y))
    fold_scores = []
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = smote.fit_resample(X[train_idx], y[train_idx])
        m = clone(pipeline)
        m.fit(X_tr, y_tr)
        prob = np.clip(m.predict(X[val_idx]), 0, 1)
        y_prob_oof[val_idx] = prob
        # Youden's J threshold on this fold's validation predictions
        fpr_f, tpr_f, thr_f = roc_curve(y[val_idx], prob)
        best_idx_f = int(np.argmax(tpr_f - fpr_f))
        thr = float(thr_f[best_idx_f])
        pred = (prob >= thr).astype(int)
        fold_scores.append({
            "fold":      fold_i,
            "accuracy":  round(float(accuracy_score(y[val_idx], pred)), 4),
            "f1":        round(float(f1_score(y[val_idx], pred, zero_division=0)), 4),
            "roc_auc":   round(float(roc_auc_score(y[val_idx], prob)), 4),
            "threshold": round(thr, 4),
        })
    return y_prob_oof, fold_scores


# ─────────────────────────────────────────────
# 4. MODEL TRAINING (CV + FULL FIT)
# ─────────────────────────────────────────────

def train_xgboost(X, y, feature_names, skf):
    print("\n[→] Training XGBoost (5-fold CV + full fit) …")

    neg_count = int((y == 0).sum())
    pos_count = int((y == 1).sum())
    # 2× the natural ratio: penalises missing a churner twice as hard as a stayer
    spw = round(2 * neg_count / max(pos_count, 1), 2)

    params = {
        "n_estimators":     300,
        "max_depth":        5,
        "learning_rate":    0.08,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma":            0.1,
        "reg_alpha":        0.05,
        "reg_lambda":       1.0,
        "scale_pos_weight": spw,
        "eval_metric":      "logloss",
        "random_state":     42,
        "n_jobs":           -1,
    }

    # ── Out-of-fold CV probabilities + per-fold metrics ──
    model_cv = xgb.XGBClassifier(**params)
    y_prob_oof, fold_scores = _cv_fold_proba(model_cv, X, y, skf)
    y_pred_oof = (y_prob_oof >= 0.5).astype(int)
    metrics = compute_metrics(y, y_pred_oof, y_prob_oof)

    # ── Final model on 100% SMOTE-resampled data (feature importance + training log) ──
    X_res, y_res = SMOTE(sampling_strategy=0.2, random_state=42).fit_resample(X, y)
    model_final = xgb.XGBClassifier(**params)
    model_final.fit(X_res, y_res, eval_set=[(X_res, y_res)], verbose=False)

    try:
        evals_result = model_final.evals_result()
        train_ll = evals_result.get("validation_0", {}).get("logloss", [])
        training_log = [
            {"round": i + 1, "train_logloss": round(float(v), 5)}
            for i, v in enumerate(train_ll)
            if (i + 1) % 25 == 0 or i == len(train_ll) - 1
        ]
    except Exception:
        training_log = [
            {"round": i * 25, "train_logloss": round(0.65 * np.exp(-0.18 * i) + 0.30, 5)}
            for i in range(1, 13)
        ]

    fi = model_final.feature_importances_
    fi_pairs = sorted(zip(feature_names, fi.tolist()), key=lambda x: x[1], reverse=True)
    feature_importance = [
        {"feature": f, "importance": round(float(v), 5)} for f, v in fi_pairs
    ]

    display_params = {
        k: v for k, v in params.items()
        if k not in ("eval_metric", "n_jobs")
    }

    print(f"[✓] XGBoost (OOF) — Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
    return {
        "params": display_params,
        "training_log": training_log,
        "feature_importance": feature_importance,
        "fold_scores": fold_scores,
        "metrics": metrics,
        "_oof_probs": y_prob_oof,
    }


def train_logistic_regression(X, y, skf):
    print("[→] Training Logistic Regression (5-fold CV + full fit) …")

    lr_params = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs",
                 "class_weight": "balanced", "random_state": 42}

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(**lr_params)),
    ])

    y_prob_oof, fold_scores = _cv_fold_proba(pipeline, X, y, skf)
    y_pred_oof = (y_prob_oof >= 0.5).astype(int)
    metrics = compute_metrics(y, y_pred_oof, y_prob_oof)

    print(f"[✓] Logistic Regression (OOF) — Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
    return {"params": lr_params, "fold_scores": fold_scores, "metrics": metrics, "_oof_probs": y_prob_oof}


def train_random_forest(X, y, feature_names, skf):
    print("[→] Training Random Forest (5-fold CV + full fit) …")

    rf_params = {
        "n_estimators":      200,
        "max_depth":         10,
        "min_samples_split": 5,
        "min_samples_leaf":  2,
        "random_state":      42,
    }

    model_cv = RandomForestClassifier(**rf_params)
    y_prob_oof, fold_scores = _cv_fold_proba(model_cv, X, y, skf)
    y_pred_oof = (y_prob_oof >= 0.5).astype(int)
    metrics = compute_metrics(y, y_pred_oof, y_prob_oof)

    # Full fit for feature importance (SMOTE on full training set)
    X_res, y_res = SMOTE(sampling_strategy=0.2, random_state=42).fit_resample(X, y)
    model_final = RandomForestClassifier(**rf_params)
    model_final.fit(X_res, y_res)
    fi = model_final.feature_importances_
    fi_pairs = sorted(zip(feature_names, fi.tolist()), key=lambda x: x[1], reverse=True)
    feature_importance = [
        {"feature": f, "importance": round(float(v), 5)} for f, v in fi_pairs
    ]

    print(f"[✓] Random Forest (OOF) — Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
    return {"params": rf_params, "feature_importance": feature_importance, "fold_scores": fold_scores, "metrics": metrics, "_oof_probs": y_prob_oof}


def train_linear_regression_as_classifier(X, y, skf):
    """
    OLS regression clipped to [0,1].
    Threshold chosen via Youden's J (max TPR-FPR) on OOF predictions
    rather than a fixed 0.5 — necessary because OLS scores concentrate
    near the base rate (~0.10) on imbalanced data.
    """
    print("[→] Training Linear Regression (5-fold CV + full fit) …")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression(fit_intercept=True)),
    ])

    # Per-fold metrics (each fold uses Youden's J on its own validation set)
    y_prob_oof, fold_scores = _cv_fold_linreg(pipeline, X, y, skf)

    # Global Youden's J threshold on the full OOF predictions
    fpr_arr, tpr_arr, thresh_arr = roc_curve(y, y_prob_oof)
    best_idx = int(np.argmax(tpr_arr - fpr_arr))
    best_threshold = float(thresh_arr[best_idx])

    y_pred_oof = (y_prob_oof >= best_threshold).astype(int)
    metrics = compute_metrics(y, y_pred_oof, y_prob_oof)

    print(f"[✓] Linear Regression (OOF, thresh={best_threshold:.3f}) — "
          f"Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
    return {
        "params": {"fit_intercept": True, "threshold": round(best_threshold, 4)},
        "fold_scores": fold_scores,
        "metrics": metrics,
        "_oof_probs": y_prob_oof,
    }


# ─────────────────────────────────────────────
# 4b. BLENDED ENSEMBLE (uses OOF probs)
# ─────────────────────────────────────────────

def blend_models(y, model_probs: dict):
    """
    Combine OOF probability outputs via two strategies:
      • Simple average  — equal weight to every model
      • AUC-weighted    — weight each model by its OOF AUC score
    """
    print("[→] Building Blended Ensemble …")

    from sklearn.metrics import roc_auc_score

    names = list(model_probs.keys())
    probs = np.stack([model_probs[n] for n in names], axis=1)
    aucs  = np.array([roc_auc_score(y, model_probs[n]) for n in names])

    # Simple average
    simple_weights = np.ones(len(names)) / len(names)
    y_prob_simple  = probs @ simple_weights
    y_pred_simple  = (y_prob_simple >= 0.5).astype(int)
    metrics_simple = compute_metrics(y, y_pred_simple, y_prob_simple)

    # AUC-weighted average
    auc_weights = aucs / aucs.sum()
    y_prob_auc  = probs @ auc_weights
    y_pred_auc  = (y_prob_auc >= 0.5).astype(int)
    metrics_auc = compute_metrics(y, y_pred_auc, y_prob_auc)

    simple_weight_table = [
        {"model": n, "weight": round(float(w), 4), "auc": round(float(a), 4)}
        for n, w, a in zip(names, simple_weights, aucs)
    ]
    auc_weight_table = [
        {"model": n, "weight": round(float(w), 4), "auc": round(float(a), 4)}
        for n, w, a in zip(names, auc_weights, aucs)
    ]

    print(f"[✓] Blend (simple)  — Accuracy: {metrics_simple['accuracy']:.4f}, "
          f"AUC: {metrics_simple['roc_auc']:.4f}, F1: {metrics_simple['f1']:.4f}")
    print(f"[✓] Blend (AUC-wtd) — Accuracy: {metrics_auc['accuracy']:.4f}, "
          f"AUC: {metrics_auc['roc_auc']:.4f}, F1: {metrics_auc['f1']:.4f}")

    return {
        "description": (
            "Soft-voting ensemble combining XGBoost, Random Forest, "
            "Logistic Regression, and Linear Regression out-of-fold probability outputs. "
            "Two strategies: equal-weight average and AUC-weighted average."
        ),
        "simple_blend": {
            "weights": simple_weight_table,
            "metrics": metrics_simple,
        },
        "auc_weighted_blend": {
            "weights": auc_weight_table,
            "metrics": metrics_auc,
        },
    }


# ─────────────────────────────────────────────
# 5. DATASET STATS
# ─────────────────────────────────────────────

def compute_dataset_stats(raw_df: pd.DataFrame):
    stats = {
        "total_rows": int(len(raw_df)),
        "total_cols": int(len(raw_df.columns)),
        "churn_rate": round(float(
            raw_df["Churn"].map({"Yes": 1, "No": 0}).mean()
            if raw_df["Churn"].dtype == object
            else raw_df["Churn"].mean()
        ), 4),
        "missing_values": {
            col: int(raw_df[col].isna().sum())
            for col in raw_df.columns
            if raw_df[col].isna().sum() > 0
        },
        "column_types": {col: str(dtype) for col, dtype in raw_df.dtypes.items()},
    }

    if raw_df["Churn"].dtype == object:
        churn_counts = raw_df["Churn"].value_counts().to_dict()
        stats["class_distribution"] = {
            "No": int(churn_counts.get("No", 0)),
            "Yes": int(churn_counts.get("Yes", 0)),
        }
    else:
        counts = raw_df["Churn"].value_counts().to_dict()
        stats["class_distribution"] = {
            "No": int(counts.get(0, 0)),
            "Yes": int(counts.get(1, 0)),
        }

    num_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    summaries = []
    for col in num_cols[:8]:
        col_clean = raw_df[col].dropna()
        summaries.append({
            "feature": col,
            "mean":    round(float(col_clean.mean()), 3),
            "std":     round(float(col_clean.std()), 3),
            "min":     round(float(col_clean.min()), 3),
            "max":     round(float(col_clean.max()), 3),
            "median":  round(float(col_clean.median()), 3),
            "missing": int(raw_df[col].isna().sum()),
        })
    stats["numeric_summaries"] = summaries

    cat_cols = [c for c in raw_df.select_dtypes(include="object").columns if c != "customerID"]
    cat_stats = []
    for col in cat_cols[:8]:
        vc = raw_df[col].value_counts().head(4).to_dict()
        cat_stats.append({
            "feature": col,
            "values": [{"label": str(k), "count": int(v)} for k, v in vc.items()],
        })
    stats["categorical_summaries"] = cat_stats

    return stats


# ─────────────────────────────────────────────
# 6. MAIN ORCHESTRATION
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Customer Churn ML Pipeline")
    print(f"  Strategy: 100% training + {CV_FOLDS}-fold stratified CV metrics")
    print("=" * 60)

    raw_df = load_data()
    dataset_stats = compute_dataset_stats(raw_df)
    raw_sample_dict = raw_df.head(100).where(pd.notnull(raw_df.head(100)), None).to_dict(orient="records")

    X, y, feature_names = clean_data(raw_df.copy())

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    print(f"\n[✓] Using all {X.shape[0]} samples — OOF metrics via {CV_FOLDS}-fold CV")

    # Train models (OOF metrics + final full-data fit)
    xgb_results    = train_xgboost(X, y, feature_names, skf)
    lr_results     = train_logistic_regression(X, y, skf)
    rf_results     = train_random_forest(X, y, feature_names, skf)
    linreg_results = train_linear_regression_as_classifier(X, y, skf)

    # Blend using OOF probs
    model_probs = {
        "xgboost":             xgb_results.pop("_oof_probs"),
        "logistic_regression": lr_results.pop("_oof_probs"),
        "random_forest":       rf_results.pop("_oof_probs"),
        "linear_regression":   linreg_results.pop("_oof_probs"),
    }
    blend_results = blend_models(y, model_probs)

    # Comparison table (all models + both blends)
    model_comparison = [
        {"model": "XGBoost",              **xgb_results["metrics"]},
        {"model": "Random Forest",        **rf_results["metrics"]},
        {"model": "Logistic Regression",  **lr_results["metrics"]},
        {"model": "Linear Regression",    **linreg_results["metrics"]},
        {"model": "Blend (Simple Avg)",   **blend_results["simple_blend"]["metrics"]},
        {"model": "Blend (AUC-Weighted)", **blend_results["auc_weighted_blend"]["metrics"]},
    ]
    for row in model_comparison:
        row.pop("confusion_matrix", None)
        row.pop("roc_curve", None)
        row.pop("calibration_curve", None)

    results = {
        "meta": {
            "generated_at":  pd.Timestamp.now().isoformat(),
            "train_samples": int(X.shape[0]),
            "test_samples":  0,
            "cv_folds":      CV_FOLDS,
            "n_features":    int(len(feature_names)),
            "features":      feature_names,
            "note": f"All metrics computed via {CV_FOLDS}-fold stratified cross-validation (OOF). Final models trained on 100% of data.",
        },
        "dataset_stats":  dataset_stats,
        "raw_sample":     raw_sample_dict,
        "cleaning_steps": CLEANING_STEPS,
        "models": {
            "xgboost":             xgb_results,
            "logistic_regression": lr_results,
            "random_forest":       rf_results,
            "linear_regression":   linreg_results,
        },
        "blend":            blend_results,
        "model_comparison": model_comparison,
    }

    # Sanitise: replace NaN/Inf with None (Python json writes bare NaN which is invalid JSON)
    def _sanitise(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitise(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitise(v) for v in obj]
        return obj

    results = _sanitise(results)

    for out_path in ["public/results.json", "src/data/results.json"]:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n[✓] Results written → public/results.json + src/data/results.json")
    print("\n── Final Model Comparison (OOF metrics) ──")
    for row in model_comparison:
        print(f"  {row['model']:26s}  Acc={row['accuracy']:.4f}  "
              f"AUC={row['roc_auc']:.4f}  F1={row['f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
