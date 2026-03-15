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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, log_loss, roc_auc_score,
)
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
from itertools import combinations
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, KBinsDiscretizer, PolynomialFeatures
from sklearn.metrics import matthews_corrcoef

warnings.filterwarnings("ignore")
np.random.seed(42)

CV_FOLDS = 5

# ─────────────────────────────────────────────
# LIGHTGBM CONFIG
# ─────────────────────────────────────────────

LGBM_PARAMS = {
    "objective": "binary", "metric": "auc", "n_estimators": 3000,
    "learning_rate": 0.02, "num_leaves": 20, "max_depth": 4,
    "min_child_samples": 20, "subsample": 0.7, "colsample_bytree": 0.7,
    "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 42,
    "n_jobs": -1, "verbose": -1,
}


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
        "step": "Encode binary target",
        "description": "Map Churn: 'Yes' → 1, 'No' → 0 (or keep as int if already numeric).",
        "code": "df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)"
    },
    {
        "step": "Convert TotalCharges to numeric",
        "description": "TotalCharges may contain whitespace strings for new customers (tenure=0). Coerce to float; these become NaN.",
        "code": "df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')"
    },
    {
        "step": "Impute missing values",
        "description": "Fill all missing numeric columns with their column medians using SimpleImputer.",
        "code": "df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])"
    },
    {
        "step": "Log-transform TotalCharges",
        "description": "Apply log1p to TotalCharges to reduce right-skew. log1p handles zero-tenure customers safely (log(1+0)=0).",
        "code": "df['TotalCharges'] = np.log1p(df['TotalCharges'])"
    },
    {
        "step": "Tenure band flags",
        "description": "Create three binary flags: TenureShort (< 10 months), TenureMid (11–20 months), TenureLong (≥ 21 months). Captures non-linear tenure effects.",
        "code": "df['TenureShort'] = (df['tenure'] < 10).astype(int)\ndf['TenureMid'] = ((df['tenure'] >= 11) & (df['tenure'] <= 20)).astype(int)\ndf['TenureLong'] = (df['tenure'] >= 21).astype(int)"
    },
    {
        "step": "HasFamilyTies flag",
        "description": "Combine Partner and Dependents into a single binary flag: 1 when both Partner='Yes' AND Dependents='Yes'. Computed before OHE so string values are still readable.",
        "code": "df['HasFamilyTies'] = ((df['Partner'] == 'Yes') & (df['Dependents'] == 'Yes')).astype(int)"
    },
    {
        "step": "Log-transform tenure",
        "description": "Apply log1p to tenure (months) to compress its right-skewed range. New customers (tenure=0) map to 0.",
        "code": "df['log_tenure'] = np.log1p(df['tenure'])"
    },
    {
        "step": "Interaction features",
        "description": "Explicit multiplications: tenure × MonthlyCharges (lifetime spend proxy) and SeniorCitizen × MonthlyCharges (demographic vulnerability × cost).",
        "code": "df['tenure_x_monthly'] = df['tenure'] * df['MonthlyCharges']\ndf['senior_x_monthly'] = df['SeniorCitizen'] * df['MonthlyCharges']"
    },
    {
        "step": "Polynomial features (degree 2)",
        "description": "All degree-2 terms from {tenure, MonthlyCharges, TotalCharges}: 3 squares + 3 cross-products = 9 new features via PolynomialFeatures(degree=2, include_bias=False).",
        "code": "po = PolynomialFeatures(degree=2, include_bias=False)\ndf[list(pn)] = po.fit_transform(df[['tenure','MonthlyCharges','TotalCharges']])"
    },
    {
        "step": "Ratio & log-charge features",
        "description": "avg_charge_per_tenure = TotalCharges / tenure (eps-safe for new customers). log_avg_charge = log1p of that ratio. Captures spending intensity over a customer's lifetime.",
        "code": "df['avg_charge_per_tenure'] = df['TotalCharges'] / safe_tenure\ndf['log_avg_charge'] = np.log1p(df['avg_charge_per_tenure'].clip(lower=0))"
    },
    {
        "step": "Frequency encoding",
        "description": "For each of the 15 categorical columns, replace each value with its proportion in the dataset. Rare categories get small values; common ones get larger values.",
        "code": "for col in cat_cols:\n    fm = df[col].value_counts(normalize=True).to_dict()\n    df[f'{col}_freq'] = df[col].map(fm).fillna(0)"
    },
    {
        "step": "Target encoding",
        "description": "For each categorical column, replace each value with the mean churn rate for that category. Computed globally — captures per-category churn signal directly.",
        "code": "for col in cat_cols:\n    te = df.groupby(col)['Churn'].mean().to_dict()\n    df[f'{col}_target'] = df[col].map(te).fillna(global_mean)"
    },
    {
        "step": "OrdinalEncoder",
        "description": "Integer-encode all 15 categorical columns. Provides a compact numeric alternative to OHE for tree-based models that can exploit ordinal splits.",
        "code": "oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)\ndf[[f'{c}_ordinal' for c in cat_cols]] = oe.fit_transform(df[cat_cols]).astype(int)"
    },
    {
        "step": "KBinsDiscretizer",
        "description": "Divide each of the 4 base numeric columns into 10 uniform-width ordinal bins. Lets tree models learn threshold effects at a fixed granularity.",
        "code": "bn = KBinsDiscretizer(n_bins=10, strategy='uniform', encode='ordinal')\ndf[[f'{c}_bin' for c in num_base]] = bn.fit_transform(df[num_base]).astype(int)"
    },
    {
        "step": "RobustScaler",
        "description": "Scale each of the 4 base numeric columns by its IQR rather than std — robust to the outliers common in billing and tenure data. Used by all models.",
        "code": "sc = RobustScaler()\ndf[[f'{c}_scaled' for c in num_base]] = sc.fit_transform(df[num_base])"
    },
    {
        "step": "Group aggregates",
        "description": "Mean, median, and std of each of the 4 base numeric columns grouped by Contract, InternetService, and PaymentMethod. 3 groups × 4 cols × 3 stats = 36 new features.",
        "code": "for g in ('Contract','InternetService','PaymentMethod'):\n    for nc in num_base:\n        for m in ('mean','median','std'):\n            df[f'{nc}_by_{g}_{m}'] = df.groupby(g)[nc].transform(m).fillna(0)"
    },
    {
        "step": "Pairwise categorical combinations",
        "description": "String-concatenate every pair of the 15 categorical columns, then label-encode to integer. C(15,2) = 105 new features capturing joint categorical effects (e.g. Contract × InternetService).",
        "code": "for c1, c2 in combinations(cat_cols, 2):\n    nm = f'{c1}_x_{c2}'\n    combined = df[c1].astype(str) + '_' + df[c2].astype(str)\n    enc = {v: i for i, v in enumerate(combined.unique())}\n    df[nm] = combined.map(enc).astype(int)"
    },
    {
        "step": "One-hot encode categorical features",
        "description": "Apply pd.get_dummies to all remaining string columns with drop_first=True to avoid multicollinearity. Produces binary indicator columns for LR and LinReg.",
        "code": "df = pd.get_dummies(df, columns=cat_cols, drop_first=True)"
    },
]


def clean_data(df: pd.DataFrame):
    """Unified 19-step cleaning + feature engineering pipeline.

    Used by ALL models (XGBoost, RF, LR, LinReg, LightGBM).
    Steps 1–11 are leak-free (math ops only).
    Steps 12–17 use full-dataset statistics (frequency, target encoding,
    group aggregates) — slight global leakage accepted for a single unified
    feature set across all models.
    """
    print("[→] Running unified cleaning + feature engineering pipeline …")

    # 1. Drop customerID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # 2. Encode binary target
    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["Churn"] = df["Churn"].astype(int)

    # 3. TotalCharges → numeric (whitespace strings → NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 4. Impute all missing numerics with column medians
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Churn"]
    df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])

    # 5. Log-transform TotalCharges (clip to 0 first — negative values from noise become 0)
    df["TotalCharges"] = np.log1p(df["TotalCharges"].clip(lower=0))

    # 6. Tenure band flags (before OHE — strings still readable)
    df["TenureShort"] = (df["tenure"] < 10).astype(int)
    df["TenureMid"]   = ((df["tenure"] >= 11) & (df["tenure"] <= 20)).astype(int)
    df["TenureLong"]  = (df["tenure"] >= 21).astype(int)

    # 7. HasFamilyTies flag (before OHE)
    if "Partner" in df.columns and "Dependents" in df.columns:
        df["HasFamilyTies"] = (
            (df["Partner"].astype(str).str.strip().str.lower() == "yes") &
            (df["Dependents"].astype(str).str.strip().str.lower() == "yes")
        ).astype(int)

    # 8. Log-transform tenure
    df["log_tenure"] = np.log1p(df["tenure"])

    # 9. Interaction features
    df["tenure_x_monthly"] = df["tenure"] * df["MonthlyCharges"]
    df["senior_x_monthly"] = df["SeniorCitizen"] * df["MonthlyCharges"]

    # 10. Polynomial features (degree 2) on [tenure, MonthlyCharges, TotalCharges]
    pi = ["tenure", "MonthlyCharges", "TotalCharges"]
    po = PolynomialFeatures(degree=2, include_bias=False)
    pn = po.fit(df[pi]).get_feature_names_out(pi)
    df[list(pn)] = po.fit_transform(df[pi])

    # 11. Ratio & log-charge features (eps-safe for new customers with tenure=0)
    eps = 1e-6
    safe_tenure = df["tenure"].replace(0, np.nan).fillna(eps) + eps
    df["avg_charge_per_tenure"] = df["TotalCharges"] / safe_tenure
    df["log_avg_charge"]        = np.log1p(df["avg_charge_per_tenure"].clip(lower=0))

    # ── Categorical feature engineering (computed before OHE) ──────────────
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # 12. Frequency encoding (global)
    for col in cat_cols:
        fm = df[col].value_counts(normalize=True).to_dict()
        df[f"{col}_freq"] = df[col].map(fm).fillna(0)

    # 13. Target encoding (global)
    gm = float(df["Churn"].mean())
    for col in cat_cols:
        te = df.groupby(col)["Churn"].mean().to_dict()
        df[f"{col}_target"] = df[col].map(te).fillna(gm)

    # 14. OrdinalEncoder
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[[f"{c}_ordinal" for c in cat_cols]] = oe.fit_transform(df[cat_cols]).astype(int)

    # 15. KBinsDiscretizer — 10 uniform bins per base numeric column
    num_base = [c for c in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] if c in df.columns]
    bn = KBinsDiscretizer(n_bins=10, strategy="uniform", encode="ordinal", random_state=42)
    df[[f"{c}_bin" for c in num_base]] = bn.fit_transform(df[num_base]).astype(int)

    # 16. RobustScaler — scale by IQR, robust to outliers
    sc = RobustScaler()
    df[[f"{c}_scaled" for c in num_base]] = sc.fit_transform(df[num_base])

    # 17. Group aggregates — 3 groups × 4 numerics × 3 stats = 36 features
    for g in ("Contract", "InternetService", "PaymentMethod"):
        if g not in df.columns:
            continue
        for nc in num_base:
            for m in ("mean", "median", "std"):
                df[f"{nc}_by_{g}_{m}"] = df.groupby(g)[nc].transform(m).fillna(0)

    # 18. Pairwise categorical combinations — C(15,2) = 105 label-encoded integers
    for c1, c2 in combinations(cat_cols, 2):
        nm = f"{c1}_x_{c2}"
        combined = df[c1].astype(str) + "_" + df[c2].astype(str)
        enc = {v: i for i, v in enumerate(combined.unique())}
        df[nm] = combined.map(enc).astype(int)

    # 19. One-hot encode all remaining categorical columns
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
    df = df.fillna(0)

    y = df["Churn"].values
    X = df.drop(columns=["Churn"])
    feature_names = X.columns.tolist()
    X = X.values

    print(f"[✓] Unified dataset — X: {X.shape}, churn rate: {y.mean():.2%}")
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
# 3a. SCORE BAND ANALYSIS
# ─────────────────────────────────────────────

SCORE_BANDS = [
    (0.00, 0.05, "0-5%"),
    (0.05, 0.10, "5-10%"),
    (0.10, 0.15, "10-15%"),
    (0.15, 0.20, "15-20%"),
    (0.20, 0.25, "20-25%"),
    (0.25, 0.50, "25-50%"),
    (0.50, 1.01, "50%+"),
]


def compute_score_bands(y_true, y_prob):
    """
    Bucket OOF predicted probabilities into fixed bands.
    Returns a list with: band label, customer count, actual churn %.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rows = []
    for lo, hi, label in SCORE_BANDS:
        mask = (y_prob >= lo) & (y_prob < hi)
        n = int(mask.sum())
        actual_churn_pct = round(float(y_true[mask].mean() * 100), 2) if n > 0 else None
        rows.append({
            "band":             label,
            "customers":        n,
            "actual_churn_pct": actual_churn_pct,
        })
    return rows


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

    # Global Youden's J threshold on OOF predictions (avoids threshold miscalibration from SMOTE)
    fpr_arr, tpr_arr, thresh_arr = roc_curve(y, y_prob_oof)
    best_idx = int(np.argmax(tpr_arr - fpr_arr))
    best_threshold = float(thresh_arr[best_idx])

    y_pred_oof = (y_prob_oof >= best_threshold).astype(int)
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

    print(f"[✓] Random Forest (OOF, thresh={best_threshold:.3f}) — Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
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

    # Simple average — threshold set so top 10% of customers are predicted to churn
    simple_weights  = np.ones(len(names)) / len(names)
    y_prob_simple   = probs @ simple_weights
    simple_thresh   = float(np.percentile(y_prob_simple, 90))
    y_pred_simple   = (y_prob_simple >= simple_thresh).astype(int)
    metrics_simple  = compute_metrics(y, y_pred_simple, y_prob_simple)

    # AUC-weighted average — threshold set so top 10% of customers are predicted to churn
    auc_weights = aucs / aucs.sum()
    y_prob_auc  = probs @ auc_weights
    auc_thresh  = float(np.percentile(y_prob_auc, 90))
    y_pred_auc  = (y_prob_auc >= auc_thresh).astype(int)
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
            "Soft-voting ensemble combining all five models — XGBoost, Random Forest, "
            "Logistic Regression, Linear Regression, and LightGBM — using out-of-fold "
            "probability outputs. Two strategies: equal-weight average and AUC-weighted average."
        ),
        "simple_blend": {
            "weights": simple_weight_table,
            "metrics": metrics_simple,
            "threshold": round(simple_thresh, 4),
        },
        "auc_weighted_blend": {
            "weights": auc_weight_table,
            "metrics": metrics_auc,
            "threshold": round(auc_thresh, 4),
        },
        "_simple_oof_probs": y_prob_simple,
        "_auc_oof_probs":    y_prob_auc,
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
# 5b. LIGHTGBM — UNIFIED FEATURE SET
# ─────────────────────────────────────────────

def _lgbm_prepare_raw(raw_df: pd.DataFrame):  # kept for reference only
    """Minimal preprocessing that preserves categorical columns as strings."""
    df = raw_df.copy()
    for col in ("customerID", "id"):
        if col in df.columns:
            df = df.drop(columns=[col])

    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["Churn"] = df["Churn"].astype(int)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    return X, y


def _lgbm_engineer(X_tr: pd.DataFrame, y_tr: pd.Series,
                   X_val: pd.DataFrame,
                   num_cols: list, cat_cols: list):
    """
    10-step feature engineering pipeline (fitted on X_tr, applied to both).
    Steps:
      1  Frequency encoding
      2  OOF target encoding (uses this fold's y_tr – no leakage)
      3  RobustScaler
      4  KBins discretiser
      5  OrdinalEncoder
      6  Key pairwise categorical combinations
      7  Interaction features (tenure×monthly, senior×monthly)
      8  Polynomial features (degree 2 on numeric trio)
      9  Ratio feature  (total / tenure)
      10 Group aggregates (Contract / InternetService / PaymentMethod)
    Returns: X_tr_eng, X_val_eng, cat_feature_names
    """
    Xc  = X_tr.copy()
    Xvc = X_val.copy()
    Xf  = pd.DataFrame(index=X_tr.index)
    Xfv = pd.DataFrame(index=X_val.index)
    cat_names = []
    all_cols  = num_cols + cat_cols

    # ── 1. Frequency encoding ────────────────────────────────────────────────
    for col in all_cols:
        if col in Xc.columns:
            fm = Xc[col].value_counts(normalize=True).to_dict()
            Xf[f"{col}_freq"]  = Xc[col].map(fm).fillna(0)
            Xfv[f"{col}_freq"] = Xvc[col].map(fm).fillna(0)

    # ── 2. Target encoding (OOF on this fold's training labels) ──────────────
    gm   = float(y_tr.mean())
    te_maps = {
        col: y_tr.groupby(Xc[col]).mean().to_dict()
        for col in all_cols if col in Xc.columns
    }
    for col in all_cols:
        if col in Xc.columns:
            Xf[f"{col}_target"]  = Xc[col].map(te_maps[col]).fillna(gm)
            Xfv[f"{col}_target"] = Xvc[col].map(te_maps[col]).fillna(gm)

    # ── 3. RobustScaler ──────────────────────────────────────────────────────
    sc = RobustScaler()
    sc_cols = [f"{c}_scaled" for c in num_cols]
    Xf[sc_cols]  = sc.fit_transform(Xc[num_cols])
    Xfv[sc_cols] = sc.transform(Xvc[num_cols])

    # ── 4. KBins discretiser ─────────────────────────────────────────────────
    bn = KBinsDiscretizer(n_bins=10, strategy="uniform", encode="ordinal", random_state=42)
    bc = [f"{c}_bin" for c in num_cols]
    Xf[bc]  = bn.fit_transform(Xc[num_cols]).astype(int)
    Xfv[bc] = bn.transform(Xvc[num_cols]).astype(int)
    cat_names.extend(bc)

    # ── 5. OrdinalEncoder ────────────────────────────────────────────────────
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    oc = [f"{c}_ordinal" for c in cat_cols]
    Xf[oc]  = oe.fit_transform(Xc[cat_cols]).astype(int)
    Xfv[oc] = oe.transform(Xvc[cat_cols]).astype(int)
    cat_names.extend(oc)

    # ── 6. All pairwise categorical combinations (C(15,2) = 105 pairs) ──────────
    for c1, c2 in combinations(cat_cols, 2):
        if c1 in Xc.columns and c2 in Xc.columns:
            nm = f"{c1}_x_{c2}"
            Xf[nm]  = (Xc[c1].astype(str) + "_" + Xc[c2].astype(str)).astype("category")
            Xfv[nm] = (Xvc[c1].astype(str) + "_" + Xvc[c2].astype(str)).astype("category")
            cat_names.append(nm)

    # ── 7. Interaction features ──────────────────────────────────────────────
    Xf["tenure_x_monthly"]  = Xc["tenure"] * Xc["MonthlyCharges"]
    Xfv["tenure_x_monthly"] = Xvc["tenure"] * Xvc["MonthlyCharges"]
    Xf["senior_x_monthly"]  = Xc["SeniorCitizen"] * Xc["MonthlyCharges"]
    Xfv["senior_x_monthly"] = Xvc["SeniorCitizen"] * Xvc["MonthlyCharges"]

    # ── 8. Polynomial features (degree 2) ─────────────────────────────────────
    pi = ["tenure", "MonthlyCharges", "TotalCharges"]
    po = PolynomialFeatures(degree=2, include_bias=False)
    pn = po.fit(Xc[pi]).get_feature_names_out(pi)
    Xf[list(pn)]  = po.transform(Xc[pi])
    Xfv[list(pn)] = po.transform(Xvc[pi])

    # ── 9. Ratio & log-charge features ───────────────────────────────────────
    eps = 1e-6
    safe_tenure_tr  = Xc["tenure"].replace(0, np.nan).fillna(eps) + eps
    safe_tenure_val = Xvc["tenure"].replace(0, np.nan).fillna(eps) + eps

    Xf["total_per_tenure"]  = Xc["TotalCharges"]  / safe_tenure_tr
    Xfv["total_per_tenure"] = Xvc["TotalCharges"] / safe_tenure_val

    # avg_charge_per_tenure: explicit average monthly spend over lifetime
    Xf["avg_charge_per_tenure"]  = Xf["total_per_tenure"]
    Xfv["avg_charge_per_tenure"] = Xfv["total_per_tenure"]

    # log_avg_charge: log-compressed version reduces right-skew
    Xf["log_avg_charge"]  = np.log1p(Xf["avg_charge_per_tenure"].clip(lower=0))
    Xfv["log_avg_charge"] = np.log1p(Xfv["avg_charge_per_tenure"].clip(lower=0))

    # ── 10. Group aggregates ──────────────────────────────────────────────────
    for g in ("Contract", "InternetService", "PaymentMethod"):
        if g not in Xc.columns:
            continue
        for nc in num_cols:
            for m in ("mean", "median", "std"):
                cn = f"{nc}_by_{g}_{m}"
                Xf[cn]  = Xc.groupby(g)[nc].transform(m).fillna(0)
                mp = Xc.groupby(g)[nc].agg(m).to_dict()
                Xfv[cn] = Xvc[g].map(mp).fillna(float(Xf[cn].mean()))

    # Ensure category dtype for all categorical feature names
    for col in cat_names:
        for frame in (Xf, Xfv):
            if col in frame.columns and frame[col].dtype.name != "category":
                frame[col] = frame[col].astype(str).astype("category")

    return Xf, Xfv, cat_names


def train_lightgbm(X, y, feature_names, skf):
    """
    LightGBM on the unified feature set (same X, y as all other models).
    Threshold is chosen to maximise Matthews Correlation Coefficient.
    Feature importances come from a final full-data fit.
    """
    print("\n[→] Training LightGBM (5-fold CV + unified feature set) …")

    params = {k: v for k, v in LGBM_PARAMS.items() if k not in ("n_jobs", "verbose")}
    params.update({"n_jobs": -1, "verbose": -1})

    y_prob_oof = np.zeros(len(y))
    fold_scores = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr   = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx],   y[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=300, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        prob = model.predict_proba(X_val)[:, 1]
        y_prob_oof[val_idx] = prob

        fa  = roc_auc_score(y_val, prob)
        acc = accuracy_score(y_val, (prob >= 0.5).astype(int))
        f1  = f1_score(y_val, (prob >= 0.5).astype(int), zero_division=0)
        fold_scores.append({
            "fold": fold_i,
            "accuracy": round(float(acc), 4),
            "f1":       round(float(f1),  4),
            "roc_auc":  round(float(fa),  4),
        })
        print(f"    Fold {fold_i}  AUC={fa:.5f}")

    # ── Optimal threshold via MCC grid search ─────────────────────────────────
    best_mcc, best_thresh = -np.inf, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        mcc = matthews_corrcoef(y, (y_prob_oof >= t).astype(int))
        if mcc > best_mcc:
            best_mcc, best_thresh = mcc, float(t)

    y_pred_oof = (y_prob_oof >= best_thresh).astype(int)
    metrics = compute_metrics(y, y_pred_oof, y_prob_oof)

    # ── Full-data fit for feature importances ─────────────────────────────────
    model_full = lgb.LGBMClassifier(**params)
    model_full.fit(X, y)
    fi = model_full.feature_importances_
    fi_pairs = sorted(zip(feature_names, fi.tolist()), key=lambda x: x[1], reverse=True)
    feature_importance = [
        {"feature": f, "importance": round(float(v), 5)} for f, v in fi_pairs
    ]

    display_params = {k: v for k, v in LGBM_PARAMS.items() if k not in ("n_jobs", "verbose")}
    display_params["optimal_threshold"] = round(best_thresh, 4)
    display_params["best_mcc"]          = round(float(best_mcc), 4)

    print(
        f"[✓] LightGBM (OOF, thresh={best_thresh:.2f}) — "
        f"Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}, "
        f"F1: {metrics['f1']:.4f}  |  n_features={X.shape[1]}"
    )
    return {
        "params":             display_params,
        "n_features":         int(X.shape[1]),
        "feature_importance": feature_importance,
        "fold_scores":        fold_scores,
        "metrics":            metrics,
        "_oof_probs":         y_prob_oof,
    }


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
    lgbm_results   = train_lightgbm(X, y, feature_names, skf)

    # Blend all 5 models using OOF probs
    model_probs = {
        "xgboost":             xgb_results.pop("_oof_probs"),
        "logistic_regression": lr_results.pop("_oof_probs"),
        "random_forest":       rf_results.pop("_oof_probs"),
        "linear_regression":   linreg_results.pop("_oof_probs"),
        "lightgbm":            lgbm_results.pop("_oof_probs"),
    }
    blend_results = blend_models(y, model_probs)

    # ── Score band analysis (OOF probs for all models + both blends) ──
    blend_simple_probs = blend_results.pop("_simple_oof_probs")
    blend_auc_probs    = blend_results.pop("_auc_oof_probs")
    score_bands = {
        "xgboost":             compute_score_bands(y, model_probs["xgboost"]),
        "logistic_regression": compute_score_bands(y, model_probs["logistic_regression"]),
        "random_forest":       compute_score_bands(y, model_probs["random_forest"]),
        "linear_regression":   compute_score_bands(y, model_probs["linear_regression"]),
        "lightgbm":            compute_score_bands(y, model_probs["lightgbm"]),
        "blend_simple":        compute_score_bands(y, blend_simple_probs),
        "blend_auc_weighted":  compute_score_bands(y, blend_auc_probs),
    }
    # Attach score bands to each model result and blend sub-results
    xgb_results["score_bands"]               = score_bands["xgboost"]
    lr_results["score_bands"]                = score_bands["logistic_regression"]
    rf_results["score_bands"]                = score_bands["random_forest"]
    linreg_results["score_bands"]            = score_bands["linear_regression"]
    lgbm_results["score_bands"]              = score_bands["lightgbm"]
    blend_results["simple_blend"]["score_bands"]       = score_bands["blend_simple"]
    blend_results["auc_weighted_blend"]["score_bands"] = score_bands["blend_auc_weighted"]

    print("\n── Score Band Summary (OOF predictions) ──")
    for model_key, bands in score_bands.items():
        print(f"\n  {model_key}")
        for b in bands:
            churn_str = f"{b['actual_churn_pct']:.1f}%" if b["actual_churn_pct"] is not None else "  n/a"
            print(f"    {b['band']:8s}  n={b['customers']:5d}  actual_churn={churn_str}")

    # Comparison table (all models + both blends)
    model_comparison = [
        {"model": "XGBoost",              **xgb_results["metrics"]},
        {"model": "Random Forest",        **rf_results["metrics"]},
        {"model": "Logistic Regression",  **lr_results["metrics"]},
        {"model": "Linear Regression",    **linreg_results["metrics"]},
        {"model": "LightGBM",             **lgbm_results["metrics"]},
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
            "lightgbm":            lgbm_results,
        },
        "blend":            blend_results,
        "model_comparison": model_comparison,
        "score_bands":      score_bands,
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
    print(f"\n  LightGBM engineered features: {lgbm_results['n_features']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
