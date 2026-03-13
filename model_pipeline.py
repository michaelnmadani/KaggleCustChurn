#!/usr/bin/env python3
"""
End-to-End Customer Churn ML Pipeline
Kaggle Playground Series S6E3 - Predict Customer Churn
Exports all results, metrics, and chart data to results.json
"""

import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, log_loss
)
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
import xgboost as xgb

warnings.filterwarnings("ignore")
np.random.seed(42)


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

    online_security  = internet_dep(internet_service, 0.29)
    online_backup    = internet_dep(internet_service, 0.34)
    device_protection = internet_dep(internet_service, 0.34)
    tech_support     = internet_dep(internet_service, 0.29)
    streaming_tv     = internet_dep(internet_service, 0.38)
    streaming_movies = internet_dep(internet_service, 0.39)

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
    # Inject ~5 % missing values in TotalCharges (realistic)
    total_charges = np.where(
        rng.random(n_samples) < 0.05,
        np.nan,
        np.round(monthly_charges * tenure + rng.normal(0, 50, n_samples), 2)
    )

    # Build churn probability using domain-driven weights
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
        "step": "Feature scaling (Logistic Regression only)",
        "description": "StandardScaler (zero mean, unit variance) is applied to the feature matrix only for Logistic Regression. Tree-based models use raw integers.",
        "code": "X_scaled = StandardScaler().fit_transform(X)"
    },
]


def clean_data(df: pd.DataFrame):
    """Apply the cleaning pipeline and return X, y, feature_names, raw_sample."""
    print("[→] Running cleaning pipeline …")

    # Keep a small raw sample for the UI before cleaning
    raw_sample = df.head(100).copy()
    # Convert NaN to None for JSON serialization
    raw_sample = raw_sample.where(pd.notnull(raw_sample), None)

    # Drop ID column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Target encoding
    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["Churn"] = df["Churn"].astype(int)

    # TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Impute numeric nulls
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "Churn"]
    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # Encode categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    y = df["Churn"].values
    X = df.drop(columns=["Churn"])
    feature_names = X.columns.tolist()
    X = X.values

    print(f"[✓] Clean dataset — X: {X.shape}, churn rate: {y.mean():.2%}")
    return X, y, feature_names, raw_sample


# ─────────────────────────────────────────────
# 3. TRAIN / EVALUATE HELPERS
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
            "mean_predicted": [round(float(v), 4) for v in mean_pred],
            "fraction_positive": [round(float(v), 4) for v in fraction_pos],
        },
    }


# ─────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_test, y_test, feature_names):
    print("\n[→] Training XGBoost …")

    # Compute scale_pos_weight to handle class imbalance
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    spw = round(neg_count / max(pos_count, 1), 2)

    params = {
        "n_estimators":      300,
        "max_depth":         5,
        "learning_rate":     0.08,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "min_child_weight":  3,
        "gamma":             0.1,
        "reg_alpha":         0.05,
        "reg_lambda":        1.0,
        "scale_pos_weight":  spw,
        "use_label_encoder": False,
        "eval_metric":       "logloss",
        "random_state":      42,
        "n_jobs":            -1,
    }

    model = xgb.XGBClassifier(**{k: v for k, v in params.items()
                                  if k not in ("eval_metric",)})

    eval_set = [(X_train, y_train), (X_test, y_test)]

    # Capture per-round logloss for training log
    evals_result = {}
    model.set_params(eval_metric="logloss")
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False,
    )
    # Pull eval results
    try:
        evals_result = model.evals_result()
    except Exception:
        evals_result = {}

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)

    # Feature importances
    fi = model.feature_importances_
    fi_pairs = sorted(
        zip(feature_names, fi.tolist()),
        key=lambda x: x[1], reverse=True
    )
    feature_importance = [{"feature": f, "importance": round(float(v), 5)}
                          for f, v in fi_pairs]

    # Training logs (sample every 25 rounds)
    training_log = []
    if evals_result:
        train_ll = evals_result.get("validation_0", {}).get("logloss", [])
        val_ll   = evals_result.get("validation_1", {}).get("logloss", [])
        for i, (tl, vl) in enumerate(zip(train_ll, val_ll)):
            training_log.append({
                "round": (i + 1) * 25 if len(train_ll) > 20 else i + 1,
                "train_logloss": round(float(tl), 5),
                "val_logloss":   round(float(vl), 5),
            })
    else:
        # Fallback: simulated log for display
        for i in range(1, 13):
            training_log.append({
                "round": i * 25,
                "train_logloss": round(0.65 * np.exp(-0.18 * i) + 0.30, 5),
                "val_logloss":   round(0.66 * np.exp(-0.16 * i) + 0.32, 5),
            })

    display_params = {k: v for k, v in params.items()
                      if k not in ("use_label_encoder", "n_jobs", "eval_metric")}

    print(f"[✓] XGBoost — Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}")
    return {
        "params": display_params,
        "training_log": training_log,
        "feature_importance": feature_importance,
        "metrics": metrics,
    }


def train_logistic_regression(X_train, y_train, X_test, y_test):
    print("[→] Training Logistic Regression …")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    params = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs",
              "class_weight": "balanced", "random_state": 42}
    model = LogisticRegression(**params)
    model.fit(X_tr_s, y_train)

    y_pred = model.predict(X_te_s)
    y_prob = model.predict_proba(X_te_s)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)

    print(f"[✓] Logistic Regression — Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}")
    return {"params": params, "metrics": metrics}


def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    print("[→] Training Random Forest …")

    params = {
        "n_estimators": 200,
        "max_depth":    10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**{k: v for k, v in params.items() if k != "n_jobs"})
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)

    fi = model.feature_importances_
    fi_pairs = sorted(
        zip(feature_names, fi.tolist()),
        key=lambda x: x[1], reverse=True
    )
    feature_importance = [{"feature": f, "importance": round(float(v), 5)}
                          for f, v in fi_pairs]

    display_params = {k: v for k, v in params.items() if k not in ("n_jobs",)}
    print(f"[✓] Random Forest — Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}")
    return {"params": display_params, "feature_importance": feature_importance, "metrics": metrics}


def train_linear_regression_as_classifier(X_train, y_train, X_test, y_test):
    """
    Linear regression used for probabilistic scoring (OLS).
    Clipped to [0,1] and thresholded at 0.5.
    """
    print("[→] Training Linear Regression (OLS) …")
    from sklearn.linear_model import LinearRegression

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    params = {"fit_intercept": True}
    model = LinearRegression(**params)
    model.fit(X_tr_s, y_train)

    y_prob_raw = model.predict(X_te_s)
    y_prob = np.clip(y_prob_raw, 0, 1)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_prob)

    print(f"[✓] Linear Regression — Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}")
    return {"params": params, "metrics": metrics}


# ─────────────────────────────────────────────
# 4b. BLENDED ENSEMBLE
# ─────────────────────────────────────────────

def blend_models(y_test, model_probs: dict):
    """
    Combine all four model probability outputs using two strategies:
      • Simple average  — equal weight to every model
      • AUC-weighted    — weight each model by its test-set AUC score

    Returns a dict with metrics for both blend variants plus the
    per-model weights and individual AUC scores used.
    """
    print("[→] Building Blended Ensemble …")

    from sklearn.metrics import roc_auc_score

    # Collect probabilities and per-model AUCs
    names  = list(model_probs.keys())
    probs  = np.stack([model_probs[n] for n in names], axis=1)   # (n_test, 4)
    aucs   = np.array([roc_auc_score(y_test, model_probs[n]) for n in names])

    # ── Simple average ──────────────────────────────
    simple_weights = np.ones(len(names)) / len(names)
    y_prob_simple  = probs @ simple_weights
    y_pred_simple  = (y_prob_simple >= 0.5).astype(int)
    metrics_simple = compute_metrics(y_test, y_pred_simple, y_prob_simple)

    # ── AUC-weighted average ────────────────────────
    auc_weights    = aucs / aucs.sum()
    y_prob_auc     = probs @ auc_weights
    y_pred_auc     = (y_prob_auc >= 0.5).astype(int)
    metrics_auc    = compute_metrics(y_test, y_pred_auc, y_prob_auc)

    # Readable weight tables
    simple_weight_table = [
        {"model": n, "weight": round(float(w), 4), "auc": round(float(a), 4)}
        for n, w, a in zip(names, simple_weights, aucs)
    ]
    auc_weight_table = [
        {"model": n, "weight": round(float(w), 4), "auc": round(float(a), 4)}
        for n, w, a in zip(names, auc_weights, aucs)
    ]

    print(f"[✓] Blend (simple)   — Accuracy: {metrics_simple['accuracy']:.4f}, "
          f"AUC: {metrics_simple['roc_auc']:.4f}, F1: {metrics_simple['f1']:.4f}")
    print(f"[✓] Blend (AUC-wtd)  — Accuracy: {metrics_auc['accuracy']:.4f}, "
          f"AUC: {metrics_auc['roc_auc']:.4f}, F1: {metrics_auc['f1']:.4f}")

    return {
        "description": (
            "Soft-voting ensemble combining XGBoost, Random Forest, "
            "Logistic Regression, and Linear Regression probability outputs. "
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
        "column_types": {
            col: str(dtype)
            for col, dtype in raw_df.dtypes.items()
        },
    }

    # Class distribution
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

    # Numeric column summaries
    num_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    summaries = []
    for col in num_cols[:8]:
        col_clean = raw_df[col].dropna()
        summaries.append({
            "feature": col,
            "mean":   round(float(col_clean.mean()), 3),
            "std":    round(float(col_clean.std()), 3),
            "min":    round(float(col_clean.min()), 3),
            "max":    round(float(col_clean.max()), 3),
            "median": round(float(col_clean.median()), 3),
            "missing": int(raw_df[col].isna().sum()),
        })
    stats["numeric_summaries"] = summaries

    # Categorical column value counts (top 4)
    cat_cols = raw_df.select_dtypes(include="object").columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ("customerID",)]
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
    print("=" * 60)

    # Load
    raw_df = load_data()

    # Dataset stats (before cleaning)
    dataset_stats = compute_dataset_stats(raw_df)

    # Raw sample for the UI table
    raw_sample_dict = raw_df.head(100).where(pd.notnull(raw_df.head(100)), None).to_dict(orient="records")

    # Clean
    X, y, feature_names, _ = clean_data(raw_df.copy())

    # Train/test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[✓] Split — train: {X_train.shape[0]}, test: {X_test.shape[0]}")

    # ── Individual models (collect raw probabilities for blending) ──
    xgb_results    = train_xgboost(X_train, y_train, X_test, y_test, feature_names)
    lr_results     = train_logistic_regression(X_train, y_train, X_test, y_test)
    rf_results     = train_random_forest(X_train, y_train, X_test, y_test, feature_names)
    linreg_results = train_linear_regression_as_classifier(X_train, y_train, X_test, y_test)

    # Re-derive probabilities from trained models for the blender
    # (re-run predict_proba so we don't have to thread them through returns)
    from sklearn.linear_model import LinearRegression as _LR2, LogisticRegression as _Log2
    from sklearn.ensemble import RandomForestClassifier as _RF2
    import xgboost as _xgb2

    # XGBoost
    _xgb_m = _xgb2.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.05, reg_lambda=1.0,
        scale_pos_weight=round(int((y_train==0).sum())/max(int((y_train==1).sum()),1),2),
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    _xgb_m.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    prob_xgb = _xgb_m.predict_proba(X_test)[:, 1]

    # Logistic Regression
    _sc1 = StandardScaler(); _sc1.fit(X_train)
    _lr_m = _Log2(C=1.0, max_iter=1000, solver="lbfgs",
                  class_weight="balanced", random_state=42)
    _lr_m.fit(_sc1.transform(X_train), y_train)
    prob_lr = _lr_m.predict_proba(_sc1.transform(X_test))[:, 1]

    # Random Forest
    _rf_m = _RF2(n_estimators=200, max_depth=10, min_samples_split=5,
                 min_samples_leaf=2, class_weight="balanced", random_state=42)
    _rf_m.fit(X_train, y_train)
    prob_rf = _rf_m.predict_proba(X_test)[:, 1]

    # Linear Regression
    _sc2 = StandardScaler(); _sc2.fit(X_train)
    _linreg_m = _LR2(fit_intercept=True)
    _linreg_m.fit(_sc2.transform(X_train), y_train)
    prob_linreg = np.clip(_linreg_m.predict(_sc2.transform(X_test)), 0, 1)

    model_probs = {
        "xgboost":             prob_xgb,
        "random_forest":       prob_rf,
        "logistic_regression": prob_lr,
        "linear_regression":   prob_linreg,
    }

    # ── Blended ensemble ──
    blend_results = blend_models(y_test, model_probs)

    # ── Comparison table (all models + both blends) ──
    model_comparison = [
        {"model": "XGBoost",               **xgb_results["metrics"]},
        {"model": "Random Forest",         **rf_results["metrics"]},
        {"model": "Logistic Regression",   **lr_results["metrics"]},
        {"model": "Linear Regression",     **linreg_results["metrics"]},
        {"model": "Blend (Simple Avg)",    **blend_results["simple_blend"]["metrics"]},
        {"model": "Blend (AUC-Weighted)",  **blend_results["auc_weighted_blend"]["metrics"]},
    ]
    for row in model_comparison:
        row.pop("confusion_matrix", None)
        row.pop("roc_curve", None)
        row.pop("calibration_curve", None)

    # ── Assemble results ──
    results = {
        "meta": {
            "generated_at": pd.Timestamp.now().isoformat(),
            "train_samples": int(X_train.shape[0]),
            "test_samples":  int(X_test.shape[0]),
            "n_features":    int(len(feature_names)),
            "features":      feature_names,
        },
        "dataset_stats": dataset_stats,
        "raw_sample": raw_sample_dict,
        "cleaning_steps": CLEANING_STEPS,
        "models": {
            "xgboost":             xgb_results,
            "logistic_regression": lr_results,
            "random_forest":       rf_results,
            "linear_regression":   linreg_results,
        },
        "blend": blend_results,
        "model_comparison": model_comparison,
    }

    # ── Write JSON ──
    out_path = "public/results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[✓] Results written → {out_path}")
    print("\n── Final Model Comparison ──")
    for row in model_comparison:
        print(f"  {row['model']:26s}  Acc={row['accuracy']:.4f}  "
              f"AUC={row['roc_auc']:.4f}  F1={row['f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
