from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def profile_csv(path: str, max_rows: int = 500) -> Dict[str, Any]:
    df = pd.read_csv(path)
    df_small = df.head(max_rows)

    dtypes = {c: str(df_small[c].dtype) for c in df_small.columns}
    missing = {c: int(df_small[c].isna().sum()) for c in df_small.columns}

    # simple target suggestions: low-cardinality columns (like yes/no, 0/1, won/lost)
    suggested = []
    for c in df_small.columns:
        nunique = df_small[c].nunique(dropna=True)
        if 2 <= nunique <= 10:
            suggested.append(c)

    sample_rows = df_small.head(5).fillna("").to_dict(orient="records")

    return {
        "columns": list(df_small.columns),
        "dtypes": dtypes,
        "missing": missing,
        "sample_rows": sample_rows,
        "suggested_target_cols": suggested[:10],
    }

def _make_pipeline(X: pd.DataFrame, model_name: str) -> Pipeline:
    # Identify categorical vs numeric
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("string")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )

    if model_name == "logreg":
        model = LogisticRegression(max_iter=2000)
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
        )
    else:
        raise ValueError("Unknown model_name")

    return Pipeline(steps=[("preprocess", pre), ("model", model)])

def _binarize_target(y: pd.Series, positive_label: Optional[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    # If already numeric 0/1-ish, coerce.
    if pd.api.types.is_numeric_dtype(y):
        y2 = y.astype(float)
        # map anything >0 to 1, else 0 (safe-ish for demo)
        y_bin = np.where(y2.fillna(0) > 0, 1, 0).astype(int)
        meta = {"type": "numeric_coerce", "positive_label": 1}
        return y_bin, meta

    # string/bool-ish
    y_str = y.astype(str).fillna("")
    uniq = sorted(list(set(y_str.tolist())))
    if positive_label and positive_label in uniq:
        pos = positive_label
    else:
        # guess: prefer common “won/yes/true/1/closed” patterns
        candidates = [u for u in uniq if u.lower() in {"yes","true","won","closed_won","1","y"}]
        pos = candidates[0] if candidates else uniq[-1]  # last as fallback

    y_bin = np.where(y_str == pos, 1, 0).astype(int)
    meta = {"type": "string_map", "positive_label": pos, "unique_values": uniq}
    return y_bin, meta

def train_best_model(
    csv_path: str,
    target_col: str,
    artifact_dir: Path,
    positive_label: Optional[str] = None,
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Drop rows missing target
    df = df.dropna(subset=[target_col])
    if len(df) < 5:
        raise ValueError("Not enough rows after dropping missing target (need at least ~5).")

    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    y, y_meta = _binarize_target(y_raw, positive_label)

    # quick leakage guard: drop obvious ID columns
    for c in list(X.columns):
        if c.lower() in {"id", "uuid", "email", "phone"}:
            X = X.drop(columns=[c])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    candidates = ["logreg", "rf"]
    results = []

    for m in candidates:
        pipe = _make_pipeline(X_train, m)
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        metrics = {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
        }
        results.append({"model": m, "metrics": metrics, "pipeline": pipe})

    # choose best by roc_auc (tie-break by precision)
    results.sort(key=lambda r: (r["metrics"]["roc_auc"], r["metrics"]["precision"]), reverse=True)
    best = results[0]

    # Save artifacts
    schema = {
        "target_col": target_col,
        "features": list(X.columns),
        "y_meta": y_meta,
        "candidate_results": [{"model": r["model"], "metrics": r["metrics"]} for r in results],
        "best_model": best["model"],
        "best_metrics": best["metrics"],
    }

    (artifact_dir / "schema.json").write_text(json.dumps(schema, indent=2))
    (artifact_dir / "metrics.json").write_text(json.dumps(best["metrics"], indent=2))
    joblib.dump(best["pipeline"], artifact_dir / "pipeline.joblib")

    return schema

def load_pipeline(artifact_dir: Path):
    return joblib.load(artifact_dir / "pipeline.joblib")