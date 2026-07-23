#!/usr/bin/env python3
"""
train_model.py
==============
Scoring-model training for the DA523 project:
"Governed Credit Limit Increase Agent".

Trains TWO models and produces the artifacts the policy engine and UI
consume downstream:

  models/model_biased_v0_9.joblib   "Legacy" model trained WITH the
                                    region_code proxy. Exists only to
                                    demonstrate the P-06 fairness gate
                                    firing (DIR < 0.80). Never used in
                                    production paths.
  models/model_v1_0.joblib          Production model. Trained WITHOUT
                                    region_code (P-03 compliant), on
                                    the whitelisted feature set only
                                    (P-02 compliant).
  models/challenger_lr_v1_0.joblib  Logistic-regression challenger
                                    (model-risk-management practice:
                                    interpretable benchmark).
  models/model_card.json            Versioned model card: features,
                                    metrics, fairness results, lineage
                                    hash of the training data.

Explainability (P-04 input):
  reason_codes(model, x_row, ...) returns the top-k signed local
  feature contributions with plain-language text. Uses shap
  TreeExplainer when the `shap` package is installed; otherwise falls
  back to a dependency-free median-occlusion attribution (replace one
  feature with its training median, measure the change in p_afford).
  Both satisfy the same contract:
      [{"feature", "contribution", "direction", "text"}, ...]

Risk banding (P-05 input):
  risk_band(p_afford, increase_pct) -> "LOW" | "MEDIUM" | "HIGH"

Run:  python train_model.py          (expects ./data from generate_data.py)
Deps: numpy, pandas, scikit-learn, joblib. Optional: shap.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # optional dependency — used automatically when present
    import shap  # type: ignore
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SEED = 42
DATA_DIR = Path("data")
MODEL_DIR = Path("models")

MODEL_VERSION = "1.0.0"
BIASED_MODEL_VERSION = "0.9.0-legacy"

# P-02 whitelist: the ONLY fields the production model may consume.
# Must stay in sync with policies/policy_set_v1.yaml -> P-02.allowed_fields
WHITELIST_NUMERIC = [
    "monthly_income_try",
    "monthly_obligations_try",
    "current_limit_try",
    "utilization_rate",
    "months_as_customer",
    "repayment_score",
    "num_late_payments_12m",
    "kkb_score_synthetic",
]
WHITELIST_CATEGORICAL = ["employment_type"]
WHITELIST = WHITELIST_NUMERIC + WHITELIST_CATEGORICAL

# P-03 barred features (protected attributes + designated proxy)
BARRED = ["gender", "age", "region_code"]

# Risk-band thresholds (consumed by P-05). Calibrated in verify_banding().
P_LOW = 0.80
P_MEDIUM = 0.50
INCREASE_LOW_MAX = 0.30
INCREASE_HIGH_MIN = 0.60

# Plain-language templates for reason codes (P-04 explainability).
REASON_TEXT = {
    "monthly_income_try":      ("Higher income supports repayment capacity",
                                "Income level limits additional repayment capacity"),
    "monthly_obligations_try": ("Low existing obligations leave room for new credit",
                                "Existing monthly obligations are high relative to capacity"),
    "current_limit_try":       ("Current limit level supports the request",
                                "Current limit level weighs against the request"),
    "utilization_rate":        ("Low utilization of the current limit",
                                "High utilization of the current limit"),
    "months_as_customer":      ("Long relationship history with the bank",
                                "Short relationship history with the bank"),
    "repayment_score":         ("Strong internal repayment behavior",
                                "Weak internal repayment behavior"),
    "num_late_payments_12m":   ("Clean recent payment record",
                                "Recent late payments on record"),
    "kkb_score_synthetic":     ("Strong credit bureau score",
                                "Low credit bureau score"),
    "employment_type":         ("Employment type supports income stability",
                                "Employment type implies income variability"),
}


# ----------------------------------------------------------------------------
# Feature assembly
# ----------------------------------------------------------------------------
def build_features(df: pd.DataFrame, include_region: bool = False) -> pd.DataFrame:
    """One-hot encode categoricals. include_region=True is used ONLY for
    the legacy biased model in the fairness demonstration."""
    cats = WHITELIST_CATEGORICAL + (["region_code"] if include_region else [])
    X = pd.get_dummies(df[WHITELIST_NUMERIC + cats], columns=cats, dtype=float)
    return X


def enforce_p03(feature_names: list[str]) -> None:
    """Fail loudly if any barred feature leaks into the production model.
    (The policy engine re-checks this at runtime; this is the build-time
    guard.)"""
    leaked = [f for f in feature_names
              if any(f == b or f.startswith(b + "_") for b in BARRED)]
    if leaked:
        raise ValueError(f"P-03 violation: barred feature(s) in model: {leaked}")


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_all(customers: pd.DataFrame):
    y = customers["historical_approved"].astype(int)

    # ---- production model: whitelist only, region excluded --------------
    X_prod = build_features(customers, include_region=False)
    enforce_p03(list(X_prod.columns))
    Xtr, Xte, ytr, yte = train_test_split(
        X_prod, y, test_size=0.2, random_state=SEED, stratify=y
    )
    gbm = GradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr)

    # ---- challenger: interpretable logistic regression -------------------
    lr = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
    ]).fit(Xtr, ytr)

    # ---- legacy biased model: region included (for the P-06 demo) --------
    X_bias = build_features(customers, include_region=True)
    Xbtr, Xbte, ybtr, ybte = train_test_split(
        X_bias, y, test_size=0.2, random_state=SEED, stratify=y
    )
    gbm_biased = GradientBoostingClassifier(random_state=SEED).fit(Xbtr, ybtr)

    holdout = {
        "prod": (Xte, yte),
        "biased": (Xbte, ybte),
    }
    return gbm, lr, gbm_biased, X_prod, X_bias, holdout


# ----------------------------------------------------------------------------
# Explainability: reason codes (P-04)
# ----------------------------------------------------------------------------
def _occlusion_contributions(model, x_row: pd.DataFrame,
                             medians: pd.Series) -> pd.Series:
    """Dependency-free local attribution: contribution of feature f =
    p(x) - p(x with f set to its training median). Positive => the
    customer's actual value pushes affordability UP versus a typical
    customer."""
    p_base = model.predict_proba(x_row)[0, 1]
    contribs = {}
    for col in x_row.columns:
        x_mod = x_row.copy()
        x_mod.iloc[0, x_mod.columns.get_loc(col)] = medians[col]
        contribs[col] = p_base - model.predict_proba(x_mod)[0, 1]
    return pd.Series(contribs)


def _collapse_onehot(contribs: pd.Series) -> pd.Series:
    """Sum one-hot columns (employment_type_salaried, ...) back into their
    base feature so reason codes speak the business vocabulary."""
    collapsed: dict[str, float] = {}
    for col, v in contribs.items():
        base = next(
            (c for c in WHITELIST_CATEGORICAL if col.startswith(c + "_")), col
        )
        collapsed[base] = collapsed.get(base, 0.0) + float(v)
    return pd.Series(collapsed)


def reason_codes(model, x_row: pd.DataFrame, medians: pd.Series,
                 background: pd.DataFrame | None = None,
                 top_k: int = 5) -> list[dict]:
    """Return the top_k signed local contributions with plain-language
    text. Contract consumed by policy P-04 and the UI."""
    if HAS_SHAP and background is not None:
        explainer = shap.TreeExplainer(
            model, data=background.sample(200, random_state=SEED)
        )
        raw = pd.Series(
            explainer.shap_values(x_row)[0], index=x_row.columns
        )
        method = "shap_tree"
    else:
        raw = _occlusion_contributions(model, x_row, medians)
        method = "median_occlusion"

    collapsed = _collapse_onehot(raw)
    top = collapsed.reindex(collapsed.abs().sort_values(ascending=False).index)[:top_k]

    codes = []
    for feat, contrib in top.items():
        positive = contrib >= 0
        text = REASON_TEXT.get(feat, (f"{feat} supports the request",
                                      f"{feat} weighs against the request"))
        codes.append({
            "feature": feat,
            "contribution": round(float(contrib), 5),
            "direction": "supports_approval" if positive else "against_approval",
            "text": text[0] if positive else text[1],
            "method": method,
        })
    return codes


# ----------------------------------------------------------------------------
# Risk banding (P-05)
# ----------------------------------------------------------------------------
def risk_band(p_afford: float, increase_pct: float) -> str:
    if increase_pct > INCREASE_HIGH_MIN or p_afford < P_MEDIUM:
        return "HIGH"
    if p_afford >= P_LOW and increase_pct <= INCREASE_LOW_MAX:
        return "LOW"
    return "MEDIUM"


# ----------------------------------------------------------------------------
# Evaluation & verification
# ----------------------------------------------------------------------------
def dir_by_group(pred: np.ndarray, groups: pd.Series) -> float:
    rates = pd.Series(pred).groupby(groups.reset_index(drop=True)).mean()
    return float(rates.min() / rates.max())


def verify(customers, requests, gbm, lr, gbm_biased,
           X_prod, X_bias, holdout) -> dict:
    print("\n" + "=" * 68)
    print("MODEL VERIFICATION REPORT")
    print("=" * 68)

    # [1] Holdout performance vs the (biased) historical label ------------
    Xte, yte = holdout["prod"]
    auc_gbm = roc_auc_score(yte, gbm.predict_proba(Xte)[:, 1])
    auc_lr = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
    Xbte, ybte = holdout["biased"]
    auc_biased = roc_auc_score(ybte, gbm_biased.predict_proba(Xbte)[:, 1])
    print(f"\n[1] Holdout AUC vs historical label: "
          f"prod GBM={auc_gbm:.3f}  challenger LR={auc_lr:.3f}  "
          f"legacy(biased)={auc_biased:.3f}")

    # [2] Accuracy vs GROUND TRUTH on the 600 requests --------------------
    req = requests.merge(customers, on="customer_id", how="left")
    Xr_prod = build_features(req)[X_prod.columns.intersection(
        build_features(req).columns)]
    Xr_prod = Xr_prod.reindex(columns=X_prod.columns, fill_value=0)
    Xr_bias = build_features(req, include_region=True).reindex(
        columns=X_bias.columns, fill_value=0)
    gt = req["ground_truth_affordable"].astype(int)

    acc_prod = accuracy_score(gt, gbm.predict(Xr_prod))
    acc_bias = accuracy_score(gt, gbm_biased.predict(Xr_bias))
    print(f"[2] Accuracy vs hidden ground truth (600 requests): "
          f"prod={acc_prod:.3f}  legacy(biased)={acc_bias:.3f}")
    print("    (Customer-level affordability vs request-level truth: larger"
          " requested increases legitimately lower request-level accuracy.)")

    # [3] Fairness: the P-06 demonstration --------------------------------
    dir_prod = dir_by_group(gbm.predict(X_prod), customers["gender"])
    dir_bias = dir_by_group(gbm_biased.predict(X_bias), customers["gender"])
    print(f"[3] Approval DIR by gender: legacy(biased)={dir_bias:.3f}  "
          f"prod(mitigated)={dir_prod:.3f}   (gate threshold: 0.80)")
    assert dir_bias < 0.80, (
        f"Legacy model must breach the fairness gate (DIR={dir_bias:.3f})"
    )
    assert dir_prod >= 0.90, (
        f"Production model insufficiently fair (DIR={dir_prod:.3f})"
    )

    # [4] Risk-band calibration on the 600 requests -----------------------
    p = gbm.predict_proba(Xr_prod)[:, 1]
    bands = pd.Series(
        [risk_band(pi, inc) for pi, inc in zip(p, req["increase_pct"])]
    )
    counts = bands.value_counts()
    print("[4] Risk-band mix over 600 requests:")
    for b in ["LOW", "MEDIUM", "HIGH"]:
        print(f"     {b:<7} {int(counts.get(b, 0)):>4}  "
              f"({counts.get(b, 0) / len(bands):.1%})")
    assert all(counts.get(b, 0) >= 20 for b in ["LOW", "MEDIUM", "HIGH"]), (
        "Degenerate band mix: every decision path (ALLOW / REVIEW tiers) "
        "needs enough cases to demo. Recalibrate P_LOW / P_MEDIUM."
    )

    # [5] Reason-code smoke test ------------------------------------------
    medians = X_prod.median()
    sample = Xr_prod.iloc[[0]]
    codes = reason_codes(gbm, sample, medians, background=X_prod)
    print(f"[5] Reason codes ({codes[0]['method']}) for {req['request_id'].iloc[0]}:")
    for c in codes[:3]:
        print(f"     {c['direction']:<17} {c['feature']:<24} "
              f"{c['contribution']:+.4f}  {c['text']}")
    assert len(codes) >= 3, "P-04 requires >= 3 reason codes"
    print("=" * 68 + "\n")

    return {
        "auc_holdout_prod": round(auc_gbm, 4),
        "auc_holdout_challenger": round(auc_lr, 4),
        "auc_holdout_legacy_biased": round(auc_biased, 4),
        "accuracy_vs_ground_truth_prod": round(acc_prod, 4),
        "accuracy_vs_ground_truth_legacy": round(acc_bias, 4),
        "dir_gender_prod": round(dir_prod, 4),
        "dir_gender_legacy_biased": round(dir_bias, 4),
        "risk_band_mix": {b: int(counts.get(b, 0))
                          for b in ["LOW", "MEDIUM", "HIGH"]},
        "explainer_method": codes[0]["method"],
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    customers = pd.read_csv(DATA_DIR / "customers.csv")
    requests = pd.read_csv(DATA_DIR / "requests.csv")
    data_manifest = json.loads((DATA_DIR / "data_manifest.json").read_text())

    gbm, lr, gbm_biased, X_prod, X_bias, holdout = train_all(customers)
    metrics = verify(customers, requests, gbm, lr, gbm_biased,
                     X_prod, X_bias, holdout)

    # ---- persist artifacts ----------------------------------------------
    joblib.dump(
        {"model": gbm, "feature_columns": list(X_prod.columns),
         "medians": X_prod.median(), "version": MODEL_VERSION},
        MODEL_DIR / "model_v1_0.joblib",
    )
    joblib.dump(
        {"model": lr, "feature_columns": list(X_prod.columns),
         "version": MODEL_VERSION},
        MODEL_DIR / "challenger_lr_v1_0.joblib",
    )
    joblib.dump(
        {"model": gbm_biased, "feature_columns": list(X_bias.columns),
         "version": BIASED_MODEL_VERSION},
        MODEL_DIR / "model_biased_v0_9.joblib",
    )

    # ---- model card (P-08 lineage; cited in the report) ------------------
    card = {
        "model_name": "credit_limit_affordability_gbm",
        "version": MODEL_VERSION,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm": "GradientBoostingClassifier (scikit-learn)",
        "training_label": "historical_approved (known-biased legacy label; "
                          "see data_manifest.bias_design)",
        "features_whitelist_P02": WHITELIST,
        "barred_features_P03": BARRED,
        "risk_band_thresholds_P05": {
            "p_low": P_LOW, "p_medium": P_MEDIUM,
            "increase_low_max": INCREASE_LOW_MAX,
            "increase_high_min": INCREASE_HIGH_MIN,
        },
        "metrics": metrics,
        "training_data_sha256": data_manifest["sha256"]["customers"],
        "known_limitations": [
            "Trained on historically biased approval labels; fairness is "
            "achieved by proxy exclusion (P-03), monitored by P-06.",
            "Customer-level affordability score; request size handled by "
            "risk banding, not by the model.",
            "Synthetic data; coefficients not transferable to production.",
        ],
    }
    (MODEL_DIR / "model_card.json").write_text(json.dumps(card, indent=2))

    for p in sorted(MODEL_DIR.iterdir()):
        print(f"written: {p}")


if __name__ == "__main__":
    main()
