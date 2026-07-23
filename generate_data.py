#!/usr/bin/env python3
"""
generate_data.py
================
Synthetic data generator for the DA523 project:
"Governed Credit Limit Increase Agent".

Produces four artifacts in ./data/ :

  customers.csv   2,000 bank customers. Contains model features,
                  governance fields, protected attributes (audit only),
                  the planted proxy `region_code`, and the biased
                  historical label `historical_approved` used to TRAIN
                  the scoring model.
  requests.csv    600 credit-limit-increase requests. Contains the
                  hidden evaluation label `ground_truth_affordable`
                  (never shown to the agent).
  users.csv       8 system users (bank staff) with roles and branch
                  scopes, consumed by the RBAC policies (P-01).
  data_manifest.json
                  Generation lineage: seed, counts, timestamp, and
                  SHA-256 of each CSV (data-governance evidence).

Bias design (see report §Dataset):
  * TRUE affordability is a function of financial variables ONLY.
  * `region_code` is correlated with gender in the population:
        P(region in {R5, R6} | F) = 0.60   vs   0.15 for M.
  * The historical approval label is suppressed for R5/R6 customers
    regardless of true affordability (55% of affordable R5/R6
    customers were historically declined) — simulating inherited
    discrimination in legacy decisions.
  A model trained on `historical_approved` therefore learns to
  penalise R5/R6, which disproportionately affects women. The
  fairness monitor (P-06) must detect this; mitigation restores
  DIR >= 0.80.

Reproducibility: fixed seed (42). Dependencies: numpy, pandas.
Run:  python generate_data.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SEED = 42
N_CUSTOMERS = 2_000
N_REQUESTS = 600
OUT_DIR = Path("data")

REGIONS = ["R1", "R2", "R3", "R4", "R5", "R6"]
BIASED_REGIONS = {"R5", "R6"}          # the planted proxy
P_BIASED_REGION_F = 0.60               # P(region in R5/R6 | gender = F)
P_BIASED_REGION_M = 0.15               # P(region in R5/R6 | gender = M)
HIST_SUPPRESSION_RATE = 0.55           # affordable R5/R6 customers declined anyway

BRANCHES = [f"B{i:02d}" for i in range(1, 13)]        # B01 .. B12

# Debate patch A1: guardrail ceiling for income. Non-binding at
# sigma=0.45 (observed max ~193k) but protects realism if sigma is
# ever raised.
INCOME_CEILING_TRY = 250_000

REQUEST_WINDOW_END = datetime(2026, 7, 15)
REQUEST_WINDOW_DAYS = 90

rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------
# Customers
# ----------------------------------------------------------------------------
def generate_customers() -> pd.DataFrame:
    n = N_CUSTOMERS

    customer_id = [f"CUST-{i:06d}" for i in range(1, n + 1)]
    age = rng.integers(21, 76, size=n)
    gender = rng.choice(["F", "M"], size=n, p=[0.5, 0.5])

    # --- planted proxy: region depends on gender -------------------------
    p_biased = np.where(gender == "F", P_BIASED_REGION_F, P_BIASED_REGION_M)
    in_biased = rng.random(n) < p_biased
    region_code = np.where(
        in_biased,
        rng.choice(["R5", "R6"], size=n),
        rng.choice(["R1", "R2", "R3", "R4"], size=n),
    )

    # --- employment (retired only plausible above ~60) -------------------
    employment_type = rng.choice(
        ["salaried", "self-employed", "contract"], size=n, p=[0.62, 0.22, 0.16]
    )
    employment_type = np.where(
        (age >= 60) & (rng.random(n) < 0.7), "retired", employment_type
    )

    # --- income (log-normal, scaled by employment type) ------------------
    base_income = rng.lognormal(mean=10.3, sigma=0.45, size=n)  # ~30k median
    scale = pd.Series(employment_type).map(
        {"salaried": 1.00, "self-employed": 1.15, "contract": 0.85, "retired": 0.70}
    ).to_numpy()
    monthly_income_try = np.minimum(
        np.round(base_income * scale, 0), INCOME_CEILING_TRY
    )

    # --- obligations, limit, utilization, behavior -----------------------
    oblig_ratio = rng.beta(2.0, 5.0, size=n)                    # mean ~0.29
    monthly_obligations_try = np.round(monthly_income_try * oblig_ratio, 0)

    current_limit_try = np.maximum(
        2_000, np.round(monthly_income_try * rng.uniform(1.0, 3.5, size=n) / 500) * 500
    )  # existing book kept within the 4x income-multiple cap (see policy P-09)

    utilization_rate = np.clip(rng.beta(2.0, 3.0, size=n), 0.0, 1.0).round(3)

    months_as_customer = rng.integers(3, 301, size=n)

    # late payments mildly driven by utilization pressure
    lam = 0.4 + 1.6 * utilization_rate
    num_late_payments_12m = np.minimum(rng.poisson(lam), 8)

    repayment_score = np.clip(
        np.round(92 - 9 * num_late_payments_12m - 10 * utilization_rate
                 + rng.normal(0, 5, size=n)),
        0, 100,
    ).astype(int)

    # KKB/Findeks-style score on the TRUE 1-1900 scale (debate patches
    # B1 + B3). Mildly convex map of repayment_score so that all five
    # real risk brackets are populated — including "iyi" (1500-1699)
    # and "cok iyi" (1700-1900), which the previous linear map could
    # never reach. Noise keeps corr(kkb, late_payments) strongly
    # negative (~ -0.6), verified in verify().
    kkb_score_synthetic = np.clip(
        np.round(
            1 + 1899 * (repayment_score / 100) ** 1.15
            + rng.normal(0, 100, size=n)
        ),
        1, 1900,
    ).astype(int)

    # --- governance fields ------------------------------------------------
    consent_marketing = rng.random(n) < 0.60
    # hardship slightly more likely under heavy obligation load
    hardship_flag = rng.random(n) < np.where(oblig_ratio > 0.45, 0.10, 0.03)
    # Reviewer-facing context (NOT a model feature; displayed only in the
    # review queue so oversight is informed, and kept OUT of the audit
    # log for privacy — reviewers fetch it live under their role).
    hardship_reason = np.where(
        hardship_flag,
        rng.choice(["income_loss", "medical", "natural_disaster",
                    "bereavement"], size=n, p=[0.45, 0.30, 0.15, 0.10]),
        "",
    )

    branch_code = rng.choice(BRANCHES, size=n)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "gender": gender,
        "region_code": region_code,
        "branch_code": branch_code,
        "employment_type": employment_type,
        "monthly_income_try": monthly_income_try,
        "monthly_obligations_try": monthly_obligations_try,
        "current_limit_try": current_limit_try,
        "utilization_rate": utilization_rate,
        "months_as_customer": months_as_customer,
        "repayment_score": repayment_score,
        "num_late_payments_12m": num_late_payments_12m,
        "kkb_score_synthetic": kkb_score_synthetic,
        "consent_marketing": consent_marketing,
        "hardship_flag": hardship_flag,
        "hardship_reason": hardship_reason,
    })

    # --- TRUE affordability: financial variables ONLY ---------------------
    z_terms = {
        "log_income":    zscore(np.log(df["monthly_income_try"])),
        "dti_ratio":     zscore(df["monthly_obligations_try"] / df["monthly_income_try"]),
        "utilization":   zscore(df["utilization_rate"]),
        "repayment":     zscore(df["repayment_score"]),
        "late_payments": zscore(df["num_late_payments_12m"]),
    }
    # Debate patch C1: enforce standardisation so scale domination can
    # never silently re-enter the logit (e.g. if someone swaps a z-term
    # for a raw variable, generation fails loudly).
    for term_name, t in z_terms.items():
        assert abs(t.mean()) < 1e-9 and abs(t.std() - 1.0) < 1e-9, (
            f"Logit term '{term_name}' is not standardised "
            f"(mean={t.mean():.4g}, std={t.std():.4g})"
        )

    afford_logit = (
        0.9 * z_terms["log_income"]
        - 0.8 * z_terms["dti_ratio"]
        - 0.7 * z_terms["utilization"]
        + 0.6 * z_terms["repayment"]
        - 0.5 * z_terms["late_payments"]
        + rng.normal(0, 0.35, size=n)                 # idiosyncratic noise
    )
    df["_afford_logit"] = afford_logit                 # internal, dropped later
    truly_affordable = sigmoid(afford_logit) > 0.5

    # --- BIASED historical label (training label) -------------------------
    # Legacy process declined 55% of affordable R5/R6 customers.
    suppressed = (
        df["region_code"].isin(BIASED_REGIONS).to_numpy()
        & (rng.random(n) < HIST_SUPPRESSION_RATE)
    )
    df["historical_approved"] = truly_affordable & ~suppressed

    return df


# ----------------------------------------------------------------------------
# Requests
# ----------------------------------------------------------------------------
def generate_requests(customers: pd.DataFrame) -> pd.DataFrame:
    idx = rng.choice(len(customers), size=N_REQUESTS, replace=False)
    cust = customers.iloc[idx].reset_index(drop=True)

    # Right-skewed request sizes: most customers ask for modest bumps,
    # a long tail asks for large ones (median ~37%, ~38% of requests
    # <= 30%, ~20% > 60%). Replaces Uniform(0.10, 1.50), which made 71%
    # of requests exceed 50% and degenerated the risk-band mix.
    increase_pct = 0.10 + 1.40 * rng.beta(1.3, 4.5, size=N_REQUESTS)
    requested_increase_try = (
        np.round(cust["current_limit_try"] * increase_pct / 250) * 250
    ).clip(lower=500)

    channel = rng.choice(
        ["mobile", "branch", "call_center"], size=N_REQUESTS, p=[0.55, 0.30, 0.15]
    )

    offsets = rng.uniform(0, REQUEST_WINDOW_DAYS, size=N_REQUESTS)
    seconds = rng.integers(8 * 3600, 20 * 3600, size=N_REQUESTS)  # business hours-ish
    timestamp = [
        (REQUEST_WINDOW_END - timedelta(days=float(d))).replace(
            hour=0, minute=0, second=0
        ) + timedelta(seconds=int(s))
        for d, s in zip(offsets, seconds)
    ]

    # Request-level ground truth: customer affordability minus a penalty
    # for the size of the requested increase. Financial variables only —
    # region/gender play NO role here.
    req_logit = (
        cust["_afford_logit"].to_numpy()
        - 1.5 * (increase_pct - 0.25)
        + rng.normal(0, 0.20, size=N_REQUESTS)
    )
    ground_truth_affordable = sigmoid(req_logit) > 0.5

    df = pd.DataFrame({
        "request_id": [f"REQ-2026-{i:04d}" for i in range(1, N_REQUESTS + 1)],
        "customer_id": cust["customer_id"],
        "requested_increase_try": requested_increase_try,
        "increase_pct": increase_pct.round(3),
        "channel": channel,
        "timestamp": timestamp,
        "ground_truth_affordable": ground_truth_affordable,
    })
    return df.sort_values("timestamp").reset_index(drop=True)


# ----------------------------------------------------------------------------
# System users (bank staff) — consumed by RBAC policy P-01
# ----------------------------------------------------------------------------
def generate_users() -> pd.DataFrame:
    users = [
        # user_id            name              role             branch_scope
        ("officer_aylin",   "Aylin Demir",    "loan_officer",   "B01|B02|B03"),
        ("officer_mehmet",  "Mehmet Kaya",    "loan_officer",   "B04|B05|B06"),
        ("officer_zeynep",  "Zeynep Arslan",  "loan_officer",   "B07|B08|B09"),
        ("officer_burak",   "Burak Şahin",    "loan_officer",   "B10|B11|B12"),
        ("senior_elif",     "Elif Yılmaz",    "senior_officer", "ALL"),
        ("senior_can",      "Can Öztürk",     "senior_officer", "ALL"),
        ("analyst_kerem",   "Kerem Aksoy",    "model_analyst",  "NONE"),
        ("auditor_selin",   "Selin Çelik",    "auditor",        "ALL"),
    ]
    return pd.DataFrame(
        users, columns=["user_id", "name", "role", "branch_scope"]
    )


# ----------------------------------------------------------------------------
# Verification: prove the bias exists where intended — and ONLY there
# ----------------------------------------------------------------------------
def verify(customers: pd.DataFrame, requests: pd.DataFrame) -> dict:
    print("\n" + "=" * 68)
    print("STRUCTURAL REALISM CHECKS (debate patches A1, B1, B3, C1)")
    print("=" * 68)

    inc = customers["monthly_income_try"]
    n_clipped = int((inc == INCOME_CEILING_TRY).sum())
    print(f"\n[R1] Income: median={inc.median():,.0f}  p99={inc.quantile(0.99):,.0f}"
          f"  max={inc.max():,.0f}  clipped_at_ceiling={n_clipped}")

    dti = customers["monthly_obligations_try"] / customers["monthly_income_try"]
    n_dti_violations = int((dti > 1.0).sum())
    print(f"[R2] DTI ratio: median={dti.median():.3f}  p95={dti.quantile(0.95):.3f}"
          f"  max={dti.max():.3f}  obligations>income cases={n_dti_violations}")
    assert n_dti_violations == 0, "DTI sanity violated: obligations exceed income"

    kkb = customers["kkb_score_synthetic"]
    corr_kkb_late = float(kkb.corr(customers["num_late_payments_12m"]))
    print(f"[R3] KKB: min={kkb.min()}  max={kkb.max()}"
          f"  corr(kkb, late_payments)={corr_kkb_late:.3f}")
    assert corr_kkb_late < -0.40, (
        f"KKB insufficiently linked to payment behaviour (corr={corr_kkb_late:.3f})"
    )

    bracket_edges = [0, 699, 1099, 1499, 1699, 1900]
    bracket_names = ["cok_riskli_1-699", "orta_700-1099", "az_riskli_1100-1499",
                     "iyi_1500-1699", "cok_iyi_1700-1900"]
    brackets = pd.cut(kkb, bracket_edges, labels=bracket_names).value_counts()
    print("[R4] Findeks bracket coverage:")
    for name in bracket_names:
        print(f"     {name:<22} {int(brackets[name]):>5}")
    assert all(brackets[name] > 0 for name in bracket_names), (
        "One or more Findeks brackets are empty — KKB map miscalibrated"
    )

    print("\n" + "=" * 68)
    print("BIAS VERIFICATION REPORT")
    print("=" * 68)

    # 1. Proxy correlation: region vs gender
    biased_share = (
        customers.assign(in_biased=customers["region_code"].isin(BIASED_REGIONS))
        .groupby("gender")["in_biased"].mean()
    )
    print("\n[1] P(region in R5/R6) by gender  (planted proxy)")
    print(biased_share.round(3).to_string())

    # 2. Historical label (training label) — should be gender-skewed
    hist_rate = customers.groupby("gender")["historical_approved"].mean()
    dir_hist = hist_rate.min() / hist_rate.max()
    print("\n[2] Historical approval rate by gender  (BIASED training label)")
    print(hist_rate.round(3).to_string())
    print(f"    Disparate Impact Ratio (historical): {dir_hist:.3f}")

    # 3. Ground truth (evaluation label) — should be ~fair
    gt = requests.merge(
        customers[["customer_id", "gender"]], on="customer_id", how="left"
    )
    gt_rate = gt.groupby("gender")["ground_truth_affordable"].mean()
    dir_gt = gt_rate.min() / gt_rate.max()
    print("\n[3] TRUE affordability rate by gender  (hidden evaluation label)")
    print(gt_rate.round(3).to_string())
    print(f"    Disparate Impact Ratio (ground truth): {dir_gt:.3f}")

    # 4. Assertions: bias present in history, absent in ground truth
    assert dir_hist < 0.90, (
        f"Planted bias too weak: historical DIR={dir_hist:.3f} (expected < 0.90)"
    )
    assert dir_gt > 0.85, (
        f"Ground truth unexpectedly skewed: DIR={dir_gt:.3f} (expected > 0.85)"
    )
    print("\n[4] Assertions passed:")
    print(f"    historical DIR {dir_hist:.3f} < 0.90  -> bias successfully planted")
    print(f"    ground-truth DIR {dir_gt:.3f} > 0.85  -> true labels ~fair")

    print("\n[5] Volumes: "
          f"{len(customers)} customers, {len(requests)} requests, "
          f"{customers['hardship_flag'].sum()} hardship-flagged, "
          f"{requests['ground_truth_affordable'].mean():.1%} of requests truly affordable")
    print("=" * 68 + "\n")

    return {
        "dir_historical_by_gender": round(float(dir_hist), 4),
        "dir_ground_truth_by_gender": round(float(dir_gt), 4),
        "p_biased_region_given_F": round(float(biased_share["F"]), 4),
        "p_biased_region_given_M": round(float(biased_share["M"]), 4),
        "income_max_try": float(inc.max()),
        "income_clipped_count": n_clipped,
        "dti_violations": n_dti_violations,
        "corr_kkb_late_payments": round(corr_kkb_late, 4),
        "kkb_bracket_counts": {n: int(brackets[n]) for n in bracket_names},
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    customers = generate_customers()
    requests = generate_requests(customers)
    users = generate_users()

    metrics = verify(customers, requests)

    # Drop internal column before writing (agents must never see it)
    customers_out = customers.drop(columns=["_afford_logit"])

    paths = {
        "customers": OUT_DIR / "customers.csv",
        "requests": OUT_DIR / "requests.csv",
        "users": OUT_DIR / "users.csv",
    }
    customers_out.to_csv(paths["customers"], index=False)
    requests.to_csv(paths["requests"], index=False)
    users.to_csv(paths["users"], index=False)

    manifest = {
        "generator": "generate_data.py",
        "seed": SEED,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "row_counts": {
            "customers": len(customers_out),
            "requests": len(requests),
            "users": len(users),
        },
        "bias_design": {
            "proxy_feature": "region_code",
            "biased_regions": sorted(BIASED_REGIONS),
            "p_biased_region_given_F": P_BIASED_REGION_F,
            "p_biased_region_given_M": P_BIASED_REGION_M,
            "historical_suppression_rate": HIST_SUPPRESSION_RATE,
        },
        "realism_controls": {
            "income_ceiling_try": INCOME_CEILING_TRY,
            "obligations_generation": "income * Beta(2,5) — DTI structurally <= 1",
            "kkb_scale": "1-1900 (Findeks-aligned), convex map of repayment_score",
            "logit_standardisation": "asserted: each term mean~0, std~1",
        },
        "verification_metrics": metrics,
        "sha256": {name: sha256_of(p) for name, p in paths.items()},
    }
    manifest_path = OUT_DIR / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    for p in list(paths.values()) + [manifest_path]:
        print(f"written: {p}")


if __name__ == "__main__":
    main()
