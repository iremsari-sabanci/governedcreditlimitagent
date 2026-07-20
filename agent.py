#!/usr/bin/env python3
"""
agent.py
========
Governed-agent orchestrator for the DA523 project:
"Governed Credit Limit Increase Agent".

One call — evaluate_request() — wires the whole governance pipeline:

    intake -> access control context -> feature assembly (whitelist) ->
    scoring model -> reason codes -> risk banding -> POLICY ENGINE ->
    audit log entry (hash-chained)

and returns everything the UI needs to render (decision badge, matched
rule, rationale, reason codes, counterfactual, reviewer requirements,
log entry). Officer actions are recorded with record_review().

Used by: the notebook demo cells, the ipywidgets in-notebook UI, and
the Streamlit app (app.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from audit_log import AuditLog
from policy_engine import (PolicyEngine, build_context, WHITELIST_FIELDS)
from train_model import build_features, reason_codes, risk_band


# ----------------------------------------------------------------------------
# Artifact loading (cached module-level singleton)
# ----------------------------------------------------------------------------
class Artifacts:
    def __init__(self, data_dir="data", model_dir="models",
                 policy_path="policies/policy_set_v1.yaml"):
        self.customers = pd.read_csv(Path(data_dir) / "customers.csv")
        self.requests = pd.read_csv(Path(data_dir) / "requests.csv")
        self.users = pd.read_csv(Path(data_dir) / "users.csv")
        bundle = joblib.load(Path(model_dir) / "model_v1_0.joblib")
        self.model = bundle["model"]
        self.model_version = bundle["version"]
        self.feature_columns = bundle["feature_columns"]
        self.medians = bundle["medians"]
        self.card = json.loads(
            (Path(model_dir) / "model_card.json").read_text())
        self.monitor_dir_min = float(self.card["metrics"]["dir_gender_prod"])
        self.engine = PolicyEngine(policy_path)

    def user(self, user_id: str) -> dict:
        row = self.users[self.users.user_id == user_id]
        if row.empty:
            raise KeyError(f"Unknown user: {user_id}")
        return row.iloc[0].to_dict()


_ARTIFACTS: Artifacts | None = None


def get_artifacts(**kw) -> Artifacts:
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = Artifacts(**kw)
    return _ARTIFACTS


# ----------------------------------------------------------------------------
# Governed evaluation
# ----------------------------------------------------------------------------
def evaluate_request(*, actor_user_id: str,
                     request_id: str | None = None,
                     customer_id: str | None = None,
                     requested_increase_try: float | None = None,
                     event_type: str = "evaluation",
                     original_reviewer_id: str | None = None,
                     monitor_dir_min: float | None = None,
                     log: AuditLog | None = None,
                     art: Artifacts | None = None) -> dict:
    """Run one governed evaluation.

    Either pass request_id (an existing REQ-...) or a custom case via
    customer_id + requested_increase_try. monitor_dir_min can be
    overridden to demo the P-06 suspension. If `log` is given, the
    decision is appended to the hash-chained audit log (P-08)."""
    art = art or get_artifacts()
    actor = art.user(actor_user_id)

    # ---- intake ----------------------------------------------------------
    if request_id is not None:
        req = art.requests[art.requests.request_id == request_id]
        if req.empty:
            raise KeyError(f"Unknown request: {request_id}")
        req = req.iloc[0].to_dict()
        customer_id = req["customer_id"]
    else:
        if customer_id is None or requested_increase_try is None:
            raise ValueError("Provide request_id OR customer_id + amount.")
        req = {"request_id": f"REQ-CUSTOM-{customer_id[-6:]}",
               "customer_id": customer_id,
               "requested_increase_try": float(requested_increase_try)}

    cust_row = art.customers[art.customers.customer_id == customer_id]
    if cust_row.empty:
        raise KeyError(f"Unknown customer: {customer_id}")
    customer = cust_row.iloc[0].to_dict()

    if "increase_pct" not in req:
        req["increase_pct"] = (req["requested_increase_try"]
                               / customer["current_limit_try"])

    # ---- feature assembly (whitelist only; accesses are what we log) -----
    fields_accessed = list(WHITELIST_FIELDS)
    X = build_features(cust_row).reindex(
        columns=art.feature_columns, fill_value=0)

    # ---- scoring + explainability ---------------------------------------
    p_afford = float(art.model.predict_proba(X)[0, 1])
    band = risk_band(p_afford, float(req["increase_pct"]))
    recommendation = "approve" if p_afford >= 0.5 else "deny"
    codes = reason_codes(art.model, X, art.medians)
    counterfactual = None
    if recommendation == "deny":
        obstacle = next((c["text"] for c in codes
                         if c["direction"] == "against_approval"), "n/a")
        counterfactual = f"Strongest obstacle to approval: {obstacle}"

    # ---- policy engine ---------------------------------------------------
    ctx = build_context(
        actor=actor, customer=customer, request=req,
        model_info={"version": art.model_version,
                    "recommendation": recommendation, "risk_band": band,
                    "p_afford": p_afford,
                    "feature_names": art.feature_columns},
        reason_codes=codes, counterfactual=counterfactual,
        monitor_dir_min=(art.monitor_dir_min if monitor_dir_min is None
                         else monitor_dir_min),
        fields_accessed=fields_accessed,
        event_type=event_type,
        original_reviewer_id=original_reviewer_id,
    )
    decision = art.engine.evaluate(ctx)

    # ---- audit (P-08) ----------------------------------------------------
    log_entry = None
    if log is not None:
        log_entry = log.append({
            "event_type": event_type,
            "actor": actor["user_id"],
            "actor_role": actor["role"],
            "customer_pseudo_id": customer_id,
            "request_id": req["request_id"],
            "fields_accessed": fields_accessed,
            "model_version": art.model_version,
            "policies_evaluated": len(decision.trace),
            "matched_rule": (decision.matched_rule["rule_id"]
                             if decision.matched_rule else "DEFAULT"),
            "effect": decision.effect,
            "reason_codes": [c["feature"] + ("+" if c["direction"] ==
                             "supports_approval" else "-") for c in codes],
            "p_afford": round(p_afford, 4),
            "risk_band": band,
            "auto_approval_suspended": decision.auto_approval_suspended,
        })

    return {
        "request_id": req["request_id"],
        "customer_id": customer_id,
        "actor": actor["user_id"], "actor_role": actor["role"],
        "requested_increase_try": float(req["requested_increase_try"]),
        "increase_pct": float(req["increase_pct"]),
        "p_afford": p_afford, "risk_band": band,
        "recommendation": recommendation,
        "reason_codes": codes, "counterfactual": counterfactual,
        "effect": decision.effect,
        "matched_rule": decision.matched_rule,
        "rationale": decision.rationale,
        "reviewer_requirements": decision.reviewer_requirements,
        "auto_approval_suspended": decision.auto_approval_suspended,
        "trace": [vars(t) for t in decision.trace],
        "log_entry": log_entry,
    }


def record_review(*, log: AuditLog, result: dict, reviewer_user_id: str,
                  action: str, justification: str,
                  art: Artifacts | None = None) -> dict:
    """Record an officer's review of a REQUIRE_HUMAN_REVIEW case.
    Enforces the independence constraint for appeals (P-07)."""
    art = art or get_artifacts()
    reviewer = art.user(reviewer_user_id)

    constraint = result["reviewer_requirements"].get("reviewer_constraint")
    if constraint == "different_from_original":
        if reviewer_user_id == result.get("original_reviewer_id"):
            raise PermissionError(
                "P-07: appeal reviewer must differ from original reviewer.")
    required_role = result["reviewer_requirements"].get("reviewer_role")
    if required_role == "senior_officer" and reviewer["role"] != "senior_officer":
        raise PermissionError(
            f"P-05: this case requires a senior_officer, got {reviewer['role']}.")

    return log.append({
        "event_type": "human_review",
        "actor": reviewer_user_id,
        "actor_role": reviewer["role"],
        "customer_pseudo_id": result["customer_id"],
        "request_id": result["request_id"],
        "fields_accessed": [],
        "model_version": result.get("model_version", "n/a"),
        "policies_evaluated": 0,
        "matched_rule": (result["matched_rule"]["rule_id"]
                         if result["matched_rule"] else "DEFAULT"),
        "effect": f"REVIEW_{action.upper()}",
        "reason_codes": [],
        "reviewer_action": action,
        "reviewer_justification": justification,
        "model_recommendation": result["recommendation"],
        "override": action != result["recommendation"].replace("deny", "reject"),
    })


# ----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover — smoke test
    log = AuditLog("audit/audit_log.jsonl")
    art = get_artifacts()
    rid = art.requests.request_id.iloc[0]
    r = evaluate_request(actor_user_id="senior_elif", request_id=rid, log=log)
    print(f"{r['request_id']}: {r['effect']} via "
          f"{r['matched_rule']['rule_id'] if r['matched_rule'] else 'DEFAULT'} "
          f"| band={r['risk_band']} p={r['p_afford']:.2f}")
    if r["effect"] == "REQUIRE_HUMAN_REVIEW":
        e = record_review(log=log, result=r, reviewer_user_id="senior_elif",
                          action="approve", justification="Smoke test approval.")
        print(f"review recorded: {e['effect']} by {e['actor']}")
    ok, bad = log.verify_chain()
    print(f"audit chain intact: {ok}")
