#!/usr/bin/env python3
"""
agent.py
========
Governed-agent orchestrator for the DA523 project:
"Governed Credit Limit Increase Agent".

evaluate_request() wires the full governance pipeline:
    intake -> access context -> whitelisted features -> scoring ->
    reason codes -> risk banding -> POLICY ENGINE -> audit entry.

Event types:
  evaluation    production decision path (officers)
  audit_replay  NON-BINDING re-performance by assurance roles
                (auditor, model_analyst; P-01.3). A replay reports what
                production WOULD decide, but: it never enters the review
                queue, never triggers reviewer requirements, is tagged
                non_binding in the log, and is excluded from operational
                statistics. Separation of duties blocks acting, not
                re-performing.

The review queue is DERIVED FROM THE AUDIT LOG (not from UI session
state): a case is pending while the log holds a binding evaluation with
effect REQUIRE_HUMAN_REVIEW and fewer completed human_review entries
than the rule's min_reviewers. record_review() enforces reviewer role
(P-05) and two-person distinctness (P-05.3): the same person cannot
co-sign twice, the case closes only when the required number of DISTINCT
reviewers have signed, and the final outcome is approve only if ALL
reviewers approved (most-restrictive-wins). Reviewers are drawn from a
shared senior pool — no targeted routing.

Note on contestability: there is deliberately NO in-agent appeal loop.
Because every consequential adverse case already routes to a named human
(P-05), no decision is solely automated, so a KVKK m.11(ğ) objection has
nothing solely-automated to attach to; the objection channel it grants
is an external controller-and-Board process, out of scope for the agent.
An internal appeal that merely re-ran the case before a second human
would also dilute the single-owner accountability P-05 establishes.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from audit_log import AuditLog
from policy_engine import PolicyEngine, build_context, WHITELIST_FIELDS
from train_model import build_features, reason_codes, risk_band

BINDING_EVENTS = ("evaluation",)


# ----------------------------------------------------------------------------
# Artifact loading (cached module-level singleton)
# ----------------------------------------------------------------------------
class Artifacts:
    def __init__(self, data_dir="data", model_dir="models",
                 policy_path="policies/policy_set_v1.yaml"):
        self.customers = pd.read_csv(Path(data_dir) / "customers.csv",
                                     keep_default_na=False)
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

    def customer(self, customer_id: str) -> dict:
        row = self.customers[self.customers.customer_id == customer_id]
        if row.empty:
            raise KeyError(f"Unknown customer: {customer_id}")
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
                     replay_reference: str | None = None,
                     monitor_dir_min: float | None = None,
                     log: AuditLog | None = None,
                     art: Artifacts | None = None) -> dict:
    """Run one governed evaluation (or a non-binding audit replay).

    Either pass request_id (an existing REQ-...) or a custom case via
    customer_id + requested_increase_try. For event_type='audit_replay',
    replay_reference (e.g. an audit working-paper or incident-ticket id)
    is recorded in the log entry."""
    art = art or get_artifacts()
    actor = art.user(actor_user_id)
    non_binding = event_type == "audit_replay"

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

    customer = art.customer(customer_id)
    cust_row = art.customers[art.customers.customer_id == customer_id]

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
    )
    decision = art.engine.evaluate(ctx)

    # ---- audit (P-08) ----------------------------------------------------
    log_entry = None
    if log is not None:
        log_entry = log.append({
            "event_type": event_type,
            "non_binding": non_binding,
            "replay_reference": replay_reference,
            "actor": actor["user_id"],
            "actor_role": actor["role"],
            "customer_pseudo_id": customer_id,
            "branch_code": customer.get("branch_code"),
            "request_id": req["request_id"],
            "fields_accessed": fields_accessed,
            "model_version": art.model_version,
            "policies_evaluated": len(decision.trace),
            "matched_rule": (decision.matched_rule["rule_id"]
                             if decision.matched_rule else "DEFAULT"),
            "effect": decision.effect,
            "rationale": decision.rationale,
            "reason_codes": [c["feature"] + ("+" if c["direction"] ==
                             "supports_approval" else "-") for c in codes],
            "p_afford": round(p_afford, 4),
            "risk_band": band,
            "recommendation": recommendation,
            "reviewer_requirements": decision.reviewer_requirements,
            "auto_approval_suspended": decision.auto_approval_suspended,
        })

    return {
        "request_id": req["request_id"],
        "customer_id": customer_id,
        "branch_code": customer.get("branch_code"),
        "hardship_flag": bool(customer.get("hardship_flag")),
        "hardship_reason": customer.get("hardship_reason", ""),
        "actor": actor["user_id"], "actor_role": actor["role"],
        "event_type": event_type, "non_binding": non_binding,
        "replay_reference": replay_reference,
        "requested_increase_try": float(req["requested_increase_try"]),
        "increase_pct": float(req["increase_pct"]),
        "limit_income_multiple": ctx["request.limit_income_multiple"],
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


# ----------------------------------------------------------------------------
# Review queue — derived from the audit log (persistent, cross-session)
# ----------------------------------------------------------------------------
def _reviews_for(entries: list[dict], request_id: str) -> list[dict]:
    return [e for e in entries
            if e.get("event_type") == "human_review"
            and e.get("request_id") == request_id]


def pending_reviews(log: AuditLog, art: Artifacts | None = None) -> list[dict]:
    """Rebuild the pending queue from the log: the latest BINDING
    evaluation per request with effect REQUIRE_HUMAN_REVIEW, minus cases
    already closed by the required number of distinct reviewers.
    Non-binding audit replays never appear here — by construction."""
    art = art or get_artifacts()
    entries = log.read_all()
    latest: dict[str, dict] = {}
    for e in entries:
        if (e.get("event_type") in BINDING_EVENTS
                and not e.get("non_binding")):
            latest[e["request_id"]] = e            # last one wins

    pending = []
    for rid, e in latest.items():
        if e.get("effect") != "REQUIRE_HUMAN_REVIEW":
            continue
        req_min = int((e.get("reviewer_requirements") or {})
                      .get("min_reviewers", 1))
        reviews = _reviews_for(entries, rid)
        if len({r["actor"] for r in reviews}) >= req_min:
            continue                                # closed
        cust = art.customer(e["customer_pseudo_id"])
        pending.append({
            "request_id": rid,
            "customer_id": e["customer_pseudo_id"],
            "branch_code": e.get("branch_code"),
            "event_type": e["event_type"],
            "matched_rule": e.get("matched_rule"),
            "rationale": e.get("rationale", ""),
            "recommendation": e.get("recommendation"),
            "p_afford": e.get("p_afford"),
            "risk_band": e.get("risk_band"),
            "reviewer_requirements": e.get("reviewer_requirements") or {},
            "hardship_flag": bool(cust.get("hardship_flag")),
            "hardship_reason": cust.get("hardship_reason", ""),
            "reviews_done": len({r["actor"] for r in reviews}),
            "reviews_required": req_min,
            "prior_reviewers": sorted({r["actor"] for r in reviews}),
        })
    return pending


def record_review(*, log: AuditLog, case: dict, reviewer_user_id: str,
                  action: str, justification: str,
                  art: Artifacts | None = None) -> dict:
    """Record one reviewer signature on a pending case.

    Enforces: reviewer role (P-05) and two-person distinctness (P-05.3)
    — the case closes only when min_reviewers DISTINCT reviewers have
    signed; final outcome is approve only if ALL signatures approve."""
    art = art or get_artifacts()
    reviewer = art.user(reviewer_user_id)
    reqs = case.get("reviewer_requirements") or {}

    if not justification or not justification.strip():
        raise ValueError("Justification is mandatory (accountability).")

    required_role = reqs.get("reviewer_role")
    if (required_role == "senior_officer"
            and reviewer["role"] != "senior_officer"):
        raise PermissionError(
            f"P-05: this case requires a senior_officer, "
            f"got {reviewer['role']}.")
    if reviewer["role"] not in ("loan_officer", "senior_officer", "admin"):
        raise PermissionError(
            f"P-05: role '{reviewer['role']}' cannot review production "
            "cases (assurance roles observe and replay, never decide).")

    entries = log.read_all()
    prior = _reviews_for(entries, case["request_id"])
    prior_ids = {r["actor"] for r in prior}
    if reviewer_user_id in prior_ids:
        raise PermissionError(
            "P-05.3: two-person review requires DISTINCT reviewers; "
            f"{reviewer_user_id} has already signed this case.")

    req_min = int(reqs.get("min_reviewers", 1))
    seq = len(prior_ids) + 1
    complete = seq >= req_min
    all_actions = [r.get("reviewer_action") for r in prior] + [action]
    final_outcome = ("approved" if complete and
                     all(a == "approve" for a in all_actions)
                     else "rejected" if complete else None)

    return log.append({
        "event_type": "human_review",
        "non_binding": False,
        "actor": reviewer_user_id,
        "actor_role": reviewer["role"],
        "customer_pseudo_id": case["customer_id"],
        "branch_code": case.get("branch_code"),
        "request_id": case["request_id"],
        "fields_accessed": [],
        "model_version": case.get("model_version", "n/a"),
        "policies_evaluated": 0,
        "matched_rule": case.get("matched_rule", "DEFAULT"),
        "effect": f"REVIEW_{action.upper()}",
        "reason_codes": [],
        "reviewer_action": action,
        "reviewer_justification": justification,
        "review_seq": f"{seq}/{req_min}",
        "case_complete": complete,
        "final_outcome": final_outcome,
        "model_recommendation": case.get("recommendation"),
        "override": action != {"approve": "approve",
                               "deny": "reject"}.get(
                                   case.get("recommendation"), ""),
    })


# ----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover — smoke test
    log = AuditLog("audit/audit_log.jsonl")
    art = get_artifacts()
    rid = art.requests.request_id.iloc[0]
    r = evaluate_request(actor_user_id="senior_elif", request_id=rid, log=log)
    print(f"{r['request_id']}: {r['effect']} | band={r['risk_band']} "
          f"p={r['p_afford']:.2f} mult={r['limit_income_multiple']}x")
    rp = evaluate_request(actor_user_id="auditor_selin", request_id=rid,
                          event_type="audit_replay",
                          replay_reference="WP-2026-014", log=log)
    print(f"replay by {rp['actor']}: would be {rp['effect']} "
          f"(non_binding={rp['non_binding']})")
    print(f"pending queue size: {len(pending_reviews(log))} "
          f"(replay excluded by construction)")
    ok, _ = log.verify_chain()
    print(f"audit chain intact: {ok}")
