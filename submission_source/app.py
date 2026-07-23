#!/usr/bin/env python3
"""
app.py
======
Streamlit deployment UI for the DA523 project:
"Governed Credit Limit Increase Agent".

Five tabs:
  1. Case Intake          login-as user (branch scope shown), pick/enter
                          a case with the customer's BRANCH visible,
                          audit-replay mode for assurance roles
  2. Decision             ALLOW / DENY / REQUIRE HUMAN REVIEW badge
                          (NON-BINDING banner for replays), matched rule,
                          reason codes, counterfactual
  3. Officer Review Queue derived from the AUDIT LOG (persistent across
                          sessions), hardship reason shown before
                          justification, co-sign progress (n/m),
                          mandatory justification
  4. Governance Dashboard decision mix & override rate (binding events
                          only — replays excluded), fairness DIR panel
  5. Audit Log            role-based visibility matrix, reviewer
                          justifications visible, raw-entry inspector,
                          chain verification, tamper demo

Run:   streamlit run app.py
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Governed Credit Limit Agent",
                   page_icon="🏦", layout="wide")


# ----------------------------------------------------------------------------
# Self-bootstrap (cloud-readiness): if the data/model artifacts are missing
# or unloadable (e.g. a scikit-learn version mismatch on a fresh host),
# reproduce them from seed 42. Doubles as live proof of reproducibility.
# ----------------------------------------------------------------------------
def _bootstrap() -> None:
    import subprocess
    import sys
    from pathlib import Path

    healthy = all(Path(p).exists() for p in
                  ("data/customers.csv", "data/requests.csv",
                   "data/users.csv", "models/model_v1_0.joblib",
                   "models/model_card.json",
                   "policies/policy_set_v1.yaml"))
    if healthy:
        try:  # unpickle AND score once — catches cross-version breakage
            import joblib
            b = joblib.load("models/model_v1_0.joblib")
            b["model"].predict_proba(
                b["medians"].to_frame().T[b["feature_columns"]])
        except Exception:
            healthy = False

    if not healthy:
        with st.spinner("First boot: reproducing dataset and models from "
                        "seed 42 (~1 minute, one time only)..."):
            subprocess.run([sys.executable, "generate_data.py"], check=True)
            subprocess.run([sys.executable, "train_model.py"], check=True)
        st.toast("Artifacts reproduced from seed. Reproducibility: proven.")


_bootstrap()

from agent import (evaluate_request, get_artifacts, pending_reviews,  # noqa: E402
                   record_review)
from audit_log import AuditLog, demo_tamper_detection  # noqa: E402

BADGE = {
    "ALLOW": ("✅ ALLOW", "green"),
    "DENY": ("⛔ DENY", "red"),
    "REQUIRE_HUMAN_REVIEW": ("🟠 REQUIRE HUMAN REVIEW", "orange"),
}
ASSURANCE_ROLES = ("auditor", "model_analyst")

if "log" not in st.session_state:
    st.session_state.log = AuditLog("audit/audit_log.jsonl")
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "dir_override" not in st.session_state:
    st.session_state.dir_override = None

art = get_artifacts()
log: AuditLog = st.session_state.log

st.title("🏦 Governed Credit Limit Increase Agent")
st.caption(f"Policy set v{art.engine.version} · model v{art.model_version} · "
           f"fairness monitor DIR = "
           f"{st.session_state.dir_override or art.monitor_dir_min:.3f}")

tab_intake, tab_decision, tab_queue, tab_dash, tab_audit = st.tabs(
    ["1 · Case Intake", "2 · Decision", "3 · Review Queue",
     "4 · Governance Dashboard", "5 · Audit Log"])


def cust_label(cid: str) -> str:
    c = art.customer(cid)
    return f"{cid}  ·  branch {c['branch_code']}"


def req_label(rid: str) -> str:
    row = art.requests[art.requests.request_id == rid].iloc[0]
    c = art.customer(row.customer_id)
    return f"{rid}  ·  {row.customer_id} @ {c['branch_code']}"


# ----------------------------------------------------------------------------
# Tab 1 — Case Intake
# ----------------------------------------------------------------------------
with tab_intake:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Acting user")
        actor_id = st.selectbox(
            "Login as", art.users.user_id.tolist(),
            format_func=lambda u: f"{u}  ({art.user(u)['role']})")
        actor = art.user(actor_id)
        st.info(f"**Role:** {actor['role']}  \n"
                f"**Branch scope:** {actor['branch_scope']}")

        is_assurance = actor["role"] in ASSURANCE_ROLES
        replay_mode = False
        replay_ref = None
        if is_assurance:
            replay_mode = st.checkbox(
                "🔍 Audit replay mode (NON-BINDING re-performance)",
                value=True,
                help="Assurance roles cannot act in production (P-01.1) "
                     "but may re-perform any case non-bindingly (P-01.3). "
                     "Replays never enter the review queue and are "
                     "excluded from operational statistics.")
            if replay_mode:
                replay_ref = st.text_input(
                    "Working-paper / incident reference (logged)",
                    value="WP-2026-001")

        st.toggle("Simulate fairness-gate breach (P-06): monitor DIR=0.70",
                  key="p06_demo")
        st.session_state.dir_override = (0.70 if st.session_state.p06_demo
                                         else None)

    with c2:
        st.subheader("Case")
        mode = st.radio("Input mode", ["Existing request", "Custom case"],
                        horizontal=True)
        if mode == "Existing request":
            rid = st.selectbox("Request", art.requests.request_id.tolist(),
                               format_func=req_label)
            row = art.requests[art.requests.request_id == rid].iloc[0]
            cust = art.customer(row.customer_id)
            kwargs = {"request_id": rid}
            st.write(f"Customer **{row.customer_id}** · branch "
                     f"**{cust['branch_code']}** · requested increase "
                     f"**{row.requested_increase_try:,.0f} TRY** "
                     f"({row.increase_pct:.0%}) · channel {row.channel}")
            if str(actor["branch_scope"]) not in ("ALL", "NONE") and \
                    cust["branch_code"] not in str(actor["branch_scope"]):
                st.warning(f"Branch {cust['branch_code']} is outside your "
                           f"scope ({actor['branch_scope']}) — P-01.2 will "
                           "deny a production evaluation.")
        else:
            cid = st.selectbox("Customer", art.customers.customer_id.tolist(),
                               format_func=cust_label)
            cust = art.customer(cid)
            cur = float(cust["current_limit_try"])
            st.write(f"Branch **{cust['branch_code']}** · current limit "
                     f"**{cur:,.0f} TRY** · monthly income "
                     f"**{cust['monthly_income_try']:,.0f} TRY** "
                     f"(statutory cap P-09: total limit ≤ 4.0× income = "
                     f"{4 * cust['monthly_income_try']:,.0f} TRY)")
            amt = st.number_input(
                "Requested increase (TRY)", min_value=500.0,
                value=round(cur * 0.2, -2), step=500.0)
            kwargs = {"customer_id": cid, "requested_increase_try": amt}

        if st.button("▶ Run governed evaluation", type="primary"):
            try:
                res = evaluate_request(
                    actor_user_id=actor_id, log=log,
                    event_type="audit_replay" if (is_assurance and
                                                  replay_mode)
                    else "evaluation",
                    replay_reference=replay_ref,
                    monitor_dir_min=st.session_state.dir_override, **kwargs)
                st.session_state.last_result = res
                label, color = BADGE[res["effect"]]
                if res["non_binding"]:
                    st.markdown(f"### :{color}[{label}] — NON-BINDING "
                                "AUDIT REPLAY")
                else:
                    st.markdown(f"### :{color}[{label}]")
                st.write(res["rationale"])
                st.caption("Full detail on the **Decision** tab.")
            except Exception as e:
                st.error(f"Evaluation blocked: {e}")


# ----------------------------------------------------------------------------
# Tab 2 — Decision
# ----------------------------------------------------------------------------
with tab_decision:
    res = st.session_state.last_result
    if res is None:
        st.info("Run an evaluation on the Case Intake tab first.")
    else:
        label, color = BADGE[res["effect"]]
        if res["non_binding"]:
            st.warning(f"🔍 **AUDIT REPLAY — NON-BINDING** "
                       f"(ref: {res['replay_reference']}). The badge below "
                       "shows what production WOULD decide. Nothing is "
                       "queued, released, or communicated to the customer.")
        st.markdown(f"## :{color}[{label}]")
        if res["auto_approval_suspended"]:
            st.warning("P-06 fairness gate active: auto-approval suspended "
                       "system-wide; ALLOW outcomes downgrade to review.")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("p(afford)", f"{res['p_afford']:.2f}")
        m2.metric("Risk band", res["risk_band"])
        m3.metric("Requested", f"{res['requested_increase_try']:,.0f} TRY")
        m4.metric("Increase", f"{res['increase_pct']:.0%}")
        m5.metric("Limit ÷ income", f"{res['limit_income_multiple']:.1f}×",
                  help="P-09 statutory cap: 4.0×")
        st.caption(f"Customer {res['customer_id']} · branch "
                   f"**{res['branch_code']}** · actor {res['actor']} "
                   f"({res['actor_role']})")

        st.subheader("Matched policy rule")
        mr = res["matched_rule"]
        if mr:
            st.code(f"{mr['policy_id']} / {mr['rule_id']}  —  "
                    f"{mr['policy_name']}\n{res['rationale']}")
        else:
            st.code(f"DEFAULT (fail-safe)\n{res['rationale']}")
        if res["reviewer_requirements"]:
            st.write("**Reviewer requirements (most-restrictive union of "
                     "all matched review rules):** "
                     f"{res['reviewer_requirements']}")

        st.subheader("Reason codes (P-04)")
        st.table(pd.DataFrame(res["reason_codes"])[
            ["feature", "direction", "contribution", "text"]])
        if res["counterfactual"]:
            st.info(res["counterfactual"])

        if (res["recommendation"] == "deny" and not res["non_binding"]):
            st.subheader("Adverse action — customer rights")
            st.write("The customer receives the reason codes above. Because "
                     "this case was decided by a named human (P-05), it is "
                     "not a *solely* automated decision. Under KVKK "
                     "m.11(ğ) the customer may object to the controller and, "
                     "if unsatisfied, escalate to the Personal Data "
                     "Protection Board — an EXTERNAL channel, deliberately "
                     "not an in-agent re-run (which would dilute the single "
                     "accountable owner of the decision).")

        with st.expander(f"Full policy evaluation trace "
                         f"({len(res['trace'])} rules)"):
            st.dataframe(pd.DataFrame(res["trace"]))


# ----------------------------------------------------------------------------
# Tab 3 — Officer Review Queue (derived from the audit log)
# ----------------------------------------------------------------------------
with tab_queue:
    actor = art.user(actor_id)
    queue = pending_reviews(log, art)
    st.caption(f"{len(queue)} pending case(s) — rebuilt from the audit log, "
               "so the queue survives restarts and is visible to every "
               "authorised officer, not just the session that created it.")
    if actor["role"] not in ("loan_officer", "senior_officer"):
        st.warning(f"Role '{actor['role']}' cannot review production cases "
                   "(assurance roles observe and replay, never decide). "
                   "Switch to an officer on the Intake tab.")
    elif not queue:
        st.info("No pending cases.")
    else:
        for i, case in enumerate(queue):
            with st.container(border=True):
                st.markdown(
                    f"**{case['request_id']}** · {case['customer_id']} @ "
                    f"branch **{case['branch_code']}** · "
                    f"{case['event_type'].upper()} · band "
                    f"{case['risk_band']} · p={case['p_afford']:.2f} · "
                    f"model says **{case['recommendation']}** · rule "
                    f"{case['matched_rule']}")
                st.caption(case["rationale"])
                if case["hardship_flag"]:
                    st.error(f"⚠️ HARDSHIP — reason on file: "
                             f"**{case['hardship_reason'] or 'unspecified'}**"
                             " (shown to reviewers only; kept out of the "
                             "audit log for privacy)")
                reqs = case["reviewer_requirements"]
                sig = (f"signatures {case['reviews_done']}/"
                       f"{case['reviews_required']}")
                if case["prior_reviewers"]:
                    sig += f" (signed: {', '.join(case['prior_reviewers'])})"
                st.write(f"**Requirements:** {reqs or 'standard review'} · "
                         f"{sig}")
                just = st.text_input("Justification (mandatory)",
                                     key=f"just_{case['request_id']}_{i}")
                c1, c2, _ = st.columns([1, 1, 3])
                for action, col in (("approve", c1), ("reject", c2)):
                    if col.button(action.capitalize(),
                                  key=f"{action}_{case['request_id']}_{i}"):
                        try:
                            entry = record_review(
                                log=log, case=case,
                                reviewer_user_id=actor_id,
                                action=action, justification=just)
                            if entry["case_complete"]:
                                st.success(
                                    f"Case closed — final outcome: "
                                    f"**{entry['final_outcome']}** "
                                    f"({entry['review_seq']}). Logged "
                                    f"{entry['event_id']}.")
                            else:
                                st.success(
                                    f"Signature {entry['review_seq']} "
                                    "recorded — awaiting a DIFFERENT "
                                    "reviewer to co-sign. Logged "
                                    f"{entry['event_id']}.")
                            st.rerun()
                        except (PermissionError, ValueError) as e:
                            st.error(str(e))


# ----------------------------------------------------------------------------
# Tab 4 — Governance Dashboard (binding events only)
# ----------------------------------------------------------------------------
with tab_dash:
    entries = log.read_all()
    binding = [e for e in entries if not e.get("non_binding")]
    evals = [e for e in binding if e.get("event_type") in
             ("evaluation",)]
    reviews = [e for e in binding if e.get("event_type") == "human_review"]
    replays = [e for e in entries if e.get("non_binding")]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Binding evaluations", len(evals))
    c2.metric("Human review signatures", len(reviews))
    overrides = [r for r in reviews if r.get("override")]
    c3.metric("Officer override rate",
              f"{(len(overrides) / len(reviews)):.0%}" if reviews else "—",
              help="0% would indicate rubber-stamping.")
    c4.metric("Pending queue", len(pending_reviews(log, art)))
    c5.metric("Audit replays (excluded)", len(replays),
              help="Non-binding re-performances by assurance roles; "
                   "excluded from all operational statistics.")

    st.subheader("Decision mix (binding only)")
    if evals:
        st.bar_chart(pd.Series([e["effect"] for e in evals]).value_counts())
    else:
        st.caption("Run some evaluations first.")

    st.subheader("Fairness panel (P-06)")
    met = art.card["metrics"]
    fair = pd.DataFrame({
        "model": ["legacy v0.9 (with region proxy)",
                  "production v1.0 (proxy excluded)"],
        "approval DIR by gender": [met["dir_gender_legacy_biased"],
                                   met["dir_gender_prod"]],
    }).set_index("model")
    st.bar_chart(fair)
    st.caption("Four-fifths threshold: 0.80. The mitigated model is also "
               "MORE accurate against ground truth "
               f"({met['accuracy_vs_ground_truth_prod']:.3f} vs "
               f"{met['accuracy_vs_ground_truth_legacy']:.3f}).")


# ----------------------------------------------------------------------------
# Tab 5 — Audit Log (role-based visibility matrix)
# ----------------------------------------------------------------------------
with tab_audit:
    entries = log.read_all()
    actor = art.user(actor_id)
    role = actor["role"]

    if role in ("auditor", "admin"):
        visible = entries
        note = "assurance/admin: FULL log visibility (observe everything)."
    elif role == "senior_officer":
        visible = entries
        note = ("senior_officer: full visibility — span of accountability "
                "is bank-wide (branch_scope=ALL); supervisors answer for "
                "their span of control and must be able to detect "
                "rubber-stamping across it.")
    elif role == "loan_officer":
        scope = str(actor["branch_scope"]).split("|")
        visible = [e for e in entries if e.get("branch_code") in scope
                   or e.get("actor") == actor_id]
        note = (f"loan_officer: entries within own branch scope "
                f"({actor['branch_scope']}) — visibility follows span of "
                "accountability, not curiosity.")
    elif role == "model_analyst":
        visible = [e for e in entries if e.get("event_type") in
                   ("evaluation", "audit_replay")]
        note = ("model_analyst: technical decision entries only (for error "
                "analysis); human-review justifications are out of scope.")
    else:
        visible, note = [], f"role '{role}': no log access."
    st.caption(f"Visibility — {note}")

    if visible:
        df = pd.DataFrame(visible)
        cols = [c for c in ["timestamp", "event_type", "non_binding",
                            "actor", "branch_code", "request_id",
                            "matched_rule", "effect", "reviewer_action",
                            "reviewer_justification", "review_seq",
                            "final_outcome", "replay_reference",
                            "entry_hash"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)
        with st.expander("🔎 Inspect a raw entry (full JSONL record — the "
                         "primary source)"):
            ev = st.selectbox("event_id",
                              [e["event_id"] for e in visible])
            st.json(next(e for e in visible if e["event_id"] == ev))
    else:
        st.info("No visible entries yet.")

    c1, c2 = st.columns(2)
    if c1.button("🔒 Verify chain integrity"):
        ok, bad = log.verify_chain()
        (st.success if ok else st.error)(
            f"Chain intact: {ok}" + ("" if ok else
                                     f" — first broken entry: #{bad}"))
    if c2.button("🧪 Tamper demo (forge one entry, detect, restore)"):
        if len(entries) < 2:
            st.warning("Need at least 2 log entries for the demo.")
        else:
            import contextlib
            import io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                demo_tamper_detection(log, line_index=1)
            st.code(buf.getvalue())
