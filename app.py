#!/usr/bin/env python3
"""
app.py
======
Streamlit deployment UI for the DA523 project:
"Governed Credit Limit Increase Agent".

Five tabs:
  1. Case Intake          login-as user, pick/enter a case, evaluate
  2. Decision             ALLOW / DENY / REQUIRE HUMAN REVIEW badge,
                          matched rule, reason codes, counterfactual,
                          appeal button
  3. Officer Review Queue pending cases, approve/deny with mandatory
                          justification, senior/two-person constraints
  4. Governance Dashboard decision mix, fairness DIR panel (legacy vs
                          production), override rate, P-06 gate demo
  5. Audit Log            filterable hash-chained log, chain
                          verification, tamper-detection demo

Run:   streamlit run app.py
Colab: see the launch cell in the notebook (pyngrok tunnel), or use the
       in-notebook ipywidgets demo instead.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from agent import evaluate_request, get_artifacts, record_review
from audit_log import AuditLog, demo_tamper_detection

st.set_page_config(page_title="Governed Credit Limit Agent",
                   page_icon="🏦", layout="wide")


# ----------------------------------------------------------------------------
# Self-bootstrap (cloud-readiness): if the data/model artifacts are missing
# or unloadable (e.g. a scikit-learn version mismatch on a fresh host),
# reproduce them from seed 42. Doubles as live proof of reproducibility —
# the app can be deployed from source code alone.
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

BADGE = {
    "ALLOW": ("✅ ALLOW", "green"),
    "DENY": ("⛔ DENY", "red"),
    "REQUIRE_HUMAN_REVIEW": ("🟠 REQUIRE HUMAN REVIEW", "orange"),
}


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "log" not in st.session_state:
    st.session_state.log = AuditLog("audit/audit_log.jsonl")
if "queue" not in st.session_state:
    st.session_state.queue = []          # pending REQUIRE_HUMAN_REVIEW results
if "closed" not in st.session_state:
    st.session_state.closed = []         # (result, review_entry) tuples
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "dir_override" not in st.session_state:
    st.session_state.dir_override = None  # P-06 demo toggle

art = get_artifacts()
log: AuditLog = st.session_state.log

st.title("🏦 Governed Credit Limit Increase Agent")
st.caption(f"Policy set v{art.engine.version} · model v{art.model_version} · "
           f"fairness monitor DIR = "
           f"{st.session_state.dir_override or art.monitor_dir_min:.3f}")

tab_intake, tab_decision, tab_queue, tab_dash, tab_audit = st.tabs(
    ["1 · Case Intake", "2 · Decision", "3 · Review Queue",
     "4 · Governance Dashboard", "5 · Audit Log"])


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
        st.toggle("Simulate fairness-gate breach (P-06): set monitor DIR=0.70",
                  key="p06_demo")
        st.session_state.dir_override = 0.70 if st.session_state.p06_demo else None

    with c2:
        st.subheader("Case")
        mode = st.radio("Input mode", ["Existing request", "Custom case"],
                        horizontal=True)
        if mode == "Existing request":
            rid = st.selectbox("Request", art.requests.request_id.tolist())
            kwargs = {"request_id": rid}
            row = art.requests[art.requests.request_id == rid].iloc[0]
            st.write(f"Customer **{row.customer_id}** · requested increase "
                     f"**{row.requested_increase_try:,.0f} TRY** "
                     f"({row.increase_pct:.0%}) · channel {row.channel}")
        else:
            cid = st.selectbox("Customer", art.customers.customer_id.tolist())
            cur = float(art.customers.set_index("customer_id")
                        .loc[cid, "current_limit_try"])
            amt = st.number_input(
                f"Requested increase (TRY) — current limit {cur:,.0f}",
                min_value=500.0, value=round(cur * 0.2, -2), step=500.0)
            kwargs = {"customer_id": cid, "requested_increase_try": amt}

        if st.button("▶ Run governed evaluation", type="primary"):
            try:
                res = evaluate_request(
                    actor_user_id=actor_id, log=log,
                    monitor_dir_min=st.session_state.dir_override, **kwargs)
                st.session_state.last_result = res
                if res["effect"] == "REQUIRE_HUMAN_REVIEW":
                    st.session_state.queue.append(res)
                label, color = BADGE[res["effect"]]
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
        st.markdown(f"## :{color}[{label}]")
        if res["auto_approval_suspended"]:
            st.warning("P-06 fairness gate active: auto-approval suspended "
                       "system-wide; ALLOW outcomes downgrade to human review.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("p(afford)", f"{res['p_afford']:.2f}")
        m2.metric("Risk band", res["risk_band"])
        m3.metric("Requested", f"{res['requested_increase_try']:,.0f} TRY")
        m4.metric("Increase", f"{res['increase_pct']:.0%}")

        st.subheader("Matched policy rule")
        mr = res["matched_rule"]
        if mr:
            st.code(f"{mr['policy_id']} / {mr['rule_id']}  —  "
                    f"{mr['policy_name']}\n{res['rationale']}")
            if res["reviewer_requirements"]:
                st.write("**Reviewer requirements:** "
                         f"{res['reviewer_requirements']}")
        else:
            st.code(f"DEFAULT (fail-safe)\n{res['rationale']}")

        st.subheader("Reason codes (P-04)")
        st.table(pd.DataFrame(res["reason_codes"])[
            ["feature", "direction", "contribution", "text"]])
        if res["counterfactual"]:
            st.info(res["counterfactual"])

        if res["recommendation"] == "deny":
            st.subheader("Adverse action — customer rights")
            st.write("The customer receives the reasons above and may "
                     "contest this decision (P-07: independent review, 48h SLA).")
            if st.button("📮 File appeal on behalf of customer"):
                appeal = evaluate_request(
                    actor_user_id=res["actor"], request_id=None,
                    customer_id=res["customer_id"],
                    requested_increase_try=res["requested_increase_try"],
                    event_type="appeal",
                    original_reviewer_id=res["actor"], log=log,
                    monitor_dir_min=st.session_state.dir_override)
                appeal["original_reviewer_id"] = res["actor"]
                st.session_state.queue.append(appeal)
                st.success("Appeal queued for independent review (P-07).")

        with st.expander("Full policy evaluation trace "
                         f"({len(res['trace'])} rules)"):
            st.dataframe(pd.DataFrame(res["trace"]))


# ----------------------------------------------------------------------------
# Tab 3 — Officer Review Queue
# ----------------------------------------------------------------------------
with tab_queue:
    actor = art.user(actor_id)
    if actor["role"] not in ("loan_officer", "senior_officer"):
        st.warning(f"Role '{actor['role']}' cannot review cases (P-01). "
                   "Switch to an officer on the Intake tab.")
    elif not st.session_state.queue:
        st.info("No pending cases.")
    else:
        for i, res in enumerate(list(st.session_state.queue)):
            with st.container(border=True):
                st.markdown(
                    f"**{res['request_id']}** · {res['customer_id']} · "
                    f"band {res['risk_band']} · p={res['p_afford']:.2f} · "
                    f"model says **{res['recommendation']}** · rule "
                    f"{res['matched_rule']['rule_id'] if res['matched_rule'] else 'DEFAULT'}")
                st.caption(res["rationale"])
                just = st.text_input("Justification (mandatory)",
                                     key=f"just_{i}")
                c1, c2, _ = st.columns([1, 1, 3])
                for action, col in (("approve", c1), ("reject", c2)):
                    if col.button(action.capitalize(),
                                  key=f"{action}_{i}"):
                        if not just.strip():
                            st.error("Justification is mandatory "
                                     "(accountability, Clause-style RACI).")
                        else:
                            try:
                                entry = record_review(
                                    log=log, result=res,
                                    reviewer_user_id=actor_id,
                                    action=action, justification=just)
                                st.session_state.closed.append((res, entry))
                                st.session_state.queue.remove(res)
                                st.success(
                                    f"{action.capitalize()}d by {actor_id}. "
                                    "Accountability chain: agent produced → "
                                    f"{actor_id} decided. Logged "
                                    f"({entry['event_id']}).")
                                st.rerun()
                            except PermissionError as e:
                                st.error(str(e))


# ----------------------------------------------------------------------------
# Tab 4 — Governance Dashboard
# ----------------------------------------------------------------------------
with tab_dash:
    entries = log.read_all()
    evals = [e for e in entries if e.get("event_type") in
             ("evaluation", "appeal")]
    reviews = [e for e in entries if e.get("event_type") == "human_review"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evaluations logged", len(evals))
    c2.metric("Human reviews", len(reviews))
    overrides = [r for r in reviews if r.get("override")]
    c3.metric("Officer override rate",
              f"{(len(overrides) / len(reviews)):.0%}" if reviews else "—",
              help="0% would indicate rubber-stamping; a healthy oversight "
                   "process disagrees with the model sometimes.")
    c4.metric("Pending queue", len(st.session_state.queue))

    st.subheader("Decision mix")
    if evals:
        mix = pd.Series([e["effect"] for e in evals]).value_counts()
        st.bar_chart(mix)
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
    st.caption("Four-fifths threshold: 0.80. The legacy model breaches the "
               "gate; P-06 would suspend auto-approval system-wide. The "
               "production model passes — and is also MORE accurate against "
               f"ground truth ({met['accuracy_vs_ground_truth_prod']:.3f} vs "
               f"{met['accuracy_vs_ground_truth_legacy']:.3f}).")


# ----------------------------------------------------------------------------
# Tab 5 — Audit Log
# ----------------------------------------------------------------------------
with tab_audit:
    entries = log.read_all()
    actor = art.user(actor_id)
    # RBAC on the log itself: officers see their own branch scope only.
    if actor["role"] in ("auditor", "admin", "senior_officer"):
        visible = entries
        st.caption(f"Role '{actor['role']}': full log visibility.")
    else:
        visible = [e for e in entries if e.get("actor") == actor_id]
        st.caption(f"Role '{actor['role']}': own entries only (P-01 applied "
                   "to the log).")

    if visible:
        df = pd.DataFrame(visible)
        cols = [c for c in ["timestamp", "event_type", "actor", "request_id",
                            "matched_rule", "effect", "reviewer_action",
                            "entry_hash"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)
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
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                demo_tamper_detection(log, line_index=1)
            st.code(buf.getvalue())
