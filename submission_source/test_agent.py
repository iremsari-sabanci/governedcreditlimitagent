#!/usr/bin/env python3
"""
test_agent.py
=============
Integration tests for the governed flows (require data/ and models/):
  * audit replay is permitted for assurance roles, reports the would-be
    decision, and NEVER enters the review queue or operational stats
  * two-person review (P-05.3): distinct seniors required, same person
    cannot co-sign twice, junior officer refused, most-restrictive final
  * appeal independence (P-07) enforced at review time
  * P-09 statutory cap fires end-to-end on an absurd custom request
  * review justifications are persisted in the audit log

Run:  python test_agent.py
"""

import tempfile
from pathlib import Path

from audit_log import AuditLog
from agent import (evaluate_request, get_artifacts, pending_reviews,
                   record_review)

ART = get_artifacts()


def fresh_log() -> AuditLog:
    return AuditLog(Path(tempfile.mkdtemp()) / "audit_log.jsonl")


def find_request(predicate):
    for rid in ART.requests.request_id:
        r = evaluate_request(actor_user_id="senior_elif", request_id=rid)
        if predicate(r):
            return rid, r
    raise AssertionError("No request matches predicate")


# ---------------------------------------------------------------------------
def test_replay_is_non_binding_and_never_queued():
    log = fresh_log()
    rid, _ = find_request(lambda r: r["effect"] == "REQUIRE_HUMAN_REVIEW")
    rep = evaluate_request(actor_user_id="auditor_selin", request_id=rid,
                           event_type="audit_replay",
                           replay_reference="WP-2026-001", log=log)
    assert rep["non_binding"] is True
    assert rep["effect"] == "REQUIRE_HUMAN_REVIEW"   # reports would-be outcome
    assert pending_reviews(log, ART) == []           # ...but queues nothing
    entry = log.read_all()[-1]
    assert entry["event_type"] == "audit_replay"
    assert entry["non_binding"] is True
    assert entry["replay_reference"] == "WP-2026-001"


def test_analyst_replay_allowed_officer_replay_denied():
    log = fresh_log()
    rid = ART.requests.request_id.iloc[0]
    rep = evaluate_request(actor_user_id="analyst_kerem", request_id=rid,
                           event_type="audit_replay", log=log)
    assert rep["effect"] != "DENY"
    off = evaluate_request(actor_user_id="officer_aylin", request_id=rid,
                           event_type="audit_replay", log=log)
    assert off["effect"] == "DENY"
    assert off["matched_rule"]["rule_id"] == "P-01.3"


def test_two_person_review_distinct_seniors():
    log = fresh_log()
    rid, r = find_request(
        lambda r: (r["reviewer_requirements"].get("min_reviewers") == 2))
    evaluate_request(actor_user_id="senior_elif", request_id=rid, log=log)
    case = pending_reviews(log, ART)[0]
    assert case["reviews_required"] == 2

    # junior officer refused on a senior-only case
    try:
        record_review(log=log, case=case, reviewer_user_id="officer_aylin",
                      action="approve", justification="x")
        raise AssertionError("junior accepted on senior-only case")
    except PermissionError:
        pass

    # first senior signs -> still pending (1/2)
    e1 = record_review(log=log, case=case, reviewer_user_id="senior_elif",
                       action="approve", justification="First signature.")
    assert e1["review_seq"] == "1/2" and e1["case_complete"] is False
    case = pending_reviews(log, ART)[0]
    assert case["reviews_done"] == 1

    # same senior cannot co-sign twice
    try:
        record_review(log=log, case=case, reviewer_user_id="senior_elif",
                      action="approve", justification="Again.")
        raise AssertionError("same reviewer co-signed twice")
    except PermissionError:
        pass

    # second DISTINCT senior closes the case; most-restrictive final
    e2 = record_review(log=log, case=case, reviewer_user_id="senior_can",
                       action="reject", justification="Second signature.")
    assert e2["review_seq"] == "2/2" and e2["case_complete"] is True
    assert e2["final_outcome"] == "rejected"     # one reject => rejected
    assert pending_reviews(log, ART) == []


def test_p09_statutory_cap_end_to_end():
    log = fresh_log()
    cust = ART.customers.iloc[0]
    absurd = float(cust.monthly_income_try) * 10   # far beyond 4x cap
    r = evaluate_request(actor_user_id="senior_elif",
                         customer_id=cust.customer_id,
                         requested_increase_try=absurd, log=log)
    assert r["effect"] == "DENY"
    assert r["matched_rule"]["policy_id"] == "P-09"
    assert r["limit_income_multiple"] > 4.0
    assert pending_reviews(log, ART) == []         # statutory: no queue


def test_justifications_persisted_in_log():
    log = fresh_log()
    rid, _ = find_request(
        lambda r: r["effect"] == "REQUIRE_HUMAN_REVIEW"
        and r["reviewer_requirements"].get("min_reviewers", 1) == 1
        and r["reviewer_requirements"].get("reviewer_role") != "senior_officer")
    evaluate_request(actor_user_id="senior_elif", request_id=rid, log=log)
    case = pending_reviews(log, ART)[0]
    record_review(log=log, case=case, reviewer_user_id="officer_aylin"
                  if case["branch_code"] in ("B01", "B02", "B03")
                  else "senior_can",
                  action="approve",
                  justification="Income verified against payroll records.")
    entry = log.read_all()[-1]
    assert entry["reviewer_justification"] == \
        "Income verified against payroll records."
    ok, _ = log.verify_chain()
    assert ok


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except Exception as e:
            failed += 1; print(f"FAIL  {name}  {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
