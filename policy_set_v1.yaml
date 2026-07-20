#!/usr/bin/env python3
"""
test_policies.py
================
Unit tests for policy_engine.py + policies/policy_set_v1.yaml.
One test per governance rule, plus combining-algorithm, fail-safe and
security tests for the restricted expression evaluator.

Run:  python test_policies.py     (plain asserts; also pytest-compatible)
"""

from policy_engine import (PolicyEngine, PolicyExpressionError,
                           SafeEvaluator, build_context, WHITELIST_FIELDS)

ENGINE = PolicyEngine("policies/policy_set_v1.yaml")

OFFICER = {"user_id": "officer_aylin", "role": "loan_officer",
           "branch_scope": "B01|B02|B03"}
SENIOR = {"user_id": "senior_elif", "role": "senior_officer",
          "branch_scope": "ALL"}
ANALYST = {"user_id": "analyst_kerem", "role": "model_analyst",
           "branch_scope": "NONE"}

CUSTOMER = {"branch_code": "B01", "hardship_flag": False}
REQUEST = {"request_id": "REQ-TEST-0001", "increase_pct": 0.20}
CODES3 = [{"feature": f, "direction": "supports_approval", "text": "t",
           "contribution": 0.1} for f in ("a", "b", "c")]


def ctx(**over):
    base = dict(
        actor=SENIOR, customer=CUSTOMER, request=REQUEST,
        model_info={"version": "1.0.0", "recommendation": "approve",
                    "risk_band": "LOW", "p_afford": 0.9,
                    "feature_names": WHITELIST_FIELDS},
        reason_codes=CODES3, counterfactual=None,
        monitor_dir_min=0.95, fields_accessed=WHITELIST_FIELDS,
    )
    base.update(over)
    return build_context(**base)


# ---------------------------------------------------------------------------
# P-01 access control
# ---------------------------------------------------------------------------
def test_p01_unauthorised_role_denied():
    d = ENGINE.evaluate(ctx(actor=ANALYST))
    assert d.effect == "DENY" and d.matched_rule["rule_id"] == "P-01.1"


def test_p01_out_of_branch_denied():
    d = ENGINE.evaluate(ctx(actor=OFFICER,
                            customer={"branch_code": "B09",
                                      "hardship_flag": False}))
    assert d.effect == "DENY" and d.matched_rule["rule_id"] == "P-01.2"


def test_p01_in_scope_officer_allowed_through():
    d = ENGINE.evaluate(ctx(actor=OFFICER))
    assert d.effect == "ALLOW"          # LOW band, small increase → P-05.5


# ---------------------------------------------------------------------------
# P-02 data minimisation
# ---------------------------------------------------------------------------
def test_p02_non_whitelisted_field_denied():
    d = ENGINE.evaluate(ctx(fields_accessed=WHITELIST_FIELDS + ["gender"]))
    assert d.effect == "DENY" and d.matched_rule["rule_id"] == "P-02.1"
    assert "gender" in d.rationale


# ---------------------------------------------------------------------------
# P-03 barred model features
# ---------------------------------------------------------------------------
def test_p03_barred_feature_in_model_denied():
    d = ENGINE.evaluate(ctx(model_info={
        "version": "0.9.0-legacy", "recommendation": "approve",
        "risk_band": "LOW", "p_afford": 0.9,
        "feature_names": WHITELIST_FIELDS + ["region_code_R5"]}))
    assert d.effect == "DENY" and d.matched_rule["rule_id"] == "P-03.1"


# ---------------------------------------------------------------------------
# P-04 explainability
# ---------------------------------------------------------------------------
def test_p04_too_few_reason_codes_denied():
    d = ENGINE.evaluate(ctx(reason_codes=CODES3[:2]))
    assert d.effect == "DENY" and d.matched_rule["rule_id"] == "P-04.1"


def test_p04_denial_without_counterfactual_reviewed():
    d = ENGINE.evaluate(ctx(
        model_info={"version": "1.0.0", "recommendation": "deny",
                    "risk_band": "MEDIUM", "p_afford": 0.6,
                    "feature_names": WHITELIST_FIELDS},
        counterfactual=None))
    assert d.effect == "REQUIRE_HUMAN_REVIEW"
    assert any(m["rule_id"] == "P-04.2" for m in d.all_matched)


# ---------------------------------------------------------------------------
# P-05 tiered oversight
# ---------------------------------------------------------------------------
def test_p05_hardship_always_reviewed():
    d = ENGINE.evaluate(ctx(customer={"branch_code": "B01",
                                      "hardship_flag": True}))
    assert d.effect == "REQUIRE_HUMAN_REVIEW"
    assert any(m["rule_id"] == "P-05.1" for m in d.all_matched)
    assert d.reviewer_requirements.get("reviewer_role") == "senior_officer"


def test_p05_adverse_recommendation_never_automated():
    d = ENGINE.evaluate(ctx(
        model_info={"version": "1.0.0", "recommendation": "deny",
                    "risk_band": "MEDIUM", "p_afford": 0.55,
                    "feature_names": WHITELIST_FIELDS},
        counterfactual="Strongest obstacle: high utilization"))
    assert d.effect == "REQUIRE_HUMAN_REVIEW"
    assert any(m["rule_id"] == "P-05.2" for m in d.all_matched)


def test_p05_high_risk_two_person_senior_review():
    d = ENGINE.evaluate(ctx(
        request={"request_id": "REQ-TEST-0002", "increase_pct": 0.80},
        model_info={"version": "1.0.0", "recommendation": "approve",
                    "risk_band": "HIGH", "p_afford": 0.7,
                    "feature_names": WHITELIST_FIELDS}))
    assert d.effect == "REQUIRE_HUMAN_REVIEW"
    gov = d.matched_rule
    assert gov["rule_id"] == "P-05.3"
    assert d.reviewer_requirements["min_reviewers"] == 2
    assert d.reviewer_requirements["reviewer_role"] == "senior_officer"


def test_p05_low_band_small_increase_auto_allowed():
    d = ENGINE.evaluate(ctx())
    assert d.effect == "ALLOW" and d.matched_rule["rule_id"] == "P-05.5"
    assert "post_controls" in d.matched_rule


# ---------------------------------------------------------------------------
# P-06 fairness gate suspension
# ---------------------------------------------------------------------------
def test_p06_suspension_downgrades_allow_to_review():
    d = ENGINE.evaluate(ctx(monitor_dir_min=0.70))
    assert d.auto_approval_suspended is True
    assert d.effect == "REQUIRE_HUMAN_REVIEW"
    assert d.matched_rule["rule_id"] == "P-06.1"
    assert "0.700" in d.rationale


def test_p06_suspension_does_not_soften_deny():
    d = ENGINE.evaluate(ctx(monitor_dir_min=0.70, actor=ANALYST))
    assert d.effect == "DENY"           # deny-overrides beats everything


# ---------------------------------------------------------------------------
# P-07 appeals
# ---------------------------------------------------------------------------
def test_p07_appeal_requires_independent_review():
    d = ENGINE.evaluate(ctx(event_type="appeal",
                            original_reviewer_id="officer_aylin"))
    assert d.effect == "REQUIRE_HUMAN_REVIEW"
    assert any(m["rule_id"] == "P-07.1" for m in d.all_matched)
    appeal = next(m for m in d.all_matched if m["rule_id"] == "P-07.1")
    assert appeal["reviewer_constraint"] == "different_from_original"
    assert appeal["sla_hours"] == 48


# ---------------------------------------------------------------------------
# Combining algorithm & fail-safe
# ---------------------------------------------------------------------------
def test_deny_overrides_review_and_allow():
    # analyst (DENY) + hardship (REVIEW) + low band (would ALLOW)
    d = ENGINE.evaluate(ctx(actor=ANALYST,
                            customer={"branch_code": "B01",
                                      "hardship_flag": True}))
    assert d.effect == "DENY"
    assert len(d.all_matched) >= 2      # both rules matched; deny governs


def test_failsafe_default_is_review():
    # MEDIUM band matches P-05.4, so to reach the default we need a
    # context matching nothing: LOW band but increase just above the
    # auto-approve cap and below every review trigger.
    d = ENGINE.evaluate(ctx(
        request={"request_id": "REQ-TEST-0003", "increase_pct": 0.35},
        model_info={"version": "1.0.0", "recommendation": "approve",
                    "risk_band": "LOW", "p_afford": 0.9,
                    "feature_names": WHITELIST_FIELDS}))
    assert d.effect == "REQUIRE_HUMAN_REVIEW"
    assert d.matched_rule is None       # nothing matched → fail-safe
    assert "fail-safe" in d.rationale.lower()


def test_full_trace_recorded_even_for_early_deny():
    d = ENGINE.evaluate(ctx(actor=ANALYST))
    total_rules = sum(len(p.get("rules", [])) for p in ENGINE.policies)
    assert len(d.trace) == total_rules  # ALL rules evaluated → full audit


# ---------------------------------------------------------------------------
# Security: the restricted expression grammar
# ---------------------------------------------------------------------------
def _rejects(expr: str) -> bool:
    try:
        SafeEvaluator({"x": 1}).evaluate(expr)
        return False
    except PolicyExpressionError:
        return True


def test_evaluator_blocks_code_execution():
    assert _rejects("__import__('os').system('rm -rf /')")
    assert _rejects("().__class__.__bases__[0].__subclasses__()")
    assert _rejects("open('/etc/passwd')")
    assert _rejects("exec('print(1)')")
    assert _rejects("x + 1 if True else 2")      # conditional expr forbidden
    assert _rejects("[i for i in range(10)]")    # comprehension forbidden
    assert _rejects("unknown.name == 1")         # unknown context name


def test_evaluator_allows_the_intended_grammar():
    ev = SafeEvaluator({"actor.role": "auditor", "monitor.dir_min": 0.7,
                        "decision.reason_codes": [1, 2, 3]})
    assert ev.evaluate("actor.role not in ['loan_officer', 'admin']")
    assert ev.evaluate("monitor.dir_min < 0.80")
    assert ev.evaluate("len(decision.reason_codes) >= 3")
    assert ev.evaluate("actor.role == 'auditor' and monitor.dir_min < 1")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}  {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
