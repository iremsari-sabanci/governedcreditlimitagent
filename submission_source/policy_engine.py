#!/usr/bin/env python3
"""
policy_engine.py
================
Policy-as-Code enforcement engine for the DA523 project:
"Governed Credit Limit Increase Agent".

Languages / layers:
  * Policies:   YAML (policies/policy_set_v1.yaml) — declarative,
                versioned, reviewable by non-programmers.
  * Conditions: restricted Python-syntax expression grammar, parsed with
                the `ast` module and validated against a NODE WHITELIST.
                Conditions are never eval()'d: a hostile policy file
                cannot execute code, import modules, or reach dunders.
  * Engine:     Python.

Combining algorithm: DENY-OVERRIDES (lineage: Cedar forbid-overrides /
XACML deny-overrides). All rules are always evaluated (the full trace is
audit material); the final effect is the most restrictive matched one:
    DENY > REQUIRE_HUMAN_REVIEW > ALLOW > default (REQUIRE_HUMAN_REVIEW)
P-06's SUSPEND_AUTO_APPROVAL is a system state: while active, any ALLOW
outcome is downgraded to REQUIRE_HUMAN_REVIEW.

Public API (consumed by the UI and audit logger):
    engine = PolicyEngine("policies/policy_set_v1.yaml")
    decision = engine.evaluate(context)      # context: flat dotted dict
    build_context(...)                       # assembles context from
                                             # actor/customer/request/model
Run `python policy_engine.py` for an end-to-end batch demo over the 600
generated requests using the trained production model.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SEVERITY = {"DENY": 3, "REQUIRE_HUMAN_REVIEW": 2, "ALLOW": 1}
SUSPEND = "SUSPEND_AUTO_APPROVAL"
DEFAULT_EFFECT = "REQUIRE_HUMAN_REVIEW"


# ----------------------------------------------------------------------------
# Safe expression evaluator (AST whitelist — the security core)
# ----------------------------------------------------------------------------
class PolicyExpressionError(Exception):
    """Raised when a condition uses forbidden syntax or unknown names."""


_ALLOWED_CMP = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
                ast.In, ast.NotIn)


class SafeEvaluator:
    """Evaluates a single boolean condition against a flat dotted-name
    context, e.g. {"actor.role": "auditor", "model.risk_band": "HIGH"}.

    Grammar (whitelist — everything else raises PolicyExpressionError):
      literals, lists/tuples, dotted names, comparisons (incl. in/not in),
      and/or/not, len(<expr>).
    """

    _KEYWORDS = {"true": True, "false": False, "null": None, "none": None}

    def __init__(self, context: dict[str, Any]):
        self.ctx = context

    def evaluate(self, expression: str) -> bool:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            raise PolicyExpressionError(f"Bad syntax in condition: {e}") from e
        return bool(self._eval(tree.body))

    # -- resolution --------------------------------------------------------
    def _dotted(self, node: ast.AST) -> str:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            raise PolicyExpressionError("Only simple dotted names are allowed.")
        parts.append(node.id)
        return ".".join(reversed(parts))

    def _lookup(self, name: str) -> Any:
        low = name.lower()
        if low in self._KEYWORDS:
            return self._KEYWORDS[low]
        if "__" in name:
            raise PolicyExpressionError(f"Forbidden name: {name}")
        if name not in self.ctx:
            raise PolicyExpressionError(f"Unknown context name: {name}")
        return self.ctx[name]

    # -- recursive evaluation ---------------------------------------------
    def _eval(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (str, int, float, bool, type(None))):
                return node.value
            raise PolicyExpressionError(f"Forbidden literal: {node.value!r}")
        if isinstance(node, ast.Name):
            return self._lookup(node.id)
        if isinstance(node, ast.Attribute):
            return self._lookup(self._dotted(node))
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval(e) for e in node.elts]
        if isinstance(node, ast.BoolOp):
            vals = (self._eval(v) for v in node.values)
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not self._eval(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._eval(node.operand)
        if isinstance(node, ast.Compare):
            left = self._eval(node.left)
            for op, comp in zip(node.ops, node.comparators):
                if not isinstance(op, _ALLOWED_CMP):
                    raise PolicyExpressionError(
                        f"Forbidden operator: {type(op).__name__}")
                right = self._eval(comp)
                ok = {
                    ast.Eq: lambda a, b: a == b,
                    ast.NotEq: lambda a, b: a != b,
                    ast.Lt: lambda a, b: a < b,
                    ast.LtE: lambda a, b: a <= b,
                    ast.Gt: lambda a, b: a > b,
                    ast.GtE: lambda a, b: a >= b,
                    ast.In: lambda a, b: a in b,
                    ast.NotIn: lambda a, b: a not in b,
                }[type(op)](left, right)
                if not ok:
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Name) and node.func.id == "len"
                    and len(node.args) == 1 and not node.keywords):
                return len(self._eval(node.args[0]))
            raise PolicyExpressionError(
                "Only len(<expr>) calls are allowed in conditions.")
        raise PolicyExpressionError(
            f"Forbidden syntax element: {type(node).__name__}")


# ----------------------------------------------------------------------------
# Decision objects
# ----------------------------------------------------------------------------
@dataclass
class RuleEvaluation:
    policy_id: str
    rule_id: str
    condition: str
    matched: bool
    effect_if_matched: str


@dataclass
class Decision:
    effect: str
    matched_rule: dict | None          # the governing (most restrictive) rule
    rationale: str
    all_matched: list[dict]
    trace: list[RuleEvaluation]
    policy_set_version: str
    auto_approval_suspended: bool = False
    reviewer_requirements: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------
class PolicyEngine:
    def __init__(self, policy_path: str | Path):
        raw = yaml.safe_load(Path(policy_path).read_text())
        self.meta = raw["policy_set"]
        self.policies = raw["policies"]
        self.version = self.meta["version"]
        self.default_effect = self.meta.get("default_effect", DEFAULT_EFFECT)
        self.required_log_fields = next(
            (p.get("required_log_fields", []) for p in self.policies
             if p["id"] == "P-08"), [])
        self._validate_policy_syntax()

    def _validate_policy_syntax(self) -> None:
        """Load-time validation: every condition must parse under the
        restricted grammar (against a dummy context lookup we only check
        syntax, not names)."""
        for pol in self.policies:
            for rule in pol.get("rules", []):
                try:
                    ast.parse(rule["when"], mode="eval")
                except SyntaxError as e:
                    raise PolicyExpressionError(
                        f"{pol['id']}/{rule['rule']}: {e}") from e

    @staticmethod
    def _render(template: str, ctx: dict[str, Any]) -> str:
        def sub(m: re.Match) -> str:
            key = m.group(1)
            val = ctx.get(key, f"<{key}?>")
            if isinstance(val, float):
                return f"{val:.3f}"
            return str(val)
        return re.sub(r"\{([A-Za-z_][A-Za-z0-9_.]*)\}", sub, template)

    def evaluate(self, context: dict[str, Any]) -> Decision:
        ev = SafeEvaluator(context)
        trace: list[RuleEvaluation] = []
        matched: list[dict] = []
        suspended = False

        for pol in self.policies:
            for rule in pol.get("rules", []):
                hit = ev.evaluate(rule["when"])
                trace.append(RuleEvaluation(
                    policy_id=pol["id"], rule_id=rule["rule"],
                    condition=rule["when"], matched=hit,
                    effect_if_matched=rule["effect"],
                ))
                if hit:
                    entry = {
                        "policy_id": pol["id"],
                        "policy_name": pol["name"],
                        "rule_id": rule["rule"],
                        "effect": rule["effect"],
                        "rationale": self._render(rule["reason"], context),
                    }
                    for extra in ("reviewer_role", "min_reviewers",
                                  "reviewer_constraint", "sla_hours",
                                  "post_controls"):
                        if extra in rule:
                            entry[extra] = rule[extra]
                    if rule["effect"] == SUSPEND:
                        suspended = True
                    matched.append(entry)

        # ---- deny-overrides combination ---------------------------------
        terminal = [m for m in matched if m["effect"] in SEVERITY]
        if terminal:
            governing = max(terminal, key=lambda m: SEVERITY[m["effect"]])
            effect = governing["effect"]
        else:
            governing = None
            effect = self.default_effect

        rationale = (governing["rationale"] if governing
                     else "No policy rule matched; fail-safe default applies.")

        # ---- P-06 suspension downgrade ----------------------------------
        if suspended and effect == "ALLOW":
            susp = next(m for m in matched if m["effect"] == SUSPEND)
            effect = "REQUIRE_HUMAN_REVIEW"
            rationale = (f"{susp['rationale']} (Auto-approval outcome "
                         f"downgraded to human review.)")
            governing = susp

        # Reviewer requirements: most-restrictive UNION across ALL matched
        # REQUIRE_HUMAN_REVIEW rules (constraints compound — e.g. a
        # high-risk hardship case needs senior + two-person review).
        review_matches = [m for m in matched
                          if m["effect"] == "REQUIRE_HUMAN_REVIEW"]
        reviewer_req: dict = {}
        if effect == "REQUIRE_HUMAN_REVIEW":
            roles = [m.get("reviewer_role") for m in review_matches
                     if m.get("reviewer_role")]
            if roles:
                reviewer_req["reviewer_role"] = (
                    "senior_officer" if "senior_officer" in roles
                    else roles[0])
            mins = [m.get("min_reviewers") for m in review_matches
                    if m.get("min_reviewers")]
            if mins:
                reviewer_req["min_reviewers"] = max(mins)
            constraints = [m.get("reviewer_constraint")
                           for m in review_matches
                           if m.get("reviewer_constraint")]
            if constraints:
                reviewer_req["reviewer_constraint"] = constraints[0]
            slas = [m.get("sla_hours") for m in review_matches
                    if m.get("sla_hours")]
            if slas:
                reviewer_req["sla_hours"] = min(slas)

        return Decision(
            effect=effect,
            matched_rule=governing,
            rationale=rationale,
            all_matched=matched,
            trace=trace,
            policy_set_version=self.version,
            auto_approval_suspended=suspended,
            reviewer_requirements=reviewer_req,
        )


# ----------------------------------------------------------------------------
# Context builder: assembles the flat dotted-name context the policies see
# ----------------------------------------------------------------------------
WHITELIST_FIELDS = [
    "employment_type", "monthly_income_try", "monthly_obligations_try",
    "current_limit_try", "utilization_rate", "months_as_customer",
    "repayment_score", "num_late_payments_12m", "kkb_score_synthetic",
]
BARRED_FEATURES = ["gender", "age", "region_code"]
ALL_BRANCHES = [f"B{i:02d}" for i in range(1, 13)]


def build_context(*, actor: dict, customer: dict, request: dict,
                  model_info: dict, reason_codes: list[dict],
                  counterfactual: str | None, monitor_dir_min: float,
                  fields_accessed: list[str],
                  event_type: str = "evaluation") -> dict[str, Any]:
    scope = (ALL_BRANCHES if actor.get("branch_scope") == "ALL"
             else [] if actor.get("branch_scope") in (None, "NONE")
             else str(actor["branch_scope"]).split("|"))
    non_whitelisted = [f for f in fields_accessed if f not in WHITELIST_FIELDS]
    barred_in_model = [f for f in model_info.get("feature_names", [])
                       if any(f == b or f.startswith(b + "_")
                              for b in BARRED_FEATURES)]
    # P-09 statutory input: total limit after the increase as a multiple
    # of monthly income. Computed defensively (0.0 = not assessable, rule
    # will not fire) so pure-engine tests need not supply the fields.
    try:
        limit_income_multiple = round(
            (float(customer["current_limit_try"])
             + float(request["requested_increase_try"]))
            / float(customer["monthly_income_try"]), 3)
    except (KeyError, TypeError, ZeroDivisionError, ValueError):
        limit_income_multiple = 0.0
    return {
        "actor.user_id": actor["user_id"],
        "actor.role": actor["role"],
        "actor.branch_scope": scope,
        "customer.branch_code": customer["branch_code"],
        "customer.hardship_flag": bool(customer["hardship_flag"]),
        "request.request_id": request["request_id"],
        "request.increase_pct": float(request["increase_pct"]),
        "request.limit_income_multiple": limit_income_multiple,
        "features.non_whitelisted_count": len(non_whitelisted),
        "features.non_whitelisted_list": non_whitelisted,
        "model.version": model_info.get("version", "?"),
        "model.recommendation": model_info["recommendation"],
        "model.risk_band": model_info["risk_band"],
        "model.p_afford": float(model_info["p_afford"]),
        "model.barred_feature_count": len(barred_in_model),
        "model.barred_feature_list": barred_in_model,
        "decision.reason_code_count": len(reason_codes),
        "decision.has_counterfactual": counterfactual is not None,
        "monitor.dir_min": float(monitor_dir_min),
        "event.type": event_type,
    }


# ----------------------------------------------------------------------------
# Integration demo: run the engine over all 600 requests with the real model
# ----------------------------------------------------------------------------
def _demo() -> None:  # pragma: no cover
    import joblib
    import pandas as pd
    from collections import Counter
    from train_model import build_features, risk_band, reason_codes as rc

    customers = pd.read_csv("data/customers.csv")
    requests = pd.read_csv("data/requests.csv")
    users = pd.read_csv("data/users.csv")
    bundle = joblib.load("models/model_v1_0.joblib")
    model, cols = bundle["model"], bundle["feature_columns"]
    medians = bundle["medians"]

    engine = PolicyEngine("policies/policy_set_v1.yaml")

    # fairness monitor input: production model DIR by gender (from card)
    import json
    card = json.loads(Path("models/model_card.json").read_text())
    dir_min = card["metrics"]["dir_gender_prod"]

    req = requests.merge(customers, on="customer_id", how="left")
    X = build_features(req).reindex(columns=cols, fill_value=0)
    p_all = model.predict_proba(X)[:, 1]

    senior = users[users.role == "senior_officer"].iloc[0].to_dict()
    mix, sample_shown = Counter(), False
    for i in range(len(req)):
        row = req.iloc[i]
        p = float(p_all[i])
        band = risk_band(p, float(row["increase_pct"]))
        recommendation = "approve" if p >= 0.5 else "deny"
        codes = rc(model, X.iloc[[i]], medians)
        counterfactual = (None if recommendation == "approve" else
                          "Strongest obstacle: " +
                          next((c["text"] for c in codes
                                if c["direction"] == "against_approval"),
                               "n/a"))
        ctx = build_context(
            actor=senior, customer=row.to_dict(), request=row.to_dict(),
            model_info={"version": bundle["version"], "recommendation":
                        recommendation, "risk_band": band, "p_afford": p,
                        "feature_names": cols},
            reason_codes=codes, counterfactual=counterfactual,
            monitor_dir_min=dir_min,
            fields_accessed=WHITELIST_FIELDS,
        )
        d = engine.evaluate(ctx)
        mix[d.effect] += 1
        if not sample_shown and d.effect == "REQUIRE_HUMAN_REVIEW":
            print(f"\nSample decision — {row['request_id']}:")
            print(f"  effect          : {d.effect}")
            print(f"  matched rule    : {d.matched_rule['policy_id']}/"
                  f"{d.matched_rule['rule_id']} "
                  f"({d.matched_rule['policy_name']})")
            print(f"  rationale       : {d.rationale}")
            print(f"  reviewer req    : {d.reviewer_requirements}")
            print(f"  rules evaluated : {len(d.trace)}  "
                  f"(matched: {len(d.all_matched)})")
            sample_shown = True

    print(f"\nDecision mix over {len(req)} requests "
          f"(policy set v{engine.version}, monitor DIR={dir_min}):")
    for effect in ("ALLOW", "REQUIRE_HUMAN_REVIEW", "DENY"):
        print(f"  {effect:<22} {mix.get(effect, 0):>4}")


if __name__ == "__main__":
    _demo()
