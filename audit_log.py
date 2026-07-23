#!/usr/bin/env python3
"""
audit_log.py
============
Hash-chained, append-only audit log for the DA523 project:
"Governed Credit Limit Increase Agent".

Implements policy P-08 (audit_and_versioning):
  * Every decision event is appended as one JSONL line.
  * Each entry embeds the SHA-256 of the previous entry (prev_hash) and
    its own SHA-256 (entry_hash) -> tampering with ANY historical line
    breaks the chain from that point on and is detectable.
  * The logger ENFORCES the P-08 contract: it refuses to append an entry
    that is missing any required_log_fields declared in the policy set
    (fields the logger itself generates are exempt from the caller).

Public API:
    log = AuditLog("audit/audit_log.jsonl",
                   policy_path="policies/policy_set_v1.yaml")
    log.append(entry_dict)          # raises AuditContractError if fields missing
    log.read_all() -> list[dict]
    log.verify_chain() -> (ok: bool, first_bad_index: int | None)
    demo_tamper_detection(log)      # notebook demo: corrupt one line, prove detection

Run `python audit_log.py` for a self-test incl. the tamper demonstration.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

GENESIS_HASH = "0" * 64

# Fields the logger generates itself; callers must supply everything else
# named in P-08's required_log_fields.
AUTO_FIELDS = {"event_id", "timestamp", "prev_hash", "entry_hash"}


class AuditContractError(Exception):
    """Raised when an entry violates the P-08 logging contract."""


def _entry_hash(entry: dict) -> str:
    """Deterministic hash over the entry EXCLUDING entry_hash itself."""
    payload = {k: v for k, v in entry.items() if k != "entry_hash"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path,
                 policy_path: str | Path = "policies/policy_set_v1.yaml"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = yaml.safe_load(Path(policy_path).read_text())
        p08 = next(p for p in raw["policies"] if p["id"] == "P-08")
        self.required_fields = set(p08["required_log_fields"])
        self.policy_set_version = raw["policy_set"]["version"]

    # ------------------------------------------------------------------ append
    def _last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return GENESIS_HASH
        last_line = self.path.read_text().rstrip("\n").rsplit("\n", 1)[-1]
        return json.loads(last_line)["entry_hash"]

    def append(self, entry: dict) -> dict:
        """Append one event. Caller supplies the business fields; the
        logger adds event_id, timestamp, prev_hash and entry_hash, then
        enforces the P-08 contract before writing."""
        entry = dict(entry)  # defensive copy
        entry.setdefault("event_id", f"EVT-{uuid.uuid4().hex[:12]}")
        entry.setdefault(
            "timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        entry.setdefault("policy_set_version", self.policy_set_version)
        entry["prev_hash"] = self._last_hash()
        entry["entry_hash"] = _entry_hash(entry)

        missing = self.required_fields - set(entry.keys())
        if missing:
            raise AuditContractError(
                f"P-08 violation: entry missing required fields {sorted(missing)}")

        with self.path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return entry

    # -------------------------------------------------------------------- read
    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line)
                for line in self.path.read_text().splitlines() if line.strip()]

    # ------------------------------------------------------------------ verify
    def verify_chain(self) -> tuple[bool, int | None]:
        """Recompute every hash and every link. Returns (True, None) if
        the log is intact, else (False, index_of_first_bad_entry)."""
        prev = GENESIS_HASH
        for i, entry in enumerate(self.read_all()):
            if entry.get("prev_hash") != prev:
                return False, i
            if entry.get("entry_hash") != _entry_hash(entry):
                return False, i
            prev = entry["entry_hash"]
        return True, None


# ----------------------------------------------------------------------------
# Notebook demo: prove tampering is detectable
# ----------------------------------------------------------------------------
def demo_tamper_detection(log: AuditLog, line_index: int = 1,
                          field: str = "effect",
                          forged_value: str = "ALLOW") -> None:
    """Corrupt one historical entry in place (e.g. rewrite a DENY as an
    ALLOW), show verification failing at exactly that line, then restore
    the original file."""
    original = log.path.read_text()
    lines = original.splitlines()
    if line_index >= len(lines):
        print("Log too short for the demo."); return

    ok, _ = log.verify_chain()
    print(f"[1] Chain intact before tampering : {ok}")

    entry = json.loads(lines[line_index])
    old = entry.get(field)
    entry[field] = forged_value           # forged, hashes NOT recomputed
    lines[line_index] = json.dumps(entry, default=str)
    log.path.write_text("\n".join(lines) + "\n")
    print(f"[2] Forged entry #{line_index}: {field} '{old}' -> '{forged_value}'")

    ok, bad = log.verify_chain()
    print(f"[3] Chain intact after tampering  : {ok} "
          f"(first broken entry index: {bad})")
    assert not ok and bad == line_index, "Tamper detection failed!"

    log.path.write_text(original)
    ok, _ = log.verify_chain()
    print(f"[4] Original restored; chain intact again: {ok}")


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
def _self_test() -> None:  # pragma: no cover
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "audit_log.jsonl"
    log = AuditLog(tmp)

    base = {
        "actor": "senior_elif", "customer_pseudo_id": "CUST-000001",
        "request_id": "REQ-2026-0001",
        "fields_accessed": ["monthly_income_try"],
        "model_version": "1.0.0",
        "policies_evaluated": 13,
        "matched_rule": "P-05.5", "effect": "ALLOW",
        "reason_codes": ["income+", "utilization+", "kkb+"],
    }
    for i in range(4):
        log.append({**base, "request_id": f"REQ-2026-{i:04d}",
                    "effect": "ALLOW" if i % 2 else "REQUIRE_HUMAN_REVIEW"})
    ok, _ = log.verify_chain()
    assert ok
    print(f"appended 4 entries; chain intact: {ok}")

    try:
        log.append({"actor": "x"})  # missing nearly everything
        raise AssertionError("contract enforcement failed")
    except AuditContractError as e:
        print(f"contract enforced: {e}")

    demo_tamper_detection(log, line_index=2)
    print("audit_log self-test passed")


if __name__ == "__main__":
    _self_test()
