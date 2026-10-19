from __future__ import annotations

import tempfile
import unittest
import hashlib
import json
from unittest.mock import patch
from pathlib import Path

from dwr.experiments.source_pilot_continuation import load_parent_imports, validate_launch_window


class SourcePilotContinuationTests(unittest.TestCase):
    def _registration(self, parent: str = "runs/parent") -> dict:
        parent_source_identity = hashlib.sha256(b"a" * 40).hexdigest()
        return {"continuation": {
            "parent_matrix": parent,
            "parent_source_commit": "a" * 40,
            "parent_config_sha256": "b" * 64,
            "original_absolute_deadline_epoch": 1792445742.30148,
            "launch_not_after_epoch": 1792440342.30148,
            "imports": {arm: {"arm_audit_sha256": "c" * 64, "manifest_sha256": "d" * 64,
                "physical_audit_sha256": "e" * 64, "source_identity": parent_source_identity}
                for arm in ("K0", "K1")},
        }}

    def test_launch_cutoff_is_strict_for_creation(self):
        registration = self._registration()
        validate_launch_window(registration, now_epoch=1792440342.30148)
        with self.assertRaisesRegex(ValueError, "past its registered launch cutoff"):
            validate_launch_window(registration, now_epoch=1792440342.30149)

    def test_missing_parent_import_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "parent matrix is missing"):
                load_parent_imports(tmp, self._registration())

    def test_import_schema_rejects_tampered_or_missing_receipt_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "runs", "parent").mkdir(parents=True)
            registration = self._registration()
            del registration["continuation"]["imports"]["K1"]["physical_audit_sha256"]
            with self.assertRaisesRegex(ValueError, "K1 import receipt has unknown or missing fields"):
                load_parent_imports(tmp, registration)

    def test_parent_audit_content_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            registration = self._registration()
            parent = Path(tmp, "runs", "parent")
            parent.mkdir(parents=True)
            (parent / "manifest.initial.json").write_text(json.dumps({
                "source": {"commit": "a" * 40}, "config": {"sha256": "b" * 64}}), encoding="utf-8")
            for arm in ("K0", "K1"):
                arm_root = parent / "arms" / arm
                arm_root.mkdir(parents=True)
                audit = {"arm_id": arm, "identities": {"source": registration["continuation"]["imports"][arm]["source_identity"]}}
                audit_raw = json.dumps(audit).encode("utf-8")
                physical_raw = b'{"physical":"receipt"}'
                manifest_raw = b'{"manifest":"terminal"}'
                (arm_root / "arm-audit.json").write_bytes(audit_raw)
                (arm_root / "physical-audit.json").write_bytes(physical_raw)
                (arm_root / "manifest.json").write_bytes(manifest_raw)
                receipt = registration["continuation"]["imports"][arm]
                receipt["arm_audit_sha256"] = hashlib.sha256(audit_raw).hexdigest()
                receipt["physical_audit_sha256"] = hashlib.sha256(physical_raw).hexdigest()
                receipt["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
            with patch("dwr.experiments.records.verify_artifacts", return_value={"status": "completed"}):
                self.assertEqual(set(load_parent_imports(tmp, registration)[0]), {"K0", "K1"})
                (parent / "arms" / "K1" / "arm-audit.json").write_bytes(b'{"arm_id":"K1","tampered":true}')
                with self.assertRaisesRegex(ValueError, "K1 audit digest mismatch"):
                    load_parent_imports(tmp, registration)

    def test_mixed_source_identity_is_checked_without_changing_pairing(self):
        from tests.test_source_pilot_controller import _matrix_config, _passing_arms
        from dwr.experiments.source_pilot import matrix_decision

        arms = _passing_arms()
        arms["K0"]["identities"]["source"] = "a" * 64
        arms["K1"]["identities"]["source"] = "b" * 64
        arms["K4"]["identities"]["source"] = "c" * 64
        identities = {arm: arms[arm]["identities"]["source"] for arm in ("K0", "K1", "K4")}
        result = matrix_decision(_matrix_config(steps=512), arms, source_identities=identities)
        self.assertEqual(result["status"], "PASS_SOURCE_VALIDATION_PILOT")
        self.assertEqual(arms["K0"]["pairing"], arms["K1"]["pairing"])
        self.assertEqual(arms["K0"]["pairing"], arms["K4"]["pairing"])
        identities["K0"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "K0 audited source identity differs"):
            matrix_decision(_matrix_config(steps=512), arms, source_identities=identities)


if __name__ == "__main__":
    unittest.main()
