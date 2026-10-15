"""Unit tests validating exception safeguards and false-positive prevention across agents and pipeline."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from agents.remediator import RemediatorAgent
from agents.supervisor import SupervisorAgent
from pipeline.config import PipelineConfig
from metrics.experiment import ExperimentTracker
from pipeline.reversion import generate_manifest_from_run


class TestRemediationSafeguards(unittest.TestCase):
    """Test suite for error-trapping and false-positive elimination."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tracker = ExperimentTracker(base_dir=self.tmp_dir.name)
        self.config = PipelineConfig(batch_size=32, lr=0.01)
        self.remediator = RemediatorAgent("test_remediator", self.tracker, self.config)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_reduce_batch_size_at_minimum_fails_gracefully(self) -> None:
        """Verify that attempting to reduce batch size below 1 fails without false positive."""
        self.config.batch_size = 1
        res = self.remediator.execute_action("reduce_batch_size", self.config)
        self.assertEqual(res["status"], "failed")
        self.assertIn("already at minimum", res["details"])
        self.assertEqual(self.config.batch_size, 1)

    def test_reduce_learning_rate_invalid_fails_gracefully(self) -> None:
        """Verify that attempting to reduce an invalid LR fails safely."""
        self.config.lr = 0.0
        res = self.remediator.execute_action("reduce_learning_rate", self.config)
        self.assertEqual(res["status"], "failed")
        self.assertIn("invalid current lr", res["details"])

    def test_convert_batchnorm_missing_file_fails_cleanly(self) -> None:
        """Verify that a missing model file is caught and marked as failed."""
        self.config.model = "/non/existent/model_xyz.py"
        res = self.remediator.execute_action("convert_batchnorm_to_groupnorm", self.config)
        self.assertEqual(res["status"], "failed")
        self.assertIn("Actuation failed", res["details"])
        # Audit log must record failure, not success
        self.assertEqual(self.remediator.action_log[-1]["status"], "failed")

    def test_convert_batchnorm_zero_replacements_fails(self) -> None:
        """Verify that AST patch synthesis with 0 converted layers is marked failed."""
        # Create a temp model file with no BatchNorm layers
        model_path = Path(self.tmp_dir.name) / "no_bn.py"
        with open(model_path, "w", encoding="utf-8") as f:
            f.write("import torch.nn as nn\nclass M(nn.Module):\n    def __init__(self): super().__init__(); self.fc = nn.Linear(10, 2)\n")

        # Mock synthesizer to target this file
        from unittest.mock import patch
        with patch("agents.plugins.ast_patcher.ASTCodeSynthesizer.patch_file", return_value=("code", "", 0)):
            res = self.remediator.execute_action("convert_batchnorm_to_groupnorm", self.config)
            self.assertEqual(res["status"], "failed")
            self.assertIn("No BatchNorm layers found", res["details"])

    def test_synthesize_code_patch_failure_reported(self) -> None:
        """Verify that AST code patch failure is reported accurately as failed."""
        from unittest.mock import patch
        with patch("agents.plugins.ast_patcher.ASTCodeSynthesizer.synthesize_code_patch", return_value={"status": "error", "error": "parse error", "replacements": 0}):
            res = self.remediator.execute_action("synthesize_code_patch", self.config)
            self.assertEqual(res["status"], "failed")
            self.assertIn("parse error", res["details"])

    def test_supervisor_does_not_mark_auto_remediated_on_failed_action(self) -> None:
        """Verify supervisor sets auto_remediated=False when the action fails to actuate."""
        supervisor = SupervisorAgent("test_supervisor", self.tracker, self.config)
        supervisor.sub_agents["remediator"] = self.remediator

        # Configure remediator to recommend an action that will fail (batch_size=1)
        self.config.batch_size = 1
        self.remediator.recommend_action = MagicMock(return_value="reduce_batch_size")

        supervisor.handle_anomaly("training", ["memory_pressure"], {"batch_size": 1})
        self.assertEqual(len(supervisor.incidents), 1)
        incident = supervisor.incidents[0]
        self.assertFalse(incident["auto_remediated"])
        self.assertIsNone(incident["action_taken"])
        self.assertEqual(incident["remediation_status"], "failed")

    def test_reversion_manifest_ignores_failed_action_entries(self) -> None:
        """Verify that reversion manifest ignores failed actions in action_log.json."""
        run_dir = Path(self.tmp_dir.name) / "run_failed_action_test"
        run_dir.mkdir(parents=True, exist_ok=True)

        import json
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump({"batch_size": 512, "lr": 0.1}, f)

        # Action log has 1 success and 1 failed attempt
        action_log = [
            {
                "action": "reduce_batch_size",
                "status": "success",
                "details": "Reduced batch size from 512 to 256",
                "config_diff": {"batch_size": {"old": 512, "new": 256}},
            },
            {
                "action": "reduce_learning_rate",
                "status": "failed",
                "details": "Actuation failed",
                "config_diff": {"lr": {"old": 0.1, "new": 0.01}},
            },
        ]
        with open(run_dir / "action_log.json", "w", encoding="utf-8") as f:
            json.dump(action_log, f)

        manifest = generate_manifest_from_run(run_dir, save_files=False)
        # Should only rollback batch_size, NOT lr (since lr action failed)
        self.assertIn("batch_size", manifest.rollback_diff)
        self.assertNotIn("lr", manifest.rollback_diff)


if __name__ == "__main__":
    unittest.main()
