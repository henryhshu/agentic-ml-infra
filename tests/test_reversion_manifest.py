"""Unit tests for AutoPARK Reversion & Rollback Manifest engine."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.reversion import (
    ReversionManifest,
    apply_rollback,
    format_reversion_markdown,
    format_reversion_terminal,
    generate_manifest_from_run,
    resolve_run_dir,
)


class TestReversionManifest(unittest.TestCase):
    """Test suite validating rollback manifest synthesis and configuration restoration."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.run_id = "run_test_reversion_001"
        self.run_dir = self.temp_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Mock initial config.json
        self.initial_config = {
            "model": "resnet18",
            "batch_size": 512,
            "accum_steps": 1,
            "lr": 0.1,
            "max_grad_norm": None,
        }
        with open(self.run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.initial_config, f, indent=2)

        # 2. Mock incidents.json
        self.incidents = [
            {
                "stage": "training",
                "anomalies": ["memory_pressure"],
                "raw_anomalies": ["CUDA out of memory for batch_size=512"],
                "diagnosis": "OOM due to oversized batch 512 on GPU. Recommend halving batch size.",
                "auto_remediated": True,
                "action_taken": "reduce_batch_size",
            },
            {
                "stage": "training",
                "anomalies": ["memory_pressure"],
                "raw_anomalies": ["CUDA out of memory for batch_size=256"],
                "diagnosis": "Memory pressure persists at 256. Recommend halving to 128.",
                "auto_remediated": True,
                "action_taken": "reduce_batch_size",
            },
        ]
        with open(self.run_dir / "incidents.json", "w", encoding="utf-8") as f:
            json.dump(self.incidents, f, indent=2)

        # 3. Mock action_log.json (Chained multi-hop mutations: 512 -> 256 -> 128)
        self.actions = [
            {
                "timestamp": "2026-10-13T01:00:00Z",
                "action": "reduce_batch_size",
                "status": "success",
                "details": "Reduced batch size from 512 to 256 and scaled accum_steps 1 -> 2",
                "config_diff": {
                    "batch_size": {"old": 512, "new": 256},
                    "accum_steps": {"old": 1, "new": 2},
                },
            },
            {
                "timestamp": "2026-10-13T01:05:00Z",
                "action": "reduce_batch_size",
                "status": "success",
                "details": "Reduced batch size from 256 to 128 and scaled accum_steps 2 -> 4",
                "config_diff": {
                    "batch_size": {"old": 256, "new": 128},
                    "accum_steps": {"old": 2, "new": 4},
                },
            },
        ]
        with open(self.run_dir / "action_log.json", "w", encoding="utf-8") as f:
            json.dump(self.actions, f, indent=2)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_manifest_multi_hop_reconstruction(self) -> None:
        """Verify multi-hop chained mutations reconstruct original initial pre-fix values."""
        manifest = generate_manifest_from_run(self.run_id, base_dir=self.temp_dir, save_files=True)

        self.assertEqual(manifest.run_id, self.run_id)
        # Final values after mutations
        self.assertEqual(manifest.final_config["batch_size"], 128)
        self.assertEqual(manifest.final_config["accum_steps"], 4)

        # Rollback diff must map back to earliest known initial value (512 and 1)
        self.assertEqual(manifest.rollback_diff["batch_size"], 512)
        self.assertEqual(manifest.rollback_diff["accum_steps"], 1)

        # Verify generated files
        self.assertTrue((self.run_dir / "reversion_manifest.json").exists())
        self.assertTrue((self.run_dir / "reversion_manifest.md").exists())

    def test_format_reversion_summaries(self) -> None:
        """Verify markdown and terminal formatting contain all 4 essential pillars."""
        manifest = generate_manifest_from_run(self.run_id, base_dir=self.temp_dir, save_files=False)

        md = format_reversion_markdown(manifest)
        self.assertIn("## 🔴 1. Issues Detected", md)
        self.assertIn("## 🛠️ 2. What Was Changed", md)
        self.assertIn("## 🧠 3. Why It Was Changed", md)
        self.assertIn("## 🔄 4. How To Change It Back", md)
        self.assertIn("| `batch_size` | `128` | **`512`** |", md)
        self.assertIn(f"--rollback-run {self.run_id}", md)

        term = format_reversion_terminal(manifest)
        self.assertIn("ISSUES DETECTED", term)
        self.assertIn("WHAT WAS CHANGED", term)
        self.assertIn("WHY IT WAS CHANGED", term)
        self.assertIn("HOW TO CHANGE IT BACK", term)
        self.assertIn("batch_size: 512", term)

    def test_apply_rollback_config_mutation(self) -> None:
        """Verify apply_rollback reverses parameters on a live PipelineConfig object."""
        # Simulated mutated config at the end of a run
        mutated_config = PipelineConfig(
            model="resnet18",
            batch_size=128,
            accum_steps=4,
        )

        # Apply rollback using the manifest from the run
        restored = apply_rollback(self.run_id, config=mutated_config, base_dir=self.temp_dir)

        # Confirms parameters are restored back to pre-intervention states
        self.assertEqual(restored.batch_size, 512)
        self.assertEqual(restored.accum_steps, 1)

    def test_resolve_run_dir_helpers(self) -> None:
        """Verify run directory resolution via ID or direct path."""
        p1 = resolve_run_dir(self.run_id, base_dir=self.temp_dir)
        self.assertEqual(p1, self.run_dir.resolve())

        p2 = resolve_run_dir(self.run_dir)
        self.assertEqual(p2, self.run_dir.resolve())

        with self.assertRaises(FileNotFoundError):
            resolve_run_dir("nonexistent_run_id", base_dir=self.temp_dir)


if __name__ == "__main__":
    unittest.main()
