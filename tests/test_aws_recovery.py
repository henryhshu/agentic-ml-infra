"""Exercise the recovery CLI without creating cloud infrastructure."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.calls = self.base / "calls"
        stub = '''#!/usr/bin/env bash
printf '%s %s\\n' "$(basename "$0")" "$*" >> "$CALLS"
case "$*" in
  "sts get-caller-identity"*) echo "${TEST_ACCOUNT:-515966515749}" ;;
  "s3api list-buckets"*) echo "" ;;
  "create cluster"*) exit "${TEST_CLUSTER_EXIT:-0}" ;;
esac
'''
        for tool in ("aws", "eksctl", "kubectl"):
            p = self.base / tool
            p.write_text(stub)
            p.chmod(0o755)
        self.env = {**os.environ, "PATH": f"{self.base}:{os.environ['PATH']}",
                    "CALLS": str(self.calls), "KUBECONFIG": str(self.base / "config")}
        self.root = Path(__file__).resolve().parents[1]

    def run_recovery(self, *args, **env):
        return subprocess.run(["bash", "eks/recover.sh", *args], cwd=self.root,
                              env={**self.env, **env}, capture_output=True, text=True)

    def test_plan_makes_no_external_calls(self):
        result = self.run_recovery("plan")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("eksctl create cluster", result.stdout)
        self.assertFalse(self.calls.exists())

    def test_up_creates_storage_before_cluster_and_checks_nodes(self):
        result = self.run_recovery("up", "--apply")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls.read_text()
        self.assertLess(calls.index("s3api create-bucket"), calls.index("eksctl create cluster"))
        self.assertLess(calls.index("eksctl create cluster"), calls.index("eks update-kubeconfig"))
        self.assertIn("kubectl get nodes", calls)

    def test_wrong_account_stops_before_creation(self):
        result = self.run_recovery("up", "--apply", TEST_ACCOUNT="000000000000")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("create-bucket", self.calls.read_text())

    def test_failed_cluster_creation_stops_followup_commands(self):
        result = self.run_recovery("up", "--apply", TEST_CLUSTER_EXIT="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("update-kubeconfig", self.calls.read_text())

    def test_missing_backup_does_not_create_storage(self):
        result = self.run_recovery("restore-results", "--apply", BACKUP_DIR=str(self.base / "missing"))
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("create-bucket", self.calls.read_text())


if __name__ == "__main__":
    unittest.main()
