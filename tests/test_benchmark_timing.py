"""Unit tests for measured MTTR decomposition and outcome classification in the benchmark harness."""
from __future__ import annotations

import unittest

from benchmarks.harness import TIMING_KEYS, classify_outcome, compute_incident_timing


def _attempt(start, end, status, attempt=0, stage="training"):
    return {"type": "attempt", "stage": stage, "attempt": attempt, "start": start, "end": end,
            "duration_s": end - start, "status": status}


def _handle(start, end, attempt=0, stage="training"):
    return {"type": "handle", "stage": stage, "attempt": attempt, "start": start, "end": end,
            "duration_s": end - start}


def _llm(start, end, caller="diagnose"):
    return {"agent": "a", "caller": caller, "start": start, "end": end, "duration_s": end - start,
            "input_tokens": 100, "output_tokens": 20}


class TestIncidentTiming(unittest.TestCase):
    def test_recovered_run_components_sum_to_mttr(self) -> None:
        timeline = [
            _attempt(0.0, 0.0, "success", stage="data_validation"),
            _attempt(10.0, 10.5, "failed"),
            _handle(10.5, 14.5),
            _attempt(14.6, 30.0, "success", attempt=1),
        ]
        llm = [_llm(10.6, 12.0), _llm(12.1, 14.0, caller="recommend_action"), _llm(20.0, 21.0, caller="analyze_epoch")]
        actions = [{"action": "reduce_batch_size", "status": "success", "start": 14.1, "duration_s": 0.01}]
        steps = [{"event": "injected_oom", "timestamp": 10.4}, {"event": "step", "timestamp": 15.6}]

        t = compute_incident_timing(timeline, llm, actions, steps)
        self.assertTrue(t["recovered"])
        self.assertAlmostEqual(t["mttr_s"], 15.6 - 10.0)
        self.assertAlmostEqual(t["t_det"], 0.5)
        self.assertAlmostEqual(t["t_reason"], 1.4 + 1.9)
        self.assertAlmostEqual(t["t_act"], 0.01)
        self.assertAlmostEqual(t["t_recover"], 1.0)
        self.assertAlmostEqual(sum(t[k] for k in TIMING_KEYS), t["mttr_s"])
        # The in-training analyze_epoch call is monitoring cost, not incident reasoning
        self.assertAlmostEqual(t["monitor_llm_s"], 1.0)
        self.assertEqual(t["llm_calls"], 3)

    def test_unrecovered_run_measures_time_to_termination(self) -> None:
        timeline = [_attempt(0.0, 1.0, "failed"), _handle(1.0, 3.0), _attempt(3.0, 4.0, "failed", 1), _handle(4.0, 6.0, 1)]
        t = compute_incident_timing(timeline, [], [], [])
        self.assertFalse(t["recovered"])
        self.assertAlmostEqual(t["mttr_s"], 6.0)
        self.assertAlmostEqual(t["t_det"], 2.0)
        self.assertEqual(t["t_recover"], 0.0)
        self.assertAlmostEqual(sum(t[k] for k in TIMING_KEYS), 6.0)

    def test_clean_run_has_no_incident(self) -> None:
        t = compute_incident_timing([_attempt(0.0, 5.0, "success")], [], [], [])
        self.assertFalse(t["incident"])
        self.assertNotIn("mttr_s", t)


class TestClassifyOutcome(unittest.TestCase):
    def test_outcomes(self) -> None:
        ok = [{"status": "success"}]
        self.assertEqual(classify_outcome("success", [], []), "passed")
        self.assertEqual(classify_outcome("success", [{"auto_remediated": True}], ok), "healed")
        self.assertEqual(classify_outcome("success", [{"auto_remediated": False}], []), "recovered")
        self.assertEqual(classify_outcome("failed", [{"auto_remediated": True}, {"auto_remediated": False}], ok), "escalated")
        self.assertEqual(classify_outcome("failed", [{"auto_remediated": True}], ok), "exhausted")
        self.assertEqual(classify_outcome("failed", [], []), "failed")


if __name__ == "__main__":
    unittest.main()
