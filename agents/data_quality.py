"""Data quality agent module for AutoPARK.

Monitors dataset partitions, schema conformance, sample decodability,
and label distributions across Apache Arrow / Parquet data pipelines.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from agents.base import AgentBase


class DataQualityAgent(AgentBase):
    """AutoPARK Data Quality & Integrity Agent.

    Inspects streaming dataset validation telemetry, detecting schema mismatches,
    corrupted parquet partitions or images, severe class imbalance, and missing columns.
    Enables Claude to triage data-layer defects prior to expensive GPU/CPU training stages.
    """

    def observe(self, event: dict[str, Any]) -> list[str]:
        anomalies: list[str] = []
        if "data_quality" not in event:
            return anomalies

        dq = event["data_quality"]
        if dq.get("schema_mismatch"):
            anomalies.append("schema_error")
        if dq.get("corrupt_images", 0) > 0:
            anomalies.append("corrupt_data")
        if dq.get("label_imbalance"):
            anomalies.append("label_imbalance")
        if dq.get("missing_columns"):
            anomalies.append("missing_columns")

        return anomalies

    def validate_parquet(self, path: Path | str) -> tuple[bool, list[str]]:
        """Opens parquet file with pyarrow and checks schema integrity."""
        issues = []
        path_obj = Path(path)
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(path_obj)
            columns = table.column_names
            if "img" not in columns:
                issues.append("missing 'img' column")
            if "label" not in columns:
                issues.append("missing 'label' column")

            if not issues:
                return True, []
            return False, issues
        except ImportError:
            self._log("pyarrow not installed, skipping parquet validation.", level="warning")
            return False, ["validation_skipped"]
        except Exception as e:
            self._log(f"Failed to validate parquet {path_obj}: {e}", level="error")
            return False, [f"read_error: {e}"]

    def diagnose(self, anomalies: list[str], context: dict[str, Any]) -> str:
        """Diagnose dataset anomalies directly using Claude Platform on AWS."""
        if not anomalies:
            return ""

        system_prompt = (
            "You are an ML Data Quality Diagnostic Agent on Claude Platform on AWS. Analyze the dataset anomalies "
            "and explain data pipeline risks in 1 concise sentence."
        )
        user_prompt = f"Anomalies: {anomalies}. Context: {context}."
        return self._call_llm(system_prompt, user_prompt, max_tokens=150)

    def recommend_action(self, diagnosis: str, context: dict[str, Any]) -> str | None:
        if "schema_error" in diagnosis or "missing_columns" in diagnosis:
            return "halt_pipeline"
        if "corrupt_data" in diagnosis:
            return "filter_corrupt_data"
        if "label_imbalance" in diagnosis:
            return "enable_class_weights"
        return None
