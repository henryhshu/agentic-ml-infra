"""Data validation stage — verifies parquet files, schema, row counts, and sample decoding."""
from __future__ import annotations

import io
import time
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from metrics.experiment import ExperimentTracker
from metrics.schemas import DataValidationMetrics, now_iso


class DataValidationStage:
    """Pipeline stage that validates CIFAR-10 parquet data before training."""

    STAGE_NAME = "data_validation"
    EXPECTED_COLUMNS = {"img", "label"}
    EXPECTED_TRAIN_ROWS = 50_000
    EXPECTED_TEST_ROWS = 10_000

    def __init__(
        self,
        config,  # PipelineConfig
        tracker: ExperimentTracker,
        checkpoint_emitter=None,
        supervisor=None,
    ):
        self.config = config
        self.tracker = tracker
        self.checkpoint_emitter = checkpoint_emitter
        self.supervisor = supervisor

    @property
    def name(self) -> str:
        return self.STAGE_NAME

    def run(self) -> DataValidationMetrics:
        """Run data validation and return metrics."""
        metrics = self.tracker.start_stage(
            self.STAGE_NAME, metrics_cls=DataValidationMetrics
        )
        self._emit_checkpoint("stage_start", {"stage": self.STAGE_NAME})

        try:
            result = self.execute()
            # Update metrics with results
            metrics.num_train_samples = result["num_train_samples"]
            metrics.num_test_samples = result["num_test_samples"]
            metrics.schema_valid = result["schema_valid"]
            metrics.sample_decode_ok = result["sample_decode_ok"]
            metrics.data_quality_score = result["data_quality_score"]
            metrics.metrics.update(result)

            if not result["schema_valid"] or not result["sample_decode_ok"]:
                self.tracker.end_stage(self.STAGE_NAME, status="failed")
                metrics.error_message = "; ".join(result.get("issues", []))
                self.tracker.save_stage(self.STAGE_NAME)
                self._emit_checkpoint("anomaly_detected", {"issues": result.get("issues", [])})

                if self.supervisor is not None:
                    self.supervisor.route_event({
                        "type": "data_event",
                        "stage": self.STAGE_NAME,
                        "data_quality": {
                            "schema_mismatch": not result["schema_valid"],
                            "corrupt_images": 0 if result["sample_decode_ok"] else 1,
                            "label_imbalance": False,
                            "missing_columns": not result["schema_valid"],
                        },
                        "result": result,
                    })

                print(f"  ❌ Data validation checks failed: {metrics.error_message}")
                raise ValueError(f"Data validation failed: {metrics.error_message}")
            else:
                self.tracker.end_stage(self.STAGE_NAME, status="success")

            self.tracker.save_stage(self.STAGE_NAME)
            self._emit_checkpoint("stage_end", result)

            if self.supervisor is not None:
                self.supervisor.route_event({
                    "type": "data_event",
                    "stage": self.STAGE_NAME,
                    "data_quality": {
                        "schema_mismatch": not result["schema_valid"],
                        "corrupt_images": 0 if result["sample_decode_ok"] else 1,
                        "label_imbalance": False,
                        "missing_columns": not result["schema_valid"],
                    },
                    "result": result,
                })

            print(f"  ✅ Data validation: {result['data_quality_score']:.1%} quality score")
            return metrics

        except Exception as exc:
            self.tracker.log_event(
                self.STAGE_NAME, "error", f"Data validation failed: {exc}"
            )
            self.tracker.end_stage(self.STAGE_NAME, status="failed")
            metrics.error_message = str(exc)
            self.tracker.save_stage(self.STAGE_NAME)
            self._emit_checkpoint("anomaly_detected", {"error": str(exc)})
            raise

    def execute(self) -> dict:
        """Validate parquet data files or custom DataModule and return quality metrics."""
        issues: list[str] = []

        # Check if non-CIFAR DataModule or custom dataset is active
        if getattr(self.config, "data_module", None) is not None or getattr(self.config, "dataset", "cifar10").lower() != "cifar10":
            from pipeline.data_module import create_data_module
            dm = create_data_module(self.config)
            dm.setup("validate")
            schema_ok, dm_issues = dm.validate_schema()
            issues.extend(dm_issues)
            num_train = dm.train_samples
            num_test = dm.val_samples
            quality_score = 1.0 if (schema_ok and len(issues) == 0) else max(0.0, 1.0 - len(issues) * 0.2)
            self.tracker.log_metric(self.STAGE_NAME, "num_train_samples", num_train)
            self.tracker.log_metric(self.STAGE_NAME, "num_test_samples", num_test)
            self.tracker.log_metric(self.STAGE_NAME, "data_quality_score", quality_score)
            return {
                "num_train_samples": num_train,
                "num_test_samples": num_test,
                "schema_valid": schema_ok,
                "sample_decode_ok": schema_ok,
                "data_quality_score": quality_score,
                "issues": issues,
            }

        from datasets.cifar10 import download_split
        cache_dir = Path(self.config.cache_dir)

        # Download and locate data files
        self.tracker.log_event(
            self.STAGE_NAME, "info", "Downloading/locating parquet shards..."
        )
        train_path = download_split("train", cache_dir)
        test_path = download_split("test", cache_dir)

        # Validate schema
        schema_valid = True

        # ── Fault injection: Corrupted schema / parquet partition ──
        if self.config.fault_injection in ("corrupt_data", "schema_error"):
            schema_valid = False
            issues.append("train: missing columns {'img'} (simulated schema error)")
            issues.append("train: corrupt parquet bytes in shard partition 0")

        for split_name, path in [("train", train_path), ("test", test_path)]:
            table = pq.read_table(str(path))
            columns = set(table.column_names)
            if not self.EXPECTED_COLUMNS.issubset(columns):
                missing = self.EXPECTED_COLUMNS - columns
                issues.append(f"{split_name}: missing columns {missing}")
                schema_valid = False
            self.tracker.log_event(
                self.STAGE_NAME,
                "schema_check",
                f"{split_name} columns: {sorted(columns)}",
            )

        # Validate row counts
        train_table = pq.read_table(str(train_path))
        test_table = pq.read_table(str(test_path))
        num_train = len(train_table)
        num_test = len(test_table)

        if self.config.subset is not None:
            # When using subset mode, just verify we have enough rows
            if num_train < self.config.subset:
                issues.append(
                    f"Train set has {num_train} rows, need at least {self.config.subset}"
                )
        else:
            if num_train != self.EXPECTED_TRAIN_ROWS:
                issues.append(
                    f"Train row count {num_train} != expected {self.EXPECTED_TRAIN_ROWS}"
                )
            if num_test != self.EXPECTED_TEST_ROWS:
                issues.append(
                    f"Test row count {num_test} != expected {self.EXPECTED_TEST_ROWS}"
                )

        # Validate sample image decoding
        sample_decode_ok = True
        num_samples_to_check = min(10, num_train)
        decode_failures = 0
        for i in range(num_samples_to_check):
            try:
                img_bytes = train_table.column("img").to_pylist()[i]["bytes"]
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                if img.size != (32, 32):
                    issues.append(f"Sample {i}: unexpected size {img.size}")
            except Exception as exc:
                decode_failures += 1
                issues.append(f"Sample {i}: decode failed — {exc}")

        if decode_failures > 0:
            sample_decode_ok = False

        # Compute quality score
        checks_passed = sum([
            schema_valid,
            sample_decode_ok,
            len(issues) == 0,
            num_train > 0,
            num_test > 0,
        ])
        data_quality_score = checks_passed / 5.0

        self.tracker.log_metric(self.STAGE_NAME, "num_train_samples", num_train)
        self.tracker.log_metric(self.STAGE_NAME, "num_test_samples", num_test)
        self.tracker.log_metric(self.STAGE_NAME, "schema_valid", schema_valid)
        self.tracker.log_metric(self.STAGE_NAME, "sample_decode_ok", sample_decode_ok)
        self.tracker.log_metric(self.STAGE_NAME, "data_quality_score", data_quality_score)

        if issues:
            for issue in issues:
                self.tracker.log_event(self.STAGE_NAME, "warning", issue)

        return {
            "num_train_samples": num_train,
            "num_test_samples": num_test,
            "schema_valid": schema_valid,
            "sample_decode_ok": sample_decode_ok,
            "data_quality_score": data_quality_score,
            "issues": issues,
        }

    def _emit_checkpoint(self, event_type: str, data: dict | None = None) -> None:
        """Emit an agent checkpoint if emitter is available."""
        if self.checkpoint_emitter is not None:
            from agents.checkpoints import AgentCheckpoint
            checkpoint = AgentCheckpoint(
                stage=self.STAGE_NAME,
                event_type=event_type,
                timestamp=now_iso(),
                metrics_snapshot=data or {},
                anomalies_detected=[],
                agent_diagnosis=None,
                recommended_action=None,
                action_taken=None,
            )
            self.checkpoint_emitter.emit(checkpoint)
