"""AWS S3 configuration and storage utilities for agentic ML training platform."""
from __future__ import annotations

import io
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


class S3StorageError(Exception):
    """Base exception for S3 storage operations."""
    pass


@dataclass
class S3Config:
    """Configuration for AWS S3 bucket and key prefixes used in ML workflows.

    Attributes:
        bucket_name: Name of the target S3 bucket.
        region: AWS region where the bucket resides (default: 'us-west-2').
        data_prefix: S3 key prefix for dataset storage (default: 'datasets/').
        experiments_prefix: S3 key prefix for experiment tracking (default: 'experiments/').
        endpoint_url: Optional custom S3 endpoint URL (e.g. for LocalStack or VPC endpoints).
    """
    bucket_name: str
    region: str = "us-west-2"
    data_prefix: str = "datasets/"
    experiments_prefix: str = "experiments/"
    endpoint_url: str | None = None

    def __post_init__(self) -> None:
        """Normalize prefixes to ensure proper trailing slashes."""
        if not self.bucket_name:
            raise ValueError("bucket_name must not be empty.")
        if self.data_prefix and not self.data_prefix.endswith("/"):
            self.data_prefix += "/"
        if self.experiments_prefix and not self.experiments_prefix.endswith("/"):
            self.experiments_prefix += "/"

    def get_s3_client(self) -> Any:
        """Create and return a configured boto3 S3 client."""
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise S3StorageError(
                "boto3 is required for S3 operations. Please install boto3."
            ) from exc

        client_kwargs: dict[str, Any] = {
            "region_name": self.region,
            "config": Config(retries={"max_attempts": 5, "mode": "standard"}),
        }
        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url

        try:
            return boto3.client("s3", **client_kwargs)
        except Exception as exc:
            raise S3StorageError(
                f"Failed to create boto3 S3 client for region {self.region}: {exc}"
            ) from exc

    def dataset_key(self, split: str, filename: str | None = None) -> str:
        """Generate the S3 object key for a dataset split.

        Args:
            split: Dataset split name (e.g., 'train', 'test').
            filename: Optional custom filename (defaults to '{split}.parquet').

        Returns:
            Normalized S3 key string.
        """
        fname = filename if filename else f"{split}.parquet"
        return f"{self.data_prefix}{fname}"

    def experiment_prefix(self, run_id: str) -> str:
        """Generate the S3 prefix for a specific experiment run ID.

        Args:
            run_id: Unique identifier for the experiment run.

        Returns:
            Normalized S3 prefix string with trailing slash.
        """
        return f"{self.experiments_prefix}{run_id}/"

    def experiment_key(self, run_id: str, filename: str) -> str:
        """Generate the S3 object key for an artifact in a run.

        Args:
            run_id: Unique identifier for the experiment run.
            filename: Name of the artifact file (e.g., 'experiment_summary.json').

        Returns:
            Normalized S3 key string.
        """
        return f"{self.experiment_prefix(run_id)}{filename}"

    def s3_uri(self, key: str) -> str:
        """Generate full s3:// URI for a key."""
        return f"s3://{self.bucket_name}/{key.lstrip('/')}"


def upload_dataset(
    s3_config: S3Config,
    local_path: str | Path,
    split: str,
    s3_client: Any = None,
) -> str:
    """Upload a local dataset parquet file to S3.

    Args:
        s3_config: Configuration containing bucket and prefix details.
        local_path: Path to the local parquet file.
        split: Dataset split ('train', 'test', etc.).
        s3_client: Optional pre-configured boto3 S3 client.

    Returns:
        The S3 URI where the dataset was uploaded.

    Raises:
        FileNotFoundError: If local_path does not exist.
        S3StorageError: If the upload to S3 fails.
    """
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"Local dataset file not found at: {path}")

    client = s3_client or s3_config.get_s3_client()
    key = s3_config.dataset_key(split, filename=path.name)
    s3_uri = s3_config.s3_uri(key)

    try:
        logger.info(f"Uploading dataset '{path}' to '{s3_uri}'...")
        client.upload_file(
            Filename=str(path),
            Bucket=s3_config.bucket_name,
            Key=key,
            ExtraArgs={"ContentType": "application/x-parquet"},
        )
        logger.info(f"Successfully uploaded dataset split '{split}' to {s3_uri}")
        return s3_uri
    except Exception as exc:
        msg = f"Failed to upload dataset '{path}' to '{s3_uri}': {exc}"
        logger.error(msg)
        raise S3StorageError(msg) from exc


def download_dataset(
    s3_config: S3Config,
    split: str,
    local_path: str | Path,
    s3_client: Any = None,
    filename: str | None = None,
) -> Path:
    """Download a dataset parquet file from S3 to a local path.

    Args:
        s3_config: Configuration containing bucket and prefix details.
        split: Dataset split ('train', 'test', etc.).
        local_path: Target local file path.
        s3_client: Optional pre-configured boto3 S3 client.
        filename: Optional S3 filename override (defaults to '{split}.parquet').

    Returns:
        Path object pointing to the downloaded local file.

    Raises:
        S3StorageError: If download fails or object does not exist in S3.
    """
    target = Path(local_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    client = s3_client or s3_config.get_s3_client()
    key = s3_config.dataset_key(split, filename=filename or f"{split}.parquet")
    s3_uri = s3_config.s3_uri(key)

    try:
        logger.info(f"Downloading dataset from '{s3_uri}' to '{target}'...")
        client.download_file(
            Bucket=s3_config.bucket_name,
            Key=key,
            Filename=str(target),
        )
        logger.info(f"Successfully downloaded dataset split '{split}' to '{target}'")
        return target
    except Exception as exc:
        msg = f"Failed to download dataset from '{s3_uri}' to '{target}': {exc}"
        logger.error(msg)
        raise S3StorageError(msg) from exc


class S3MetricsStore:
    """S3-backed metrics and experiment storage for agentic ML workflows.

    Mirrors the local MetricsStore interface while persisting artifacts directly
    in AWS S3. Supports uploading/downloading entire runs, reading JSON summaries,
    comparing metrics across runs, and retrieving latest runs.
    """

    def __init__(self, s3_config: S3Config, s3_client: Any = None):
        """Initialize the S3MetricsStore.

        Args:
            s3_config: S3 configuration dataclass.
            s3_client: Optional pre-configured boto3 S3 client.
        """
        self.config = s3_config
        self._client = s3_client

    @property
    def client(self) -> Any:
        """Lazy-load and return the S3 client."""
        if self._client is None:
            self._client = self.config.get_s3_client()
        return self._client

    def upload_experiment(self, run_dir: str | Path) -> str:
        """Upload an entire local experiment run directory to S3.

        Recursively uploads all files (configs, metrics, checkpoints, summary)
        under `run_dir` to `s3://<bucket>/<experiments_prefix>/<run_id>/...`.

        Args:
            run_dir: Path to the local experiment directory (e.g., 'experiments/run_20260816_120000_a1b2').

        Returns:
            The S3 URI prefix of the uploaded experiment.

        Raises:
            FileNotFoundError: If run_dir does not exist.
            S3StorageError: If any upload operation fails.
        """
        local_path = Path(run_dir).resolve()
        if not local_path.exists() or not local_path.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {local_path}")

        run_id = local_path.name
        prefix = self.config.experiment_prefix(run_id)
        uploaded_files = 0

        logger.info(f"Uploading experiment '{run_id}' from '{local_path}' to '{self.config.s3_uri(prefix)}'...")

        try:
            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(local_path)
                    s3_key = f"{prefix}{rel_path.as_posix()}"
                    
                    content_type = "application/json" if file_path.suffix == ".json" else "application/octet-stream"
                    
                    self.client.upload_file(
                        Filename=str(file_path),
                        Bucket=self.config.bucket_name,
                        Key=s3_key,
                        ExtraArgs={"ContentType": content_type},
                    )
                    uploaded_files += 1

            logger.info(f"Successfully uploaded {uploaded_files} files for experiment '{run_id}'")
            return self.config.s3_uri(prefix)

        except Exception as exc:
            msg = f"Failed to upload experiment '{run_id}' from '{local_path}': {exc}"
            logger.error(msg)
            raise S3StorageError(msg) from exc

    def sync_run(self, run_dir: str | Path, run_id: str | None = None) -> int:
        """Sync a local experiment directory to S3 and return the number of files uploaded.

        Args:
            run_dir: Path to the local experiment directory.
            run_id: Optional explicit run ID. Defaults to directory name.

        Returns:
            Count of uploaded artifact files.
        """
        local_path = Path(run_dir).resolve()
        if not local_path.exists() or not local_path.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {local_path}")

        r_id = run_id or local_path.name
        prefix = self.config.experiment_prefix(r_id)
        uploaded_files = 0

        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(local_path)
                s3_key = f"{prefix}{rel_path.as_posix()}"
                content_type = "application/json" if file_path.suffix == ".json" else "application/octet-stream"
                self.client.upload_file(
                    Filename=str(file_path),
                    Bucket=self.config.bucket_name,
                    Key=s3_key,
                    ExtraArgs={"ContentType": content_type},
                )
                uploaded_files += 1

        logger.info(f"Successfully synced {uploaded_files} files for experiment '{r_id}' to S3")
        return uploaded_files

    def download_experiment(self, run_id: str, local_dir: str | Path) -> Path:
        """Download all files for an experiment run from S3 to a local directory.

        Args:
            run_id: Unique identifier for the experiment run.
            local_dir: Destination base directory where `{run_id}/` will be created.

        Returns:
            Path to the downloaded local experiment directory (`local_dir / run_id`).

        Raises:
            S3StorageError: If download fails or run_id does not exist in S3.
        """
        dest_base = Path(local_dir)
        target_dir = dest_base / run_id
        target_dir.mkdir(parents=True, exist_ok=True)

        prefix = self.config.experiment_prefix(run_id)
        logger.info(f"Downloading experiment '{run_id}' from '{self.config.s3_uri(prefix)}' to '{target_dir}'...")

        try:
            paginator = self.client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(
                Bucket=self.config.bucket_name,
                Prefix=prefix,
            )

            downloaded_count = 0
            for page in page_iterator:
                for obj in page.get("Contents", []):
                    s3_key = obj["Key"]
                    rel_key = s3_key[len(prefix):]
                    if not rel_key:
                        continue

                    local_file = target_dir / rel_key
                    local_file.parent.mkdir(parents=True, exist_ok=True)

                    self.client.download_file(
                        Bucket=self.config.bucket_name,
                        Key=s3_key,
                        Filename=str(local_file),
                    )
                    downloaded_count += 1

            if downloaded_count == 0:
                logger.warning(f"No files found in S3 for experiment '{run_id}' under prefix '{prefix}'")

            logger.info(f"Successfully downloaded {downloaded_count} files for '{run_id}' to '{target_dir}'")
            return target_dir

        except Exception as exc:
            msg = f"Failed to download experiment '{run_id}' from S3: {exc}"
            logger.error(msg)
            raise S3StorageError(msg) from exc

    def list_experiments(self) -> list[str]:
        """List all experiment run IDs stored in S3.

        Returns:
            Sorted list of run_id strings matching 'run_*'.
        """
        prefix = self.config.experiments_prefix
        run_ids: set[str] = set()

        try:
            paginator = self.client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(
                Bucket=self.config.bucket_name,
                Prefix=prefix,
                Delimiter="/",
            )

            for page in page_iterator:
                # CommonPrefixes contains directories under experiments_prefix
                for common_prefix in page.get("CommonPrefixes", []):
                    p_name = common_prefix.get("Prefix", "")
                    # Extract folder name: experiments/run_2026.../ -> run_2026...
                    parts = p_name.strip("/").split("/")
                    if parts:
                        run_folder = parts[-1]
                        if run_folder.startswith("run_"):
                            run_ids.add(run_folder)

            return sorted(list(run_ids))

        except Exception as exc:
            logger.error(f"Error listing experiments from bucket '{self.config.bucket_name}': {exc}")
            return []

    def load_experiment(self, run_id: str) -> dict[str, Any]:
        """Load and parse `experiment_summary.json` for a run directly from S3.

        Args:
            run_id: Unique identifier for the experiment run.

        Returns:
            Parsed experiment summary dictionary, or empty dict if not found/error.
        """
        key = self.config.experiment_key(run_id, "experiment_summary.json")
        try:
            response = self.client.get_object(Bucket=self.config.bucket_name, Key=key)
            body = response["Body"].read().decode("utf-8")
            return json.loads(body)
        except self.client.exceptions.NoSuchKey:
            logger.warning(f"experiment_summary.json not found for run '{run_id}' in S3 (Key: {key})")
            return {}
        except Exception as exc:
            logger.error(f"Failed to load experiment '{run_id}' from S3: {exc}")
            return {}

    def load_stage(self, run_id: str, stage_name: str) -> dict[str, Any]:
        """Load and parse `{stage_name}_metrics.json` for a run directly from S3.

        Args:
            run_id: Unique identifier for the experiment run.
            stage_name: Name of the stage (e.g., 'training', 'data_validation').

        Returns:
            Parsed stage metrics dictionary, or empty dict if not found/error.
        """
        key = self.config.experiment_key(run_id, f"{stage_name}_metrics.json")
        try:
            response = self.client.get_object(Bucket=self.config.bucket_name, Key=key)
            body = response["Body"].read().decode("utf-8")
            return json.loads(body)
        except self.client.exceptions.NoSuchKey:
            logger.warning(f"{stage_name}_metrics.json not found for run '{run_id}' in S3 (Key: {key})")
            return {}
        except Exception as exc:
            logger.error(f"Failed to load stage '{stage_name}' for run '{run_id}' from S3: {exc}")
            return {}

    def compare_experiments(
        self,
        run_ids: Sequence[str],
        metric_key: str = "final_test_accuracy",
    ) -> list[dict[str, Any]]:
        """Compare metric values across multiple experiment runs in S3.

        Args:
            run_ids: List of run IDs to compare.
            metric_key: Metric key to extract (default: 'final_test_accuracy').

        Returns:
            List of comparison records with run_id, metric value, and duration.
        """
        results: list[dict[str, Any]] = []
        for run_id in run_ids:
            exp = self.load_experiment(run_id)
            if not exp:
                continue

            stages = exp.get("stages", {})
            val = None

            for stage_name, stage_data in stages.items():
                if isinstance(stage_data, dict):
                    metrics_dict = stage_data.get("metrics", {})
                    if metric_key in metrics_dict:
                        val = metrics_dict[metric_key]
                    elif metric_key in stage_data:
                        val = stage_data[metric_key]

            res = {
                "run_id": run_id,
                metric_key: val,
                "duration_s": exp.get("duration_s"),
            }
            results.append(res)
        return results

    def get_latest(self) -> dict[str, Any] | None:
        """Retrieve the latest experiment summary from S3.

        Inspects the most recent run_id in sorted order.

        Returns:
            Latest experiment summary dict, or None if no experiments found.
        """
        experiments = self.list_experiments()
        if not experiments:
            return None

        latest_run_id = experiments[-1]
        summary = self.load_experiment(latest_run_id)
        return summary if summary else None
