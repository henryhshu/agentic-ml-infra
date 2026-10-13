"""AutoPARK Reversion & Rollback Manifest Module.

Analyzes telemetry, incident audits, and shared-RAM mutation logs from past runs
to generate explicit, actionable Reversion & Rollback Manifests.

Provides:
  1. Complete 4-Part Incident Summary:
     - Issue Detected (anomalies, raw errors, stages)
     - What Was Changed (exact parameter mutations and code diffs)
     - Why It Was Changed (Claude cognitive root-cause diagnosis)
     - How To Change It Back (CLI flags, Python overrides, and git/patch commands)
  2. Multi-Hop Rollback Resolution:
     Tracks chained mutations across multiple remediation cycles to reconstruct
     the exact initial pre-intervention parameters.
  3. Programmatic & CLI Rollback Execution:
     Re-applies pre-intervention parameters directly to live PipelineConfig or executes
     restored runs via `python run_pipeline.py --rollback-run <run_id>`.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.config import PipelineConfig


@dataclass
class MutationRecord:
    timestamp: str
    action: str
    status: str
    details: str
    config_diff: dict[str, Any]
    patch: str | None = None


@dataclass
class IncidentRecord:
    stage: str
    anomalies: list[str]
    raw_anomalies: list[str]
    diagnosis: str
    action_taken: str | None


@dataclass
class ReversionManifest:
    """Complete Reversion and Rollback Specification for an AutoPARK run."""
    run_id: str
    timestamp: str
    issues_detected: list[dict[str, Any]]
    mutations: list[dict[str, Any]]
    original_config: dict[str, Any]
    final_config: dict[str, Any]
    rollback_diff: dict[str, Any]
    rollback_command: str
    reversion_python_code: str
    patch_revert_command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReversionManifest:
        return cls(**data)


def resolve_run_dir(run_id_or_path: str | Path, base_dir: str | Path = "experiments") -> Path:
    """Resolve a run identifier or path to an absolute Path."""
    p = Path(run_id_or_path)
    if p.exists() and p.is_dir():
        return p.resolve()

    base = Path(base_dir)
    target = base / run_id_or_path
    if target.exists():
        if target.is_symlink():
            return target.resolve()
        return target.resolve()

    raise FileNotFoundError(f"Experiment run directory not found: '{run_id_or_path}' (searched in {base.resolve()})")


def generate_manifest_from_run(
    run_id_or_path: str | Path,
    base_dir: str | Path = "experiments",
    save_files: bool = True,
) -> ReversionManifest:
    """Extract telemetry from an existing run and synthesize a Reversion & Rollback Manifest.
    
    Compatible with both historical benchmark runs and newly executed runs.
    """
    run_dir = resolve_run_dir(run_id_or_path, base_dir)
    run_id = run_dir.name

    # 1. Load config.json (initial configuration)
    config_data: dict[str, Any] = {}
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception:
            pass

    # 2. Load incidents.json (issues detected & why changed)
    incidents_raw: list[dict[str, Any]] = []
    incidents_path = run_dir / "incidents.json"
    if incidents_path.exists():
        try:
            with open(incidents_path, "r", encoding="utf-8") as f:
                incidents_raw = json.load(f)
        except Exception:
            pass

    # 3. Load action_log.json (what was changed)
    action_log: list[dict[str, Any]] = []
    action_path = run_dir / "action_log.json"
    if action_path.exists():
        try:
            with open(action_path, "r", encoding="utf-8") as f:
                action_log = json.load(f)
        except Exception:
            pass

    # 4. Compute original pre-intervention values and rollback diff
    # Chronological forward pass: track first known 'old' value for every mutated key
    original_values: dict[str, Any] = {}
    final_values: dict[str, Any] = {}
    collected_patches: list[str] = []

    for item in action_log:
        cdiff = item.get("config_diff", {})
        for key, val in cdiff.items():
            if isinstance(val, dict) and "old" in val and "new" in val:
                if key not in original_values:
                    original_values[key] = val["old"]
                final_values[key] = val["new"]
            elif key in ("patch", "ast_patch") and isinstance(val, str) and val.strip():
                collected_patches.append(val)

    # Reconstruct complete original & final configs
    orig_config = dict(config_data)
    for k, v in original_values.items():
        orig_config[k] = v

    final_config = dict(config_data)
    for k, v in final_values.items():
        final_config[k] = v

    # The rollback diff maps any changed knob back to its initial pre-intervention value
    rollback_diff = {k: orig_config[k] for k in original_values}

    # 5. Formulate rollback command and code snippets
    cli_flags: list[str] = ["python", "run_pipeline.py"]
    if "model" in orig_config:
        cli_flags.extend(["--model", str(orig_config["model"])])
    if "batch_size" in rollback_diff:
        cli_flags.extend(["--batch-size", str(rollback_diff["batch_size"])])
    if "lr" in rollback_diff:
        cli_flags.extend(["--lr", str(rollback_diff["lr"])])
    if "accum_steps" in rollback_diff and rollback_diff["accum_steps"] != 1:
        cli_flags.extend(["--accum-steps", str(rollback_diff["accum_steps"])])

    rollback_cmd = f"python run_pipeline.py --rollback-run {run_id}"

    # Python code snippet
    code_lines = ["# Revert mutated parameters to original pre-intervention values:"]
    for k, v in rollback_diff.items():
        code_lines.append(f"config.{k} = {repr(v)}")
    reversion_python = "\n".join(code_lines)

    # Patch reversion
    patch_cmd: str | None = None
    if collected_patches:
        patch_file = run_dir / "reversion_patch.diff"
        if save_files:
            with open(patch_file, "w", encoding="utf-8") as f:
                f.write("\n\n".join(collected_patches))
        patch_cmd = f"patch -R -p1 < {patch_file.name}  # Or: git checkout -- <modified_files>"

    manifest = ReversionManifest(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        issues_detected=incidents_raw,
        mutations=action_log,
        original_config=orig_config,
        final_config=final_config,
        rollback_diff=rollback_diff,
        rollback_command=rollback_cmd,
        reversion_python_code=reversion_python,
        patch_revert_command=patch_cmd,
    )

    if save_files:
        manifest_json_path = run_dir / "reversion_manifest.json"
        with open(manifest_json_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        manifest_md_path = run_dir / "reversion_manifest.md"
        with open(manifest_md_path, "w", encoding="utf-8") as f:
            f.write(format_reversion_markdown(manifest))

    return manifest


def format_reversion_markdown(manifest: ReversionManifest) -> str:
    """Format the ReversionManifest into a clear GitHub Markdown summary."""
    lines = [
        f"# AutoPARK Reversion & Rollback Manifest (`{manifest.run_id}`)",
        "",
        f"> **Generated at**: `{manifest.timestamp}`  ",
        f"> **Run ID**: `{manifest.run_id}`  ",
        f"> **Total Autonomous Interventions**: {len(manifest.mutations)}  ",
        "",
        "---",
        "",
        "## 🔴 1. Issues Detected",
        "",
    ]

    if not manifest.issues_detected:
        lines.append("*(No incidents recorded in this run)*\n")
    else:
        for idx, inc in enumerate(manifest.issues_detected, start=1):
            stage = inc.get("stage", "unknown")
            anoms = ", ".join(inc.get("anomalies", [])) or "None"
            raw_errs = "\n".join(f"> `{e}`" for e in inc.get("raw_anomalies", []))
            lines.extend([
                f"### Incident {idx} (Stage: `{stage}`)",
                f"- **Anomaly Classification**: `{anoms}`",
                "- **Raw Error Signatures**:",
                raw_errs,
                "",
            ])

    lines.extend([
        "---",
        "",
        "## 🛠️ 2. What Was Changed",
        "",
    ])

    if not manifest.mutations:
        lines.append("*(No parameter mutations applied during this run)*\n")
    else:
        lines.extend([
            "| # | Action | Status | Details | Parameter Diffs |",
            "|---|---|---|---|---|",
        ])
        for idx, m in enumerate(manifest.mutations, start=1):
            act = m.get("action", "")
            stat = m.get("status", "")
            det = m.get("details", "")
            diff_str = ", ".join(
                f"`{k}`: {v.get('old')} &rarr; {v.get('new')}"
                for k, v in m.get("config_diff", {}).items()
                if isinstance(v, dict) and "old" in v
            )
            if not diff_str and "use_group_norm" in m.get("config_diff", {}):
                diff_str = "GroupNorm AST rewrite applied"
            lines.append(f"| {idx} | `{act}` | `{stat}` | {det} | {diff_str} |")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 🧠 3. Why It Was Changed (Cognitive Diagnosis)",
        "",
    ])

    if not manifest.issues_detected:
        lines.append("*(No cognitive diagnosis required)*\n")
    else:
        for idx, inc in enumerate(manifest.issues_detected, start=1):
            diag = inc.get("diagnosis", "").strip()
            if diag:
                lines.extend([
                    f"### Diagnosis for Incident {idx}:",
                    diag,
                    "",
                ])

    lines.extend([
        "---",
        "",
        "## 🔄 4. How To Change It Back (Reversion & Rollback Guide)",
        "",
        "### A. Automated One-Command Rollback",
        "Re-run the training pipeline restoring the exact pre-intervention configuration:",
        "```bash",
        manifest.rollback_command,
        "```",
        "",
        "### B. Manual Configuration Override",
        "```python",
        manifest.reversion_python_code,
        "```",
        "",
    ])

    if manifest.rollback_diff:
        lines.extend([
            "### C. Parameter Rollback Summary",
            "| Parameter | Mutated (Post-Fix) | Restored (Pre-Fix) |",
            "|---|---|---|",
        ])
        for k, orig_v in manifest.rollback_diff.items():
            curr_v = manifest.final_config.get(k, "?")
            lines.append(f"| `{k}` | `{curr_v}` | **`{orig_v}`** |")
        lines.append("")

    if manifest.patch_revert_command:
        lines.extend([
            "### D. Codebase / AST Patch Reversion",
            "To reverse synthesized source-code AST modifications:",
            "```bash",
            manifest.patch_revert_command,
            "```",
            "",
        ])

    return "\n".join(lines)


def format_reversion_terminal(manifest: ReversionManifest) -> str:
    """Format concise, beautiful terminal summary for CLI output."""
    sep = "=" * 70
    sub_sep = "-" * 70
    out = [
        f"\n{sep}",
        f"  📋 AUTOPARK REVERSION & ROLLBACK MANIFEST (`{manifest.run_id}`)",
        f"{sep}",
        "",
        "🔴 1. ISSUES DETECTED:",
    ]
    if not manifest.issues_detected:
        out.append("   (None recorded)")
    else:
        for idx, inc in enumerate(manifest.issues_detected, start=1):
            anoms = ", ".join(inc.get("anomalies", []))
            stage = inc.get("stage", "training")
            out.append(f"   [{idx}] Stage '{stage}' ➔ Anomaly: {anoms}")
            for err in inc.get("raw_anomalies", []):
                out.append(f"       Signature: {err[:90]}")

    out.extend(["", "🛠️  2. WHAT WAS CHANGED:"])
    if not manifest.mutations:
        out.append("   (No mutations executed)")
    else:
        for idx, m in enumerate(manifest.mutations, start=1):
            diffs = [
                f"{k}: {v.get('old')} -> {v.get('new')}"
                for k, v in m.get("config_diff", {}).items()
                if isinstance(v, dict) and "old" in v
            ]
            diff_desc = f" ({', '.join(diffs)})" if diffs else ""
            out.append(f"   [{idx}] {m.get('action')}{diff_desc}")
            out.append(f"       Details: {m.get('details')}")

    out.extend(["", "🧠 3. WHY IT WAS CHANGED:"])
    if not manifest.issues_detected:
        out.append("   (No diagnosis available)")
    else:
        for idx, inc in enumerate(manifest.issues_detected, start=1):
            diag = inc.get("diagnosis", "").strip()
            # Extract first 2-3 lines of diagnosis for concise CLI view
            first_lines = [l for l in diag.splitlines() if l.strip() and not l.startswith("#")][:2]
            diag_snippet = " ".join(first_lines) if first_lines else diag[:140]
            out.append(f"   [{idx}] {diag_snippet[:120]}...")

    out.extend([
        "",
        "🔄 4. HOW TO CHANGE IT BACK:",
        f"   • Automated Rollback Command:",
        f"     {manifest.rollback_command}",
    ])

    if manifest.rollback_diff:
        out.append("   • Restored Parameters:")
        for k, v in manifest.rollback_diff.items():
            out.append(f"     - {k}: {v}")

    if manifest.patch_revert_command:
        out.append(f"   • Code Patch Revert: {manifest.patch_revert_command}")

    out.extend(["", sep, ""])
    return "\n".join(out)


def apply_rollback(
    run_id_or_path: str | Path,
    config: PipelineConfig | None = None,
    base_dir: str | Path = "experiments",
) -> PipelineConfig:
    """Load the rollback manifest for an old run and apply the inverted diff to a PipelineConfig."""
    manifest = generate_manifest_from_run(run_id_or_path, base_dir=base_dir, save_files=False)

    target_config = config or PipelineConfig()

    for k, v in manifest.rollback_diff.items():
        if hasattr(target_config, k):
            setattr(target_config, k, v)

    return target_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoPARK Reversion & Rollback Manifest Inspector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-id", type=str, default="latest", help="Run ID or path (default: latest)")
    parser.add_argument("--base-dir", type=str, default="experiments", help="Experiments base directory")
    parser.add_argument("--apply", action="store_true", help="Apply rollback configuration to a new run")
    parser.add_argument("--list", action="store_true", help="List all available runs with manifests")

    args = parser.parse_args()

    if args.list:
        base = Path(args.base_dir)
        runs = sorted([d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")])
        print(f"Available AutoPARK Experiment Runs ({len(runs)} found):")
        for r in runs:
            print(f"  • {r}")
        return

    try:
        manifest = generate_manifest_from_run(args.run_id, base_dir=args.base_dir, save_files=True)
        print(format_reversion_terminal(manifest))
        print(f"📄 Full Markdown manifest saved to: experiments/{manifest.run_id}/reversion_manifest.md")
        print(f"📊 JSON manifest saved to:     experiments/{manifest.run_id}/reversion_manifest.json\n")
    except Exception as e:
        print(f"❌ Failed to inspect run '{args.run_id}': {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
