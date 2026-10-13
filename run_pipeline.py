#!/usr/bin/env python
"""AutoPARK Unified Pipeline CLI.

Runs the end-to-end AutoPARK ML training pipeline (Data Validation -> Data Loading
-> Training -> [Tuning] -> Evaluation) with optional in-the-loop agentic self-healing
powered by Claude on AWS.

Usage examples:
    # Quick smoke test (non-agentic, tiny model, small subset)
    python run_pipeline.py --model tiny --epochs 2 --subset 500 --no-tune

    # Full pipeline with AutoPARK agentic self-healing
    python run_pipeline.py --model tiny --epochs 5 --agentic --no-tune

    # Pipeline with hyperparameter tuning
    python run_pipeline.py --model tiny --epochs 3 --subset 500 --tune --tune-samples 2

    # ResNet18 full training
    python run_pipeline.py --model resnet18 --epochs 20

    # Comparative evaluation: standard vs. AutoPARK agentic mode
    python run_pipeline.py --model tiny --epochs 3 --subset 1000 --no-agentic --no-tune
    python run_pipeline.py --model tiny --epochs 3 --subset 1000 --agentic --no-tune
"""
from __future__ import annotations

import argparse
import sys
from dotenv import load_dotenv

# Always reload latest credentials and configuration from .env
load_dotenv(override=True)

from pipeline.config import PipelineConfig
from pipeline.runner import PipelineRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the end-to-end agentic ML training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model & training
    p.add_argument("--model", type=str, default="tiny",
                    help="Model architecture from registry (default: tiny; choices: tiny, resnet18, mlp, simple_cnn)")
    p.add_argument("--dataset", type=str, default="cifar10",
                    help="Dataset module to load (default: cifar10; choices: cifar10, synthetic)")
    p.add_argument("--num-classes", type=int, default=10,
                    help="Number of classification target classes (default: 10)")
    p.add_argument("--epochs", type=int, default=5,
                    help="Number of training epochs (default: 5)")
    p.add_argument("--batch-size", type=int, default=128,
                    help="Training batch size (default: 128)")
    p.add_argument("--lr", type=float, default=0.1,
                    help="Initial learning rate (default: 0.1)")
    p.add_argument("--momentum", type=float, default=0.9,
                    help="SGD momentum (default: 0.9)")
    p.add_argument("--weight-decay", type=float, default=5e-4,
                    help="Weight decay (default: 5e-4)")

    # Device & workers
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None,
                    help="Compute device (default: auto-detected cuda/mps/cpu)")
    p.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers (default: 0)")

    # Agentic monitoring
    agentic_group = p.add_mutually_exclusive_group()
    agentic_group.add_argument("--agentic", action="store_true", default=False,
                                help="Enable agentic monitoring")
    agentic_group.add_argument("--no-agentic", action="store_true", default=False,
                                help="Disable agentic monitoring (default)")

    # Hyperparameter tuning
    tune_group = p.add_mutually_exclusive_group()
    tune_group.add_argument("--tune", action="store_true", default=False,
                             help="Enable hyperparameter tuning stage")
    tune_group.add_argument("--no-tune", action="store_true", default=False,
                             help="Disable hyperparameter tuning (default)")
    p.add_argument("--tune-samples", type=int, default=4,
                    help="Number of tuning trials (default: 4)")

    # Pipeline control
    p.add_argument("--max-retries", type=int, default=3,
                    help="Max automatic retries per stage (default: 3)")
    p.add_argument("--subset", type=int, default=None,
                    help="Use only N samples for fast testing")
    p.add_argument("--run-id", type=str, default=None,
                    help="Explicit run ID (default: auto-generated)")
    p.add_argument("--cache-dir", type=str, default="data/cifar10",
                    help="Data cache directory")
    p.add_argument("--log-every", type=int, default=50,
                    help="Print training progress every N batches")

    # Reversion & Rollback Inspection
    p.add_argument("--inspect-run", type=str, default=None,
                    help="Inspect an old run ID or 'latest' to view issues detected, changes, diagnosis, and rollback manifest")
    p.add_argument("--rollback-run", type=str, default=None,
                    help="Restore pre-intervention parameters from an old run ID or 'latest' and execute pipeline")

    return p.parse_args()


def main():
    args = parse_args()

    # Handle run inspection
    if args.inspect_run:
        from pipeline.reversion import generate_manifest_from_run, format_reversion_terminal
        manifest = generate_manifest_from_run(args.inspect_run)
        print(format_reversion_terminal(manifest))
        print(f"📄 Full Markdown manifest: experiments/{manifest.run_id}/reversion_manifest.md")
        print(f"📊 JSON manifest:          experiments/{manifest.run_id}/reversion_manifest.json\n")
        sys.exit(0)

    # Resolve agentic flag
    enable_agentic = args.agentic and not args.no_agentic
    enable_tuning = args.tune and not args.no_tune

    from pipeline.config import get_default_device
    device = args.device or get_default_device()

    config = PipelineConfig(
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=device,
        use_gpu=device in ("mps", "cuda"),
        enable_agentic=enable_agentic,
        max_retries=args.max_retries,
        enable_tuning=enable_tuning,
        tune_num_samples=args.tune_samples,
        subset=args.subset,
        cache_dir=args.cache_dir,
        run_id=args.run_id,
        log_every=args.log_every,
        dataset=args.dataset,
        num_classes=args.num_classes,
    )

    # Handle rollback execution
    if args.rollback_run:
        from pipeline.reversion import apply_rollback, generate_manifest_from_run
        manifest = generate_manifest_from_run(args.rollback_run)
        print(f"\n🔄 Rolling back to pre-intervention configuration from run '{manifest.run_id}':")
        if not manifest.rollback_diff:
            print("   ℹ️ No parameters were mutated in that run; executing with original configuration.")
        else:
            for k, v in manifest.rollback_diff.items():
                print(f"   • Reverting {k} ➔ {v}")
        apply_rollback(args.rollback_run, config)
        print()


    runner = PipelineRunner(config)
    result = runner.run()

    # Print final summary
    if result["status"] == "success":
        print("🚀 Pipeline completed successfully!")
        print(f"   Results: {result['results_dir']}")
    else:
        print("💥 Pipeline failed. Check logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
