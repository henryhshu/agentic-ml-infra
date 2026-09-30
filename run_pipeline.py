#!/usr/bin/env python
"""Unified pipeline CLI for the agentic ML training platform.

Usage examples:
    # Quick smoke test (non-agentic, tiny model, small subset)
    python run_pipeline.py --model tiny --epochs 2 --subset 500 --no-tune

    # Full pipeline with agentic monitoring
    python run_pipeline.py --model tiny --epochs 5 --agentic --no-tune

    # Pipeline with hyperparameter tuning
    python run_pipeline.py --model tiny --epochs 3 --subset 500 --tune --tune-samples 2

    # ResNet18 full training
    python run_pipeline.py --model resnet18 --epochs 20

    # Compare agentic vs non-agentic
    python run_pipeline.py --model tiny --epochs 3 --subset 1000 --no-agentic --no-tune
    python run_pipeline.py --model tiny --epochs 3 --subset 1000 --agentic --no-tune
"""
from __future__ import annotations

import argparse
import sys

from pipeline.config import PipelineConfig
from pipeline.runner import PipelineRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the end-to-end agentic ML training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model & training
    p.add_argument("--model", choices=["tiny", "resnet18"], default="tiny",
                    help="Model architecture (default: tiny)")
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
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps",
                    help="Compute device (default: mps for Mac GPU)")
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

    return p.parse_args()


def main():
    args = parse_args()

    # Resolve agentic flag
    enable_agentic = args.agentic and not args.no_agentic
    enable_tuning = args.tune and not args.no_tune

    config = PipelineConfig(
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        use_gpu=args.device in ("mps", "cuda"),
        enable_agentic=enable_agentic,
        max_retries=args.max_retries,
        enable_tuning=enable_tuning,
        tune_num_samples=args.tune_samples,
        subset=args.subset,
        cache_dir=args.cache_dir,
        run_id=args.run_id,
        log_every=args.log_every,
    )

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
