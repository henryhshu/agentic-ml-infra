#!/usr/bin/env python
"""Unified CLI for the Agentic ML Fault Injection & Efficacy Benchmarking Suite.

Usage examples:
    # Run the complete benchmark suite across all scenarios
    python run_benchmark.py --all

    # Run a specific benchmark scenario
    python run_benchmark.py --scenario nan_divergence

    # List all preset benchmark scenarios
    python run_benchmark.py --list

    # Quick benchmarking with small dataset subset
    python run_benchmark.py --all --subset 300 --epochs 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarks.scenarios import PRESET_SCENARIOS, get_scenario, list_scenarios
from benchmarks.harness import BenchmarkHarness
from benchmarks.report import format_terminal_summary, generate_markdown_report, save_json_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Agentic ML Fault Injection & Efficacy Benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--all",
        action="store_true",
        help="Run all preset benchmark scenarios",
    )
    p.add_argument(
        "--scenario",
        choices=list(PRESET_SCENARIOS.keys()),
        help="Run a specific preset benchmark scenario",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List all available preset benchmark scenarios and exit",
    )
    p.add_argument(
        "--model",
        choices=["tiny", "resnet18"],
        default="tiny",
        help="Model architecture for benchmark runs (default: tiny)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs per training run (default: 2)",
    )
    p.add_argument(
        "--subset",
        type=int,
        default=500,
        help="Subset size of CIFAR-10 data samples for fast benchmarking (default: 500)",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per stage (default: 3)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="File path to save the generated Markdown benchmark report",
    )
    p.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="File path to save raw benchmark results as JSON",
    )

    return p.parse_args()


def print_scenario_list():
    """Display all registered benchmark scenarios in a clean table."""
    scenarios = list_scenarios()
    print("\n" + "=" * 80)
    print("  📋 PRESET AGENTIC ML BENCHMARK SCENARIOS")
    print("=" * 80)
    for sc in scenarios:
        rec_tag = "Auto-Healable" if sc.is_recoverable else "Early-Halt"
        print(f"\n  • ID:          {sc.id}")
        print(f"    Title:       {sc.title} [{rec_tag}]")
        print(f"    Category:    {sc.category}")
        print(f"    Description: {sc.description}")
        print(f"    Baseline:    {sc.expected_baseline_outcome}")
        print(f"    Agentic:     {sc.expected_agentic_outcome}")
    print("\n" + "=" * 80 + "\n")


def main():
    args = parse_args()

    if args.list:
        print_scenario_list()
        sys.exit(0)

    if not args.all and not args.scenario:
        print("Error: Specify either --all or --scenario <name> (use --list to see options).")
        sys.exit(1)

    # Determine scenarios to run
    if args.scenario:
        target_scenarios = [get_scenario(args.scenario)]
    else:
        target_scenarios = list_scenarios()

    harness = BenchmarkHarness(
        model=args.model,
        epochs=args.epochs,
        subset=args.subset,
        max_retries=args.max_retries,
    )

    # Run the benchmark suite
    suite_result = harness.run_suite(target_scenarios)

    # Display terminal summary
    summary_text = format_terminal_summary(suite_result)
    print(summary_text)

    # Generate Markdown and JSON reports
    md_path = generate_markdown_report(suite_result, args.output)
    json_path = save_json_report(suite_result, args.json_output)

    # Also update symlink/copy to benchmarks/LATEST_BENCHMARK.md
    latest_md = Path("benchmarks/LATEST_BENCHMARK.md")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"📄 Markdown Benchmark Report: {md_path}")
    print(f"📄 Latest Report Symlink:     {latest_md}")
    print(f"📊 JSON Telemetry Results:    {json_path}\n")


if __name__ == "__main__":
    main()
