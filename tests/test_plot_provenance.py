"""Check published chart inputs against recorded evidence, without cloud calls."""
import json
import statistics
from collections import Counter
from pathlib import Path

import pytest

from benchmarks import plot_analytical_graphs as plots


def test_legacy_results_cannot_generate_figures(tmp_path):
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps({"scenarios": []}))
    with pytest.raises(ValueError, match="schema-v2"):
        plots.generate_all_figures(source, tmp_path / "pdf", tmp_path / "png")
    assert not (tmp_path / "png").exists()


def test_published_aggregates_match_run_records_and_have_traces():
    root = Path(__file__).resolve().parents[1]
    bundle = root / "benchmarks/reports/eks/eks-20261027T131911Z/bundle"
    data = json.loads((bundle / "results.json").read_text())
    for row in data["aggregates"]:
        runs = [r for r in data["runs"] if r["id"] == row["id"]]
        assert row["n"] == len(runs)
        assert row["outcomes"] == dict(Counter(r["outcome"] for r in runs))
        for mode in ("baseline", "agentic"):
            assert row[f"{mode}_duration_s"]["median"] == pytest.approx(
                statistics.median(r[mode]["duration_s"] for r in runs))
            for run in runs:
                trace = bundle / "traces" / Path(run[mode]["run_dir"]).name
                timing = json.loads((trace / "timing.json").read_text())
                assert timing["timeline"]
        mttr = [r["agentic"]["timing"]["mttr_s"] for r in runs
                if r["agentic"]["timing"].get("incident")]
        if mttr:
            assert row["mttr_s"]["median"] == pytest.approx(statistics.median(mttr))
            assert row["mttr_s"]["min"] == min(mttr)
            assert row["mttr_s"]["max"] == max(mttr)
