"""Comparison must remain paired, include failures, and preserve missing telemetry."""
import copy
import json

import numpy as np
import pytest

from evaluation.compare_schema_runs import (
    bootstrap_delta,
    compare,
    normalize_record,
    plot_runs,
    read_run,
    validate_runs,
)
from evaluation.plot_schema_retrieval import load_reports


def record(question_id, correct, ms=100, calls=None):
    return {
        "id": question_id, "db_id": "db1", "difficulty": "simple", "correct": correct,
        "elapsed_ms": ms, "prediction": {"schema_retrieval": {"latency_ms": 2,
        "generation": {"calls": calls or [], "oom_retries": 0}}},
    }


def run(name, rows, mode="full"):
    return {"name": name, "directory": name,
            "summary": {"questions": len(rows), "schema_mode": mode, "dataset_sha256": "same-dataset", "seed": 42,
                        "backend": "mock", "mode": "greedy", "execution_metric": "project"},
            "config": {"model": {"model_id": "same"}, "validation": {"schema_selection": mode != "full"},
                       "retrieval": {"mode": "bm25"}, "generation": {"batch_size": 1}},
            "rows": {json.dumps([row["id"], row["db_id"]], ensure_ascii=False): normalize_record(row) for row in rows}}


def test_pairing_by_id_not_file_order_and_failures_in_denominator(tmp_path):
    full = run("full", [record(1, True), record(2, False), record(3, True)])
    pruned = run("pruned", [record(3, False, 50), record(1, True, 50), record(2, True, 50)], "bm25")
    report = compare([full, pruned], tmp_path, samples=100)
    row = report["runs"][1]
    assert row["questions"] == 3
    assert row["project_ex_pct"] == pytest.approx(200 / 3)
    assert row["gained"] == row["lost"] == 1
    assert row["ex_delta_pp"] == 0
    assert row["end_to_end_speedup_paired"] == 2
    assert row["cuda_peak_allocated_mb_device_max_mean"] is None
    assert row["input_tokens_call_sum_mean"] is None
    assert (tmp_path / "paired_questions.csv").exists()
    # A stale VRAM plot from an older profiled run must not survive missing telemetry.
    (tmp_path / "cuda_peak_allocated_mb_device_max.png").write_bytes(b"old plot")
    skipped = plot_runs([full, pruned], report, tmp_path)
    assert "input_tokens_first: no telemetry" in skipped
    assert (tmp_path / "accuracy_and_transitions.png").exists()
    assert not (tmp_path / "cuda_peak_allocated_mb_device_max.png").exists()


@pytest.mark.parametrize("change", ["ids", "config", "dataset", "backend", "difficulty"])
def test_rejects_unfair_comparisons(change):
    full = run("full", [record(1, True)])
    pruned = run("pruned", [record(1, False)], "bm25")
    if change == "ids":
        pruned["rows"] = {}
    elif change == "config":
        pruned["config"]["generation"]["batch_size"] = 2
    elif change == "difficulty":
        next(iter(pruned["rows"].values()))["difficulty"] = "challenging"
    else:
        pruned["summary"]["dataset_sha256" if change == "dataset" else "backend"] = "different"
    with pytest.raises(ValueError):
        validate_runs([full, pruned])


def test_input_tokens_are_call_sum_not_sequence_multiplied():
    row = normalize_record(record(1, False, calls=[
        {"input_tokens": 100, "output_tokens": 40, "sequences": 3, "generation_ms": 100, "context_overflow": False},
        {"input_tokens": 80, "output_tokens": 0, "sequences": 0, "generation_ms": 0, "context_overflow": True}]))
    assert row["input_tokens_call_sum"] == 180
    assert row["input_tokens_first"] == 100
    assert row["output_tokens_total"] == 40
    assert row["generation_ms"] == 100
    assert row["context_overflows"] == 1
    assert row["cuda_peak_allocated_mb_device_max"] is None


def test_partial_telemetry_is_not_silently_counted_as_zero():
    row = normalize_record(record(1, True, calls=[{"input_tokens": 10}, {}]))
    assert row["input_tokens_call_sum"] is None
    with pytest.raises(ValueError, match="boolean"):
        normalize_record({"id": 1})


def test_duplicate_ids_rejected(tmp_path):
    full = run("full", [record(1, True)])
    for filename in ("summary", "config"):
        (tmp_path / f"{filename}.json").write_text(json.dumps(full[filename]))
    (tmp_path / "predictions.jsonl").write_text((json.dumps(record(1, True)) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        read_run("full", tmp_path)


def test_bootstrap_deterministic_and_database_blocks():
    base = np.array([1., 0., 0.])
    variant = np.array([0., 1., 1.])
    first = bootstrap_delta(base, variant, ["a", "a", "b"], 100, 42, "db")
    assert first == bootstrap_delta(base, variant, ["a", "a", "b"], 100, 42, "db")
    assert first == pytest.approx((0, 1))


def test_zero_latency_does_not_create_infinite_speedup(tmp_path):
    full = run("full", [record(1, True, 0)])
    pruned = run("bm25", [record(1, True, 0)], "bm25")
    report = compare([full, pruned], tmp_path, samples=10)
    assert report["runs"][1]["end_to_end_speedup_paired"] is None


def test_retrieval_plots_require_matching_provenance(tmp_path):
    report = {"label": "full", "dataset_sha256": "abc", "schemas_sha256": "xyz",
              "summary": {"questions": 2, "label_errors": 0}}
    content = {"tokenizer_provenance": {"model": "one"}, "reports": [report]}
    left = tmp_path / "left.json"
    left.write_text(json.dumps(content))
    assert len(load_reports([left])) == 1
    right = tmp_path / "right.json"
    changed = copy.deepcopy(content)
    changed["reports"][0]["label"] = "bm25"
    changed["tokenizer_provenance"]["model"] = "two"
    right.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="provenance"):
        load_reports([left, right])


def test_different_runtime_weights_rejected():
    full = run("full", [record(1, True)])
    pruned = run("bm25", [record(1, False)], "bm25")
    full["runtime_manifest"] = {"weights_sha256": "one"}
    pruned["runtime_manifest"] = {"weights_sha256": "another"}
    with pytest.raises(ValueError, match="manifests"):
        validate_runs([full, pruned])


def test_truncated_predictions_rejected_against_summary(tmp_path):
    full = run("full", [record(1, True), record(2, False)])
    for filename in ("summary", "config"):
        (tmp_path / f"{filename}.json").write_text(json.dumps(full[filename]))
    (tmp_path / "predictions.jsonl").write_text(json.dumps(record(1, True)) + "\n")
    with pytest.raises(ValueError, match="Prediction count"):
        read_run("full", tmp_path)
