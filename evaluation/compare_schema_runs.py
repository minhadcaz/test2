"""Pair full-schema and pruning evaluations, export metrics and honest plots.

Usage: python -m evaluation.compare_schema_runs --run full=results/full \
    --run bm25=results/bm25 --output results/comparison
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METRICS = {
    "elapsed_ms": "End-to-end latency (ms)",
    "input_tokens_call_sum": "Sum of prompt tokens across calls / question",
    "input_tokens_first": "Input tokens (first generation call)",
    "output_tokens_total": "Generated tokens / question (all sequences)",
    "generation_ms": "Generation time / question (ms)",
    "retrieval_ms": "Schema selection time (ms)",
    "cuda_peak_allocated_mb_device_max": "Largest device peak allocated memory (MiB)",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def total_if_complete(calls: list[dict], key: str):
    if not calls:
        return None
    values = []
    for call in calls:
        value = finite(call.get(key))
        if value is None:
            return None
        values.append(value)
    return sum(values)


def normalize_record(record: dict) -> dict:
    prediction = record.get("prediction") or {}
    retrieval = prediction.get("schema_retrieval") or {}
    generation = retrieval.get("generation") or {}
    calls = generation.get("calls") or []
    peaks = record.get("cuda_peak_allocated_mb_by_device") or {}
    values = [finite(value) for value in peaks.values()]
    valid_peaks = [value for value in values if value is not None]
    if type(record.get("correct")) is not bool:
        raise ValueError(f"Missing boolean 'correct' for question {record.get('id')!r}")
    return {
        "id": record["id"], "db_id": record.get("db_id"),
        "difficulty": record.get("difficulty"), "correct": record["correct"],
        "error_type": record.get("error_type") or ("none" if record["correct"] else "unknown"),
        "elapsed_ms": finite(record.get("elapsed_ms")),
        "input_tokens_call_sum": total_if_complete(calls, "input_tokens"),
        "input_tokens_first": finite(calls[0].get("input_tokens")) if calls else None,
        "output_tokens_total": total_if_complete(calls, "output_tokens"),
        "generation_ms": total_if_complete(calls, "generation_ms"),
        "retrieval_ms": finite(retrieval.get("latency_ms")),
        "cuda_peak_allocated_mb_device_max": max(valid_peaks) if valid_peaks else None,
        "context_overflows": sum(call["context_overflow"] for call in calls)
        if calls and all(type(call.get("context_overflow")) is bool for call in calls) else None,
        "oom_retries": finite(generation.get("oom_retries")),
    }


def comparison_config(config: dict) -> dict:
    """Remove only intervention fields; all generation/execution fields must match."""
    value = copy.deepcopy(config)
    value.pop("retrieval", None)
    validation = value.get("validation")
    if isinstance(validation, dict):
        validation.pop("schema_selection", None)
    return value


def read_run(name: str, directory: Path) -> dict:
    summary = load_json(directory / "summary.json")
    config = load_json(directory / "config.json")
    rows = {}
    with (directory / "predictions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = normalize_record(json.loads(line))
            key = json.dumps([row["id"], row["db_id"]], ensure_ascii=False)
            if key in rows:
                raise ValueError(f"Duplicate (id, db_id) in {name}: {key}")
            rows[key] = row
    if not rows:
        raise ValueError(f"Empty run: {name}")
    if summary.get("questions") != len(rows):
        raise ValueError(f"Prediction count differs from summary.questions in {name}: "
                         f"{len(rows)} vs {summary.get('questions')}")
    return {"name": name, "directory": str(directory), "summary": summary,
            "config": config, "rows": rows,
            "runtime_manifest": load_json(directory / "runtime_manifest.json")
            if (directory / "runtime_manifest.json").exists() else None}


def validate_runs(runs: list[dict]) -> None:
    baseline = runs[0]
    if baseline["summary"].get("schema_mode") != "full":
        raise ValueError("First run must have summary.schema_mode='full' (the baseline).")
    for run in runs[1:]:
        if set(run["rows"]) != set(baseline["rows"]):
            raise ValueError(f"Question sets differ: {run['name']}; do not compare partial runs.")
        for key in baseline["rows"]:
            if run["rows"][key]["difficulty"] != baseline["rows"][key]["difficulty"]:
                raise ValueError(f"Difficulty changed for {key} in {run['name']}")
        for field in ("dataset_sha256", "seed", "backend", "mode", "execution_metric"):
            left = baseline["summary"].get(field)
            right = run["summary"].get(field)
            if left != right or (field == "dataset_sha256" and not left):
                raise ValueError(f"Incompatible {field}: {baseline['name']} vs {run['name']}")
        if run.get("runtime_manifest") is not None and baseline.get("runtime_manifest") is not None:
            if run["runtime_manifest"] != baseline["runtime_manifest"]:
                raise ValueError(f"Runtime manifests differ for {run['name']} (weights/hardware/data).")
        if comparison_config(run["config"]) != comparison_config(baseline["config"]):
            raise ValueError(f"Model/generation/execution configs differ for {run['name']}.")


def bootstrap_delta(base: np.ndarray, variant: np.ndarray, groups: list[str],
                    samples: int, seed: int, unit: str) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    delta = variant - base
    if unit == "question":
        labels = [str(i) for i in range(len(groups))]
    else:
        labels = groups
    unique = sorted(set(labels))
    blocks = [np.array([i for i, value in enumerate(labels) if value == group]) for group in unique]
    estimates = np.empty(samples)
    for i in range(samples):
        selected = rng.integers(0, len(blocks), len(blocks))
        indices = np.concatenate([blocks[j] for j in selected])
        estimates[i] = delta[indices].mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare(runs: list[dict], output: Path, samples=2000, seed=42, unit="question") -> dict:
    validate_runs(runs)
    output.mkdir(parents=True, exist_ok=True)
    baseline = runs[0]
    keys = list(baseline["rows"])
    groups = [str(baseline["rows"][key]["db_id"]) for key in keys]
    base_correct = np.array([baseline["rows"][key]["correct"] for key in keys], dtype=float)
    summaries, pairs, difficulties = [], [], []
    for run in runs:
        rows = [run["rows"][key] for key in keys]
        correctness = np.array([row["correct"] for row in rows], dtype=float)
        low, high = bootstrap_delta(base_correct, correctness, groups, samples, seed, unit)
        summary = {
            "run": run["name"], "questions": len(rows),
            "correct": int(correctness.sum()), "project_ex_pct": correctness.mean() * 100,
            "ex_delta_pp": (correctness - base_correct).mean() * 100,
            "ex_delta_ci95_low_pp": low * 100, "ex_delta_ci95_high_pp": high * 100,
            "gained": int(((base_correct == 0) & (correctness == 1)).sum()),
            "lost": int(((base_correct == 1) & (correctness == 0)).sum()),
        }
        for source_field, output_field in (("valid_sql_rate", "valid_sql_rate_pct"),
                                           ("exact_match", "exact_match_pct")):
            value = finite(run["summary"].get(source_field))
            summary[output_field] = value * 100 if value is not None else None
        for count_metric in ("context_overflows", "oom_retries"):
            counts = [row[count_metric] for row in rows if row[count_metric] is not None]
            summary[f"{count_metric}_sum"] = sum(counts) if counts else None
            summary[f"{count_metric}_n"] = len(counts)
        for metric in METRICS:
            values = [row[metric] for row in rows if row[metric] is not None]
            summary[f"{metric}_n"] = len(values)
            for stat, value in (("mean", np.mean(values) if values else None),
                                ("p50", np.median(values) if values else None),
                                ("p95", np.percentile(values, 95) if values else None)):
                summary[f"{metric}_{stat}"] = float(value) if value is not None else None
        paired_latencies = [(baseline["rows"][key]["elapsed_ms"], run["rows"][key]["elapsed_ms"])
                            for key in keys if baseline["rows"][key]["elapsed_ms"] is not None
                            and run["rows"][key]["elapsed_ms"] is not None]
        base_ms = sum(x for x, _ in paired_latencies)
        variant_ms = sum(y for _, y in paired_latencies)
        summary["end_to_end_speedup_paired"] = base_ms / variant_ms if variant_ms > 0 else None
        summary["end_to_end_speedup_n"] = len(paired_latencies)
        for metric in ("input_tokens_first", "input_tokens_call_sum"):
            valid = [(baseline["rows"][key][metric], run["rows"][key][metric]) for key in keys
                     if baseline["rows"][key][metric] is not None and run["rows"][key][metric] is not None]
            denominator = sum(x for x, _ in valid)
            summary[f"{metric}_reduction_pct"] = (
                100 * (1 - sum(y for _, y in valid) / denominator) if denominator > 0 else None)
            summary[f"{metric}_reduction_n"] = len(valid)
        summaries.append(summary)
        for difficulty in sorted({str(row["difficulty"]) for row in rows}):
            subset = [row for row in rows if str(row["difficulty"]) == difficulty]
            difficulties.append({"run": run["name"], "difficulty": difficulty, "n": len(subset),
                                 "project_ex_pct": 100 * sum(row["correct"] for row in subset) / len(subset)})
        for key, row in zip(keys, rows):
            base = baseline["rows"][key]
            pair = {"run": run["name"], **row, "baseline_correct": base["correct"]}
            pair["transition"] = ("both_correct" if base["correct"] and row["correct"] else
                                  "lost" if base["correct"] else "gained" if row["correct"] else "both_wrong")
            for metric in METRICS:
                pair[f"baseline_{metric}"] = base[metric]
                pair[f"delta_{metric}"] = row[metric] - base[metric] if row[metric] is not None and base[metric] is not None else None
            pairs.append(pair)
    report = {"metric": "project_execution_match_not_official_BIRD_scorer", "baseline": baseline["name"],
              "provenance_warning": (None if all(run.get("runtime_manifest") for run in runs) else
                  "Runtime manifest missing: matching config alone cannot verify weights, SQLite files or hardware."),
              "bootstrap": {"samples": samples, "seed": seed, "unit": unit,
                            "units": len(keys) if unit == "question" else len(set(groups)),
                            "note": "Paired descriptive CI; no model-seed or hardware variability included."},
              "runs": summaries, "sources": [{"run": run["name"], "directory": run["directory"],
                  "dataset_sha256": run["summary"].get("dataset_sha256"),
                  "mock_notice": run["summary"].get("mock_notice"),
                  "runtime_manifest": run.get("runtime_manifest")} for run in runs]}
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "comparison.csv", summaries)
    write_csv(output / "paired_questions.csv", pairs)
    write_csv(output / "by_difficulty.csv", difficulties)
    return report


def plot_runs(runs: list[dict], report: dict, output: Path, pdf=False) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False})
    names = [run["name"] for run in runs]
    summaries = report["runs"]
    mock = any(run["summary"].get("backend") == "mock" for run in runs)
    suffix = " — SYNTHETIC / MOCK, not model quality" if mock else ""
    skipped = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", bbox_inches="tight")
        if pdf:
            fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        else:
            (output / f"{name}.pdf").unlink(missing_ok=True)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(names, [row["project_ex_pct"] for row in summaries], color="#267a92")
    axes[0].set(ylabel="Project execution match (%)", ylim=(0, 100), title="Same model, same questions" + suffix)
    for i, row in enumerate(summaries):
        axes[0].text(i, row["project_ex_pct"] + 1, f"{row['project_ex_pct']:.1f}", ha="center")
    variants = summaries[1:]
    positions = np.arange(len(variants))
    axes[1].bar(positions - .18, [row["gained"] for row in variants], width=.36, label="Gained", color="#26936b")
    axes[1].bar(positions + .18, [row["lost"] for row in variants], width=.36, label="Lost", color="#c65e55")
    axes[1].set(xticks=positions, xticklabels=names[1:], ylabel="Questions", title="Paired changes vs full schema")
    axes[1].legend()
    save(fig, "accuracy_and_transitions")

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, row in enumerate(variants):
        ax.plot([row["ex_delta_ci95_low_pp"], row["ex_delta_ci95_high_pp"]], [i, i], color="#267a92", linewidth=2)
        ax.scatter(row["ex_delta_pp"], i, color="#267a92")
    ax.axvline(0, color="gray", linestyle="--")
    ax.set(yticks=positions, yticklabels=names[1:], xlabel="Execution match difference (percentage points)",
           title=f"Paired 95% bootstrap CI (unit={report['bootstrap']['unit']})" + suffix)
    save(fig, "accuracy_delta_ci")

    errors = sorted({row["error_type"] for run in runs for row in run["rows"].values()})
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(runs))
    for error in errors:
        counts = np.array([sum(row["error_type"] == error for row in run["rows"].values()) for run in runs])
        ax.bar(names, counts, bottom=bottom, label=error)
        bottom += counts
    ax.set(ylabel="Questions", title="Outcomes and errors (none = correct)" + suffix)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, "error_breakdown")

    for metric, label in METRICS.items():
        datasets = [[row[metric] for row in run["rows"].values() if row[metric] is not None] for run in runs]
        if not any(datasets):
            skipped.append(f"{metric}: no telemetry")
            for extension in ("png", "pdf"):
                (output / f"{metric}.{extension}").unlink(missing_ok=True)
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        available = [(name, data) for name, data in zip(names, datasets) if data]
        axes[0].boxplot([data for _, data in available], tick_labels=[f"{name}\nn={len(data)}" for name, data in available], showfliers=False)
        axes[0].set(ylabel=label, title="Distribution (outliers hidden)" + suffix)
        for name, data in available:
            sorted_values = np.sort(data)
            axes[1].step(sorted_values, np.arange(1, len(data) + 1) / len(data), label=f"{name} (n={len(data)})")
        axes[1].set(xlabel=label, ylabel="Fraction of questions", title="Empirical cumulative distribution")
        axes[1].legend()
        save(fig, metric)

    fig, ax = plt.subplots(figsize=(6, 5))
    plotted = False
    for run in runs[1:]:
        points = [(runs[0]["rows"][key]["elapsed_ms"], row["elapsed_ms"]) for key, row in run["rows"].items()
                  if runs[0]["rows"][key]["elapsed_ms"] is not None and row["elapsed_ms"] is not None]
        if points:
            x, y = zip(*points)
            ax.scatter(x, y, alpha=.45, s=15, label=run["name"])
            plotted = True
    if plotted:
        maximum = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, maximum], [0, maximum], "--", color="gray", label="Equal latency")
        ax.set(xlabel="Full-schema end-to-end latency (ms)", ylabel="Pruning end-to-end latency (ms)",
               title="Paired latency: below diagonal is faster" + suffix)
        ax.legend()
        save(fig, "paired_latency")
    else:
        plt.close(fig)
        skipped.append("paired_latency: no paired latency telemetry")
        for extension in ("png", "pdf"):
            (output / f"paired_latency.{extension}").unlink(missing_ok=True)
    (output / "plot_notes.json").write_text(json.dumps({"skipped": skipped}, indent=2), encoding="utf-8")
    return skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="name=run_directory; full baseline first")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-unit", choices=["question", "db"], default="question")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    runs = []
    for item in args.run:
        name, separator, directory = item.partition("=")
        if not separator or not name or not directory:
            parser.error("--run must be name=directory")
        if any(run["name"] == name for run in runs):
            parser.error(f"Duplicate run name: {name}")
        runs.append(read_run(name, Path(directory)))
    if len(runs) < 2:
        parser.error("Supply a full baseline and at least one pruning run")
    report = compare(runs, args.output, args.bootstrap_samples, args.seed, args.bootstrap_unit)
    if report["provenance_warning"]:
        print("WARNING:", report["provenance_warning"])
    if not args.no_plots:
        skipped = plot_runs(runs, report, args.output, args.pdf)
        for note in skipped:
            print(f"Skipped {note}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
