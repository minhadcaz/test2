"""Visualize measured retrieval summaries without inventing model EX or GPU data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.compare_schema_runs import write_csv


def load_reports(paths: list[Path]) -> list[dict]:
    reports, labels = [], set()
    signature = None
    for path in paths:
        content = json.loads(path.read_text(encoding="utf-8"))
        for report in content["reports"]:
            current = {key: report.get(key) for key in (
                "dataset_sha256", "schemas_sha256", "token_counter", "token_scope", "use_evidence")}
            current["questions"] = report["summary"]["questions"]
            current["tokenizer_provenance"] = content.get("tokenizer_provenance")
            if signature is not None and current != signature:
                raise ValueError(f"Incompatible retrieval provenance: {path}")
            signature = current
            if report["label"] in labels:
                raise ValueError(f"Duplicate label: {report['label']}")
            labels.add(report["label"])
            if report["summary"]["label_errors"]:
                raise ValueError(f"Label errors in {report['label']}; fix before comparing coverage")
            reports.append(report)
    if not reports:
        raise ValueError("No retrieval reports")
    return reports


def plot_reports(reports: list[dict], output: Path, pdf=False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"full": "#536574", "bm25": "#247a98", "dense": "#ce8640", "hybrid": "#39936f"}
    labels = [report["label"] for report in reports]
    rows = [{"label": report["label"], "mode": report["mode"], **report["summary"]} for report in reports]
    write_csv(output / "retrieval_comparison.csv", rows)
    write_csv(output / "retrieval_by_database.csv", [
        {"label": report["label"], "db_id": db_id, **stats}
        for report in reports for db_id, stats in report["by_database"].items()])

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", bbox_inches="tight")
        if pdf:
            fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for report in reports:
        stats = report["summary"]
        x, y = stats["prompt_token_reduction_micro"] * 100, stats["all_references_covered"] * 100
        ax.scatter(x, y, s=65, color=colors.get(report["mode"], "gray"))
        ax.annotate(report["label"], (x, y), xytext=(8, 9 if report["mode"] == "hybrid" else -17),
                    textcoords="offset points", fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": "gray", "linewidth": .5})
    ax.set(xlabel="Raw prompt token reduction vs full schema (%)",
           ylabel="Questions with all gold SQL references retained (%)",
           title="Measured schema retention vs context reduction (not model EX)",
           ylim=(0, 107))
    ax.grid(alpha=.2)
    save(fig, "coverage_vs_tokens")

    fig, ax = plt.subplots(figsize=(11, 5))
    values = [row["mean_selected_prompt_tokens"] for row in rows]
    bars = ax.bar(labels, values, color=[colors.get(row["mode"], "gray") for row in rows])
    ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=3)
    ax.set(ylabel="Mean raw prompt tokens (before chat template)", title="Prompt size on official Mini-Dev metadata")
    ax.tick_params(axis="x", labelrotation=40)
    save(fig, "raw_prompt_tokens")

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(rows))
    fields = [("table_recall_macro", "Table recall (macro)"),
              ("column_recall_macro", "Column recall (macro)"),
              ("all_references_covered", "All references retained")]
    for index, (field, label) in enumerate(fields):
        ax.bar(x + (index - 1) * .25, [row[field] * 100 for row in rows], width=.25, label=label)
    ax.set(xticks=x, xticklabels=labels, ylabel="Percent", ylim=(0, 108), title="Schema linking metrics (gold SQL labels)")
    ax.tick_params(axis="x", labelrotation=40)
    ax.legend(loc="lower left")
    save(fig, "schema_recall")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, [row["retrieval_ms_mean"] for row in rows], color=[colors.get(row["mode"], "gray") for row in rows])
    ax.set(ylabel="Recorded mean retrieval latency (ms)",
           title="CPU retrieval timings from separate runs; cache/load conditions may differ")
    ax.tick_params(axis="x", labelrotation=40)
    save(fig, "retrieval_latency")

    (output / "provenance.json").write_text(json.dumps({
        "dataset_sha256": reports[0]["dataset_sha256"], "schemas_sha256": reports[0]["schemas_sha256"],
        "questions": reports[0]["summary"]["questions"], "token_scope": reports[0]["token_scope"],
        "note": "Schema coverage is not model execution accuracy. CPU timings are observational, not a controlled speedup test.",
    }, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    reports = load_reports(args.input)
    plot_reports(reports, args.output, args.pdf)
    print(f"Saved measured retrieval plots to {args.output}")


if __name__ == "__main__":
    main()
