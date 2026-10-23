"""Render run_sweep's sweep.csv as Markdown tables for the README."""
import argparse
import json
from pathlib import Path

import pandas as pd


def _pct(value) -> str:
    return "" if pd.isna(value) or value == "" else f"{float(value) * 100:.1f}"


def _row(r) -> str:
    return (f"| {r.method} | {_pct(r.execution_accuracy)} [{_pct(r.ci95_low)}–{_pct(r.ci95_high)}] "
            f"| {r.average_candidates:.2f} | {r.est_latency_s:.1f} | {_pct(r.early_stop_rate)} "
            f"| {_pct(r.acc_easy)} | {_pct(r.acc_medium)} | {_pct(r.acc_hard)} |")


def render(directory: Path, top: int = 8) -> str:
    frame = pd.read_csv(directory / "sweep.csv")
    stats = json.loads((directory / "pool_stats.json").read_text(encoding="utf-8"))
    header = ("| Cấu hình | EX % [CI95] | Ứng viên TB | Độ trễ ước tính (s) | Dừng sớm % | Easy % | Medium % | Hard % |\n"
              "|---|---|---|---|---|---|---|---|")
    main = frame[frame["mode"].isin(["greedy", "fixed_sc", "oracle"])]
    adaptive = frame[frame["mode"] == "adaptive_sc"]
    best = adaptive.sort_values(["execution_accuracy", "average_candidates"], ascending=[False, True]).head(top)
    cheapest = adaptive.sort_values(["average_candidates", "execution_accuracy"], ascending=[True, False]).head(3)
    lines = [
        f"Pool: {stats['questions']} câu; tỷ lệ ứng viên chạy được {stats['pool_valid_rate']:.1%}, "
        f"đúng {stats['pool_correct_rate']:.1%}; trung bình {stats['avg_distinct_results']:.2f} kết quả khác nhau/câu; "
        f"greedy {stats['greedy_avg_s']:.1f}s/câu, một batch sampling {stats['batch_avg_s']:.1f}s.",
        "", "**Greedy, fixed-K và oracle**", "", header, *[_row(r) for r in main.itertuples()],
        "", f"**Adaptive — {top} cấu hình chính xác nhất** (tên: lịch vòng, ngưỡng τ, min_valid)", "",
        header, *[_row(r) for r in best.itertuples()],
        "", "**Adaptive — 3 cấu hình rẻ nhất**", "", header, *[_row(r) for r in cheapest.itertuples()],
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(render(args.directory))


if __name__ == "__main__":
    main()
