"""Accuracy / cost trade-off charts from run_sweep's sweep.csv."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    frame["threshold"] = pd.to_numeric(frame["threshold"], errors="coerce")
    fixed = frame[frame["mode"] == "fixed_sc"].sort_values("average_candidates")
    greedy = frame[frame["mode"] == "greedy"].iloc[0]
    oracle = frame[frame["mode"] == "oracle"]
    for x, xlabel, name in [("average_candidates", "Average sampled candidates per question", "sweep_candidates"),
                            ("est_latency_s", "Estimated latency per question (s)", "sweep_latency")]:
        fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
        ax.errorbar(fixed[x], fixed.execution_accuracy * 100,
                    yerr=[(fixed.execution_accuracy - fixed.ci95_low) * 100,
                          (fixed.ci95_high - fixed.execution_accuracy) * 100],
                    marker="s", color="#475569", capsize=3, label="Fixed-K (K=1,2,4,8,16)")
        adaptive = frame[frame["mode"] == "adaptive_sc"].copy()
        adaptive["min_valid"] = pd.to_numeric(adaptive["min_valid"]).astype(int)
        # Schedules whose total exceeds the pool collapse onto shorter ones; plot distinct curves only.
        adaptive = adaptive.drop_duplicates(["min_valid", "threshold", "execution_accuracy", x])
        styles = {2: "--", 4: "-"}
        colors = dict(zip(sorted(adaptive["schedule"].unique()), ["#2563eb", "#16a34a", "#d97706"]))
        for (schedule, min_valid), group in adaptive.groupby(["schedule", "min_valid"]):
            group = group.sort_values("threshold")
            if len(group) < 2:
                continue
            ax.plot(group[x], group.execution_accuracy * 100, marker="o", ls=styles.get(min_valid, "-"),
                    color=colors[schedule], label=f"Adaptive {schedule}, min_valid={min_valid} (τ=0.5→1.0)")
        ax.scatter([greedy[x]], [greedy.execution_accuracy * 100], marker="*", s=180, color="#dc2626",
                   label="Greedy", zorder=5)
        if not oracle.empty:
            ax.axhline(oracle.execution_accuracy.iloc[0] * 100, ls="--", color="#94a3b8", label="Oracle pass@16")
        ax.set(xlabel=xlabel, ylabel="Execution accuracy (%)", title="Accuracy vs. cost")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="lower right")
        fig.savefig(args.csv.parent / f"{name}.png", dpi=160)
        plt.close(fig)
    print(f"Charts saved in {args.csv.parent}")


if __name__ == "__main__":
    main()
