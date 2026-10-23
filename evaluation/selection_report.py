"""Tables (Markdown) and figures (PNG) from selection_study's output directory."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MAIN = {
    "vote (baseline)": "Vote (baseline)",
    "CAV (3 checks)": "CAV-3",
    "CAV without degenerate": "CAV-2 (grounding + join)",
    "learned re-ranker (10-fold CV)": "Learned re-ranker (CV)",
}
COLORS = {"Vote (baseline)": "#8c8b86", "CAV-3": "#2a78d6",
          "CAV-2 (grounding + join)": "#eb6834", "Learned re-ranker (CV)": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CHECK_NAMES = {"degenerate": "Kết quả rỗng / toàn NULL", "ungrounded": "Literal không có trong DB",
               "bad_join": "JOIN không theo khóa ngoại", "duplicates": "Dòng trùng lặp (chỉ dùng cho re-ranker)"}


def _pct(v) -> str:
    return f"{v * 100:.1f}"


def _p(v) -> str:
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def tables(frame: pd.DataFrame, meta: dict) -> str:
    main = frame[frame.method.isin(MAIN)].copy()
    main["name"] = main.method.map(MAIN)
    head = ("| Ngân sách | Phương pháp | EX % [CI95] | Δ EX (điểm) [CI95] | McNemar −/+ (p) | Ứng viên TB "
            "| Độ trễ (s) | Thu hẹp khoảng cách oracle | Easy | Medium | Hard |\n|---|---|---|---|---|---|---|---|---|---|---|")
    lines = ["## Bảng 1. So sánh phương pháp chọn SQL (cùng pool ứng viên)", "", head]
    for r in main.itertuples():
        delta = "—" if r.name == "Vote (baseline)" else (
            f"{r.delta_vs_baseline * 100:+.1f} [{r.delta_ci95_low * 100:+.1f}, {r.delta_ci95_high * 100:+.1f}]")
        test = "—" if r.name == "Vote (baseline)" else f"{r.mcnemar_lost}/{r.mcnemar_gained} ({_p(r.mcnemar_p)})"
        gap = "—" if r.name == "Vote (baseline)" else f"{r.oracle_gap_closed * 100:.0f}%"
        name = f"**{r.name}**" if r.name != "Vote (baseline)" else r.name
        lines.append(f"| {r.budget} | {name} | {_pct(r.execution_accuracy)} [{_pct(r.ci95_low)}–{_pct(r.ci95_high)}] "
                     f"| {delta} | {test} | {r.average_candidates:.2f} | {r.est_latency_s:.1f} | {gap} "
                     f"| {_pct(r.acc_easy)} | {_pct(r.acc_medium)} | {_pct(r.acc_hard)} |")
    lines += ["", "Δ và CI95: paired bootstrap trên câu hỏi; McNemar exact trên thứ tự ứng viên gốc "
              "(−: câu baseline đúng nhưng phương pháp sai, +: ngược lại). "
              "Thu hẹp khoảng cách oracle = Δ / (oracle@ngân sách − baseline).", ""]

    lines += ["## Bảng 2. Ablation từng kiểm tra (Δ EX so với vote, điểm %)", ""]
    ablation = frame[frame.method.str.startswith("CAV")].copy()
    ablation = ablation[ablation.method != "CAV (3 checks) + CAV stopping"]
    pivot = ablation.pivot_table(index="method", columns="budget", values="delta_vs_baseline", sort=False) * 100
    budgets = list(dict.fromkeys(frame.budget))
    lines.append("| Cấu hình | " + " | ".join(budgets) + " |")
    lines.append("|---|" + "---|" * len(budgets))
    for method, row in pivot.iterrows():
        lines.append(f"| {method} | " + " | ".join(f"{row[b]:+.1f}" for b in budgets) + " |")

    lines += ["", "## Bảng 3. Độ tin cậy của từng kiểm tra (trên mọi ứng viên chạy được)", "",
              "| Kiểm tra | Tỷ lệ kích hoạt | Trên ứng viên sai | Trên ứng viên đúng | Precision (cờ ⇒ sai) |",
              "|---|---|---|---|---|"]
    diag = meta["diagnostics"]
    for key, label in CHECK_NAMES.items():
        d = diag[key]
        lines.append(f"| {label} | {_pct(d['fire_rate'])}% | {_pct(d['fire_rate_on_wrong'])}% "
                     f"| {_pct(d['fire_rate_on_correct'])}% | {_pct(d['precision_wrong'] or 0)}% |")

    stop = frame[frame.budget.str.startswith("adaptive") & frame.method.isin(
        ["vote (baseline)", "CAV (3 checks)", "CAV (3 checks) + CAV stopping"])]
    lines += ["", "## Bảng 4. Dùng CAV cho cả điều kiện dừng sớm (adaptive)", "",
              "| Ngân sách | Phương pháp | EX % | Ứng viên TB | Độ trễ (s) | Dừng sớm % |", "|---|---|---|---|---|---|"]
    for r in stop.itertuples():
        lines.append(f"| {r.budget} | {r.method} | {_pct(r.execution_accuracy)} | {r.average_candidates:.2f} "
                     f"| {r.est_latency_s:.1f} | {_pct(r.early_stop_rate)} |")

    split = main[main.name != "Vote (baseline)"]
    base = main[main.name == "Vote (baseline)"].set_index("budget")
    lines += ["", "## Bảng 5. Kiểm tra tính ổn định: Δ EX trên hai nửa dữ liệu (câu lẻ / chẵn)", "",
              "| Ngân sách | Phương pháp | Δ nửa lẻ | Δ nửa chẵn |", "|---|---|---|---|"]
    for r in split.itertuples():
        b = base.loc[r.budget]
        lines.append(f"| {r.budget} | {r.name} | {(r.acc_half_odd - b.acc_half_odd) * 100:+.1f} "
                     f"| {(r.acc_half_even - b.acc_half_even) * 100:+.1f} |")
    lines += ["", f"Dữ liệu: {meta['questions']} câu, pool {meta['pool_size']} ứng viên/câu, "
              f"{meta['permutations']} hoán vị thứ tự, {meta['bootstrap_resamples']} lần bootstrap. "
              f"Chi phí kiểm tra: {meta['check_overhead_ms_per_candidate']:.2f} ms/ứng viên."]
    return "\n".join(lines)


def fig_bars(frame, out):
    main = frame[frame.method.isin(MAIN)].copy()
    main["name"] = main.method.map(MAIN)
    budgets = list(dict.fromkeys(main.budget))
    names = list(MAIN.values())
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    width = 0.19
    for i, name in enumerate(names):
        sub = main[main.name == name].set_index("budget").loc[budgets]
        xs = [b + (i - 1.5) * (width + 0.015) for b in range(len(budgets))]
        ax.bar(xs, sub.execution_accuracy * 100, width, color=COLORS[name], label=name,
               edgecolor="white", linewidth=1)
        ax.errorbar(xs, sub.execution_accuracy * 100,
                    yerr=[(sub.execution_accuracy - sub.ci95_low) * 100, (sub.ci95_high - sub.execution_accuracy) * 100],
                    fmt="none", ecolor=MUTED, elinewidth=0.9, capsize=2)
        if name in ("Vote (baseline)", "Learned re-ranker (CV)"):
            for x, v in zip(xs, sub.execution_accuracy * 100):
                ax.text(x, 31, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color="white", rotation=90)
    oracle = main[main.name == "Vote (baseline)"].set_index("budget").loc[budgets].oracle_at_budget * 100
    for b, v in enumerate(oracle):
        ax.hlines(v, b - 0.45, b + 0.45, colors=INK, linestyles=(0, (3, 2)), lw=1)
        ax.text(b, v + 0.8, f"oracle {v:.0f}", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(budgets)), [b.replace(" τ", "\nτ") for b in budgets])
    ax.set_ylim(30, 95)
    ax.set_ylabel("Execution accuracy (%)")
    ax.set_title("Chọn SQL từ cùng một pool ứng viên: vote so với CAV và re-ranker",
                 loc="left", fontsize=11, color=INK)
    ax.legend(ncols=4, fontsize=8.5, frameon=False, loc="upper left")
    _style(ax)
    fig.savefig(out / "fig_selection_accuracy.png", dpi=200)
    plt.close(fig)


def fig_pareto(frame, sweep_csv, out):
    main = frame[frame.method.isin(MAIN)].copy()
    main["name"] = main.method.map(MAIN)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for name in MAIN.values():
        sub = main[main.name == name]
        fixed = sub[sub.budget.str.startswith("fixed")].sort_values("average_candidates")
        adaptive = sub[sub.budget.str.startswith("adaptive")]
        ax.plot(fixed.average_candidates, fixed.execution_accuracy * 100, color=COLORS[name], lw=2,
                marker="s", ms=6, label=f"{name} — fixed K")
        ax.scatter(adaptive.average_candidates, adaptive.execution_accuracy * 100, color=COLORS[name],
                   marker="o", s=60, edgecolor="white", linewidth=1.5, zorder=4)
    if sweep_csv and sweep_csv.exists():
        greedy = pd.read_csv(sweep_csv).query("mode == 'greedy'").iloc[0]
        ax.scatter([1], [greedy.execution_accuracy * 100], marker="*", s=160, color=INK, zorder=5)
        ax.annotate("Greedy", (1, greedy.execution_accuracy * 100), xytext=(6, -4),
                    textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.scatter([], [], marker="o", color=MUTED, label="adaptive 4-4-8 (τ=0.5, 0.75)")
    ax.set(xlabel="Số ứng viên sinh trung bình / câu", ylabel="Execution accuracy (%)")
    ax.set_title("Độ chính xác theo chi phí sinh", loc="left", fontsize=11, color=INK)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    fig.savefig(out / "fig_selection_pareto.png", dpi=200)
    plt.close(fig)


def fig_checks(meta, out):
    diag = meta["diagnostics"]
    keys = list(CHECK_NAMES)
    fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    ys = range(len(keys))
    wrong = [diag[k]["fire_rate_on_wrong"] * 100 for k in keys]
    right = [diag[k]["fire_rate_on_correct"] * 100 for k in keys]
    ax.barh([y - 0.18 for y in ys], wrong, 0.34, color="#eb6834", label="Ứng viên sai", edgecolor="white")
    ax.barh([y + 0.18 for y in ys], right, 0.34, color="#2a78d6", label="Ứng viên đúng", edgecolor="white")
    for y, w, r in zip(ys, wrong, right):
        ax.text(w + 0.3, y - 0.18, f"{w:.1f}%", va="center", fontsize=8, color=INK)
        ax.text(r + 0.3, y + 0.18, f"{r:.1f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(list(ys), [CHECK_NAMES[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlabel("Tỷ lệ ứng viên bị gắn cờ (%)")
    ax.set_title("Các kiểm tra gần như chỉ gắn cờ ứng viên sai", loc="left", fontsize=11, color=INK)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    fig.savefig(out / "fig_check_diagnostics.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--sweep-csv", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.directory / "selection.csv")
    meta = json.loads((args.directory / "meta.json").read_text(encoding="utf-8"))
    (args.directory / "tables.md").write_text(tables(frame, meta), encoding="utf-8")
    fig_bars(frame, args.directory)
    fig_pareto(frame, args.sweep_csv, args.directory)
    fig_checks(meta, args.directory)
    print(f"Tables and figures saved in {args.directory}")


if __name__ == "__main__":
    main()
