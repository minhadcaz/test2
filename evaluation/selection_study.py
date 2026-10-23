"""Offline study of candidate selection on run_sweep's pool.jsonl (no model calls).

Every pooled SQL is re-executed once on the database to compute gold-free consistency flags
(core.consistency_checks). Selectors are then replayed on the same candidate orders as the
sweep: the baseline execution vote, the vote restricted to the least-flagged candidates
(constraint-aware voting, CAV) with a leave-one-check-out ablation, and a cross-validated
logistic re-ranker. Uncertainty: bootstrap over questions (paired for deltas) plus an exact
McNemar test on the canonical candidate order.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics
import time

import numpy as np

from core.adaptive_sampler import decide
from core.config import ROOT, load_config
from core.consistency_checks import CHECKS, ConsistencyChecker
from core.result_grouper import group_candidates
from core.sql_executor import SQLExecutor
from databases.create_student_db import DEFAULT_DB, create_database
from evaluation.run_sweep import _batch_cost, _candidates, _select

BUDGETS = [("fixed", 4), ("fixed", 8), ("fixed", 16),
           ("adaptive", ([4, 4, 8], 0.5, 2)), ("adaptive", ([4, 4, 8], 0.75, 2))]
FEATURES = ["vote", "log_size", "degenerate", "ungrounded", "bad_join", "duplicates",
            "log_rows", "columns", "sql_length"]


def enrich(rows, db, config, flags_path):
    """Attach flags to every successful candidate (once per distinct SQL) and time the checks."""
    cache = {}
    executor = SQLExecutor(dict(config["execution"], max_rows=config["execution"]["max_result_rows"]))
    checker = ConsistencyChecker(db)
    timings = []
    for row in rows:
        for record in [row["greedy"], *row["pool"]]:
            if not record["success"]:
                continue
            if record["sql"] not in cache:
                result = executor.execute(db, record["sql"])
                start = time.perf_counter()
                flags = checker.check(record["sql"], result.rows)
                timings.append((time.perf_counter() - start) * 1000)
                tuples = [tuple(map(str, r)) for r in result.rows]
                cache[record["sql"]] = flags | {
                    "duplicates": len(tuples) != len(set(tuples)), "rows": result.row_count,
                    "columns": len(result.columns)}
            record.update(cache[record["sql"]])
    flags_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return statistics.mean(timings) if timings else None


def _flags(record, checks):
    return sum(bool(record.get(c)) for c in checks)


def _clean(records, checks):
    """Successful candidates with the fewest flags; falls back gracefully when all are flagged."""
    ok = [r for r in records if r["success"]]
    if not ok or not checks:
        return ok
    best = min(_flags(r, checks) for r in ok)
    return [r for r in ok if _flags(r, checks) == best]


def _group_features(records):
    ok = [r for r in records if r["success"]]
    groups = {}
    for r in ok:
        groups.setdefault(r["result"], []).append(r)
    out = []
    for members in groups.values():
        head = members[0]
        out.append((members, [
            len(members) / len(ok), math.log1p(len(members)), head["degenerate"],
            statistics.mean(m["ungrounded"] for m in members), statistics.mean(m["bad_join"] for m in members),
            head["duplicates"], math.log1p(head["rows"]), head["columns"] / 5,
            statistics.mean(len(m["sql"]) for m in members) / 500]))
    return out


class LogisticRanker:
    def __init__(self, l2=1.0, steps=3000, lr=0.1):
        self.l2, self.steps, self.lr = l2, steps, lr

    def fit(self, x, y):
        x = np.asarray(x, float)
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        x = np.c_[(x - self.mean) / self.std, np.ones(len(x))]
        y = np.asarray(y, float)
        self.w = np.zeros(x.shape[1])
        for _ in range(self.steps):
            p = 1 / (1 + np.exp(-x @ self.w))
            grad = x.T @ (p - y) / len(y) + self.l2 * np.r_[self.w[:-1], 0] / len(y)
            self.w -= self.lr * grad
        return self

    def score(self, x):
        x = (np.asarray(x, float) - self.mean) / self.std
        return np.c_[x, np.ones(len(x))] @ self.w


def train_rankers(rows, folds, seed):
    """One ranker per question fold, trained on the full 16-candidate pools of the other folds."""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    fold_of = {q: i % folds for i, q in enumerate(order)}
    rankers = {}
    for fold in range(folds):
        x, y = [], []
        for q, row in enumerate(rows):
            if fold_of[q] != fold:
                for members, features in _group_features(row["pool"]):
                    x.append(features)
                    y.append(any(m["correct"] for m in members))
        rankers[fold] = LogisticRanker().fit(x, y)
    return {q: rankers[fold_of[q]] for q in range(len(rows))}, rankers


def make_selectors(weights, rankers):
    def vote(records, _q):
        return _select(records, weights)

    def cav(checks):
        return lambda records, _q: _select(_clean(records, checks), weights)

    def learned(records, q):
        groups = _group_features(records)
        if not groups:
            return False
        scores = rankers[q].score([f for _, f in groups])
        return any(m["correct"] for m in groups[int(np.argmax(scores))][0])

    selectors = {"vote (baseline)": vote, "CAV (3 checks)": cav(CHECKS)}
    for check in CHECKS:
        selectors[f"CAV only {check}"] = cav([check])
        selectors[f"CAV without {check}"] = cav([c for c in CHECKS if c != check])
    selectors["learned re-ranker (10-fold CV)"] = learned
    return selectors


def run_budget(row, order, budget, sample, select, q, stop_checks):
    records = [row["pool"][i] for i in order]
    kind, spec = budget
    if kind == "fixed":
        used = records[:spec]
        return select(used, q), len(used), _batch_cost(row, len(used)), False
    schedule, threshold, min_valid = spec
    sample = dict(sample, confidence_threshold=threshold, min_valid_candidates=min_valid)
    count, cost, stopped = 0, 0.0, False
    for requested in schedule:
        step = min(requested, len(records) - count)
        if step <= 0:
            break
        count += step
        cost += _batch_cost(row, step)
        voters = _clean(records[:count], stop_checks)
        decision = decide(group_candidates(_candidates(voters)), count, 0, sample)
        if decision.should_stop:
            stopped = decision.reason == "consensus"
            break
    return select(records[:count], q), count, cost, stopped


def _budget_name(budget):
    kind, spec = budget
    return f"fixed K={spec}" if kind == "fixed" else f"adaptive {'-'.join(map(str, spec[0]))} τ={spec[1]} m={spec[2]}"


def bootstrap(values, rng, resamples):
    n = len(values)
    means = sorted(statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1], means


def mcnemar(base, new):
    b = sum(x and not y for x, y in zip(base, new))
    c = sum(y and not x for x, y in zip(base, new))
    n = b + c
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
    return b, c, p


def study(args):
    config = load_config(args.config)
    rows = [json.loads(line) for line in (args.pool_dir / "pool.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if args.db.resolve() == DEFAULT_DB.resolve():
        create_database()
    args.output.mkdir(parents=True, exist_ok=True)
    check_ms = enrich(rows, args.db, config, args.output / "candidate_flags.json")
    rankers, fold_rankers = train_rankers(rows, args.folds, config["seed"])
    rng = random.Random(config["seed"])
    orders = [list(range(len(row["pool"]))) for row in rows]
    permutations = [orders] + [[rng.sample(o, len(o)) for o in orders] for _ in range(args.permutations)]
    selectors = make_selectors(config["ranking"], rankers)
    sample = dict(config["sampling"], max_candidates=len(rows[0]["pool"]))
    boot_rng = random.Random(config["seed"] + 1)
    summary, per_question = [], []
    for budget in BUDGETS:
        base_q = None
        for name, select in selectors.items():
            variants = [(name, [])]
            if budget[0] == "adaptive" and name == "CAV (3 checks)":
                variants.append(("CAV (3 checks) + CAV stopping", list(CHECKS)))
            for label, stop_checks in variants:
                outcomes = [[run_budget(row, order, budget, sample, select, q, stop_checks)
                             for q, (row, order) in enumerate(zip(rows, perm))] for perm in permutations]
                per_q = [statistics.fmean(run[q][0] for run in outcomes) for q in range(len(rows))]
                canonical = [o[0] for o in outcomes[0]]
                oracle = statistics.fmean(
                    any(row["pool"][i]["correct"] for i in perm[q][:run[q][1]])
                    for perm, run in zip(permutations, outcomes) for q, row in enumerate(rows))
                low, high, _ = bootstrap(per_q, boot_rng, args.resamples)
                record = {
                    "budget": _budget_name(budget), "method": label,
                    "execution_accuracy": statistics.fmean(per_q), "ci95_low": low, "ci95_high": high,
                    "average_candidates": statistics.fmean(o[1] for run in outcomes for o in run),
                    "est_latency_s": statistics.fmean(o[2] for run in outcomes for o in run) / 1000,
                    "early_stop_rate": statistics.fmean(o[3] for run in outcomes for o in run),
                    "oracle_at_budget": oracle,
                }
                for level in ("easy", "medium", "hard"):
                    subset = [v for v, row in zip(per_q, rows) if row["difficulty"] == level]
                    record[f"acc_{level}"] = statistics.fmean(subset)
                record["acc_half_odd"] = statistics.fmean(per_q[1::2])
                record["acc_half_even"] = statistics.fmean(per_q[0::2])
                if base_q is None:
                    base_q, base_canonical = per_q, canonical
                deltas = [n - b for n, b in zip(per_q, base_q)]
                d_low, d_high, d_all = bootstrap(deltas, boot_rng, args.resamples)
                b, c, p = mcnemar(base_canonical, canonical)
                gap = record["oracle_at_budget"] - statistics.fmean(base_q)
                record |= {
                    "delta_vs_baseline": statistics.fmean(deltas), "delta_ci95_low": d_low,
                    "delta_ci95_high": d_high,
                    "p_bootstrap": min(1.0, 2 * min(sum(d <= 0 for d in d_all), sum(d >= 0 for d in d_all)) / len(d_all))
                    if any(deltas) else 1.0,
                    "mcnemar_lost": b, "mcnemar_gained": c, "mcnemar_p": p,
                    "oracle_gap_closed": statistics.fmean(deltas) / gap if gap > 0 else 0.0,
                }
                summary.append(record)
                per_question.append({"budget": record["budget"], "method": label,
                                     **{row["id"]: round(v, 4) for row, v in zip(rows, per_q)}})
                print(f"{record['budget']:<28} {label:<34} EX={record['execution_accuracy']:.3f} "
                      f"Δ={record['delta_vs_baseline']:+.3f} cand={record['average_candidates']:.2f}", flush=True)
    diagnostics = check_diagnostics(rows)
    meta = {"questions": len(rows), "pool_size": len(rows[0]["pool"]), "permutations": len(permutations),
            "bootstrap_resamples": args.resamples, "check_overhead_ms_per_candidate": check_ms,
            "ranker_weights_fold0": dict(zip(FEATURES + ["bias"], fold_rankers[0].w.round(3).tolist())),
            "diagnostics": diagnostics}
    _write_csv(args.output / "selection.csv", summary)
    _write_csv(args.output / "per_question.csv", per_question)
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


def check_diagnostics(rows):
    """How often each flag fires on correct vs wrong executed candidates (pooled)."""
    executed = [r for row in rows for r in row["pool"] if r["success"]]
    out = {"executed_candidates": len(executed)}
    for check in [*CHECKS, "duplicates"]:
        fired = [r for r in executed if r[check]]
        wrong = [r for r in executed if not r["correct"]]
        right = [r for r in executed if r["correct"]]
        out[check] = {
            "fire_rate": len(fired) / len(executed),
            "fire_rate_on_wrong": sum(r[check] for r in wrong) / len(wrong),
            "fire_rate_on_correct": sum(r[check] for r in right) / len(right),
            "precision_wrong": sum(not r["correct"] for r in fired) / len(fired) if fired else None,
        }
    return out


def _write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool_dir", type=Path, help="Directory with run_sweep's pool.jsonl")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "selection")
    study(parser.parse_args())


if __name__ == "__main__":
    main()
