"""Sweep greedy / fixed-K / adaptive settings from one shared candidate pool per question.

Phase `collect` runs the model once per question: one greedy candidate plus a pool of N
sampled candidates, each validated, executed and checked against gold. Phase `simulate`
replays every configuration on that pool with the project's own grouping, ranking and
stopping rules, so all methods see identical samples and the sweep costs one model pass.
Bootstrap permutations of the pool give confidence intervals for the sampled methods.
Correction is disabled to isolate the effect of sampling and stopping.
"""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from core.adaptive_sampler import decide
from core.candidate_ranker import rank_candidates
from core.config import ROOT, load_config
from core.model_backend import create_backend
from core.pipeline import Pipeline
from core.prompt_builder import build_prompt
from core.result_grouper import group_candidates
from core.schema_inspector import SchemaInspector
from core.sql_executor import SQLExecutor
from core.sql_generator import generate_candidates
from core.types import Candidate, ExecutionResult, ValidationResult
from databases.create_student_db import DEFAULT_DB, create_database
from evaluation.metrics import execution_match

FIXED_K = [1, 2, 4, 8, 16]
SCHEDULES = {"2-2-4-8": [2, 2, 4, 8], "4-4-8": [4, 4, 8], "4-4-8-16": [4, 4, 8, 16]}
THRESHOLDS = [0.5, 0.6, 0.75, 0.9, 1.0]
MIN_VALID = [2, 4]


def _record(candidate, gold, gold_sql) -> dict:
    execution, validation = candidate.execution, candidate.validation
    return {
        "sql": candidate.sql, "error": candidate.error,
        "success": bool(execution and execution.success),
        "result": execution.normalized_result if execution and execution.success else None,
        "schema_score": validation.schema_score if validation else 0.0,
        "warnings": len(validation.warnings) if validation else 0,
        "error_type": (execution.error_type if execution and execution.error_type else
                       validation.error_type if validation else "generation"),
        "correct": execution_match(execution, gold, gold_sql),
        "generation_ms": candidate.generation_ms,
    }


def collect(args, config):
    dataset = args.dataset.resolve()
    items = json.loads(dataset.read_text(encoding="utf-8"))[: args.limit]
    if args.db.resolve() == DEFAULT_DB.resolve():
        create_database()
    out = args.output / "pool.jsonl"
    # A resumed pool must have the same data, retrieval and generation settings.
    config["correction"]["enabled"] = False
    manifest = {"dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "config": config, "pool": args.pool, "batch_size": args.batch_size,
                "default_database": str(args.db.resolve()), "collector_version": 2}
    manifest_path = args.output / "collection_manifest.json"
    if out.exists() and out.stat().st_size:
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Candidate pool settings differ; use a new output directory")
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line}
    config["correction"]["enabled"] = False
    backend = create_backend(config)
    pipeline = Pipeline(config, backend)
    executor = SQLExecutor(config["execution"])
    schema_inspector = SchemaInspector(**config["schema"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with out.open("a", encoding="utf-8") as handle:
        for index, item in enumerate(items, 1):
            if item["id"] in done:
                continue
            database = (dataset.parent / item["db_path"]).resolve() if item.get("db_path") else args.db
            gold = executor.execute(database, item["gold_sql"])
            if not gold.success:
                raise ValueError(f"Gold SQL failed for {item['id']}: {gold.error}")
            schema = schema_inspector.inspect(database)
            if config["validation"]["schema_selection"]:
                schema = pipeline.schema_selector.select(item["question"], schema, item.get("evidence", ""))
                retrieval = dict(pipeline.schema_selector.last_metadata)
            else:
                retrieval = {"mode": "full", "latency_ms": 0.0}
            prompt = build_prompt(item["question"], schema, item.get("evidence", ""))
            greedy_params = dict(config["generation"], temperature=0.0)
            sample_params = dict(config["generation"], batch_size=args.batch_size)
            backend.start_request(config["seed"])
            start = time.perf_counter()
            greedy = generate_candidates(backend, prompt, 1, greedy_params)
            greedy_ms = (time.perf_counter() - start) * 1000
            pool, batch_ms = [], []
            while len(pool) < args.pool:
                start = time.perf_counter()
                new = generate_candidates(backend, prompt, min(args.batch_size, args.pool - len(pool)),
                                          sample_params, len(pool))
                batch_ms.append((time.perf_counter() - start) * 1000)
                if not new:
                    break
                pool += new
            for candidate in greedy + pool:
                pipeline._check(candidate, database)
            row = {"id": item["id"], "difficulty": item.get("difficulty"), "question": item["question"],
                   "gold_sql": item["gold_sql"], "greedy_ms": greedy_ms, "batch_ms": batch_ms,
                   "batch_size": args.batch_size, "db_id": item.get("db_id"),
                   "db_path": str(database), "schema_retrieval": retrieval,
                   "greedy": _record(greedy[0] if greedy else Candidate(0, error="No generation"), gold, item["gold_sql"]),
                   "pool": [_record(c, gold, item["gold_sql"]) for c in pool]}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(items)}] {item['id']} greedy={row['greedy']['correct']} "
                  f"pool_correct={sum(r['correct'] for r in row['pool'])}/{len(pool)} "
                  f"{(greedy_ms + sum(batch_ms)) / 1000:.0f}s", flush=True)


def _candidates(records):
    return [Candidate(i, sql=r["sql"] or "", error=r["error"],
                      validation=ValidationResult(is_safe=True, is_valid=r["success"],
                                                  warnings=[""] * r["warnings"],
                                                  schema_score=r["schema_score"]),
                      execution=ExecutionResult(success=r["success"], normalized_result=r["result"]))
            for i, r in enumerate(records)]


def _select(records, weights):
    candidates = _candidates(records)
    group_candidates(candidates)
    chosen = rank_candidates(candidates, weights)
    return records[chosen.id]["correct"] if chosen else False


def _batch_cost(row, count):
    """Sampled-generation latency estimate: one measured batch time per started batch."""
    per_batch = statistics.mean(row["batch_ms"]) if row["batch_ms"] else 0.0
    return math.ceil(count / row["batch_size"]) * per_batch


def _adaptive(row, order, schedule, sample, weights):
    records = [row["pool"][i] for i in order]
    used, cost, reason = 0, 0.0, "rounds_exhausted"
    for requested in schedule:
        count = min(requested, sample["max_candidates"] - used, len(records) - used)
        if count <= 0:
            reason = "max_candidates"
            break
        used += count
        cost += _batch_cost(row, count)
        candidates = _candidates(records[:used])
        decision = decide(group_candidates(candidates), used, 0, sample)
        if decision.should_stop:
            reason = decision.reason
            break
    return _select(records[:used], weights), used, cost, reason


def simulate(args, config):
    rows = [json.loads(line) for line in (args.output / "pool.jsonl").read_text(encoding="utf-8").splitlines() if line]
    weights = config["ranking"]
    rng = random.Random(config["seed"])
    orders = [list(range(len(row["pool"]))) for row in rows]
    permutations = [orders] + [[rng.sample(o, len(o)) for o in orders] for _ in range(args.bootstrap)]
    settings = [("greedy", "greedy", {})]
    settings += [(f"fixed_k{k}", "fixed_sc", {"k": k}) for k in FIXED_K]
    settings += [(f"adaptive_{name}_t{t}_m{m}", "adaptive_sc", {"schedule": s, "threshold": t, "min_valid": m})
                 for name, s in SCHEDULES.items() for t in THRESHOLDS for m in MIN_VALID]
    settings += [("oracle_pass@16", "oracle", {})]
    summaries, per_question = [], []
    for label, mode, extra in settings:
        sample = dict(config["sampling"], max_candidates=len(rows[0]["pool"]))
        if mode == "adaptive_sc":
            sample.update(confidence_threshold=extra["threshold"], min_valid_candidates=extra["min_valid"])
        runs = []
        for perm in (permutations[:1] if mode in {"greedy", "oracle"} else permutations):
            outcomes = []
            for row, order in zip(rows, perm):
                if mode == "greedy":
                    outcomes.append((row["greedy"]["correct"], 1, row["greedy_ms"], "greedy", row["difficulty"]))
                elif mode == "oracle":
                    outcomes.append((any(r["correct"] for r in row["pool"]), len(row["pool"]),
                                     _batch_cost(row, len(row["pool"])), "oracle", row["difficulty"]))
                elif mode == "fixed_sc":
                    records = [row["pool"][i] for i in order[: extra["k"]]]
                    outcomes.append((_select(records, weights), len(records),
                                     _batch_cost(row, len(records)), "fixed", row["difficulty"]))
                else:
                    correct, used, cost, reason = _adaptive(row, order, extra["schedule"], sample, weights)
                    outcomes.append((correct, used, cost, reason, row["difficulty"]))
            runs.append(outcomes)
        first = runs[0]
        accuracies = sorted(sum(o[0] for o in run) / len(run) for run in runs)
        candidates = [statistics.mean(o[1] for o in run) for run in runs]
        latencies = [statistics.mean(o[2] for o in run) for run in runs]
        summary = {
            "method": label, "mode": mode,
            "schedule": "-".join(map(str, extra.get("schedule", []))) or extra.get("k", ""),
            "threshold": extra.get("threshold", ""), "min_valid": extra.get("min_valid", ""),
            "questions": len(rows),
            "execution_accuracy": statistics.mean(accuracies),
            "ci95_low": accuracies[int(0.025 * (len(accuracies) - 1))],
            "ci95_high": accuracies[int(0.975 * (len(accuracies) - 1))],
            "average_candidates": statistics.mean(candidates),
            "est_latency_s": statistics.mean(latencies) / 1000,
            "early_stop_rate": statistics.mean(
                sum(o[3] == "consensus" for o in run) / len(run) for run in runs),
        }
        for level in ("easy", "medium", "hard"):
            subset = [o for run in runs for o in run if o[4] == level]
            summary[f"acc_{level}"] = sum(o[0] for o in subset) / len(subset) if subset else ""
        summaries.append(summary)
        per_question.append({"method": label, **{row["id"]: o[0] for row, o in zip(rows, first)}})
    with (args.output / "sweep.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (args.output / "per_question.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_question[0]))
        writer.writeheader()
        writer.writerows(per_question)
    pool_stats = {
        "questions": len(rows),
        "pool_valid_rate": statistics.mean(sum(r["success"] for r in row["pool"]) / len(row["pool"]) for row in rows),
        "pool_correct_rate": statistics.mean(sum(r["correct"] for r in row["pool"]) / len(row["pool"]) for row in rows),
        "avg_distinct_results": statistics.mean(len({r["result"] for r in row["pool"] if r["success"]}) for row in rows),
        "greedy_avg_s": statistics.mean(row["greedy_ms"] for row in rows) / 1000,
        "batch_avg_s": statistics.mean(statistics.mean(row["batch_ms"]) for row in rows) / 1000,
    }
    (args.output / "pool_stats.json").write_text(json.dumps(pool_stats, indent=2), encoding="utf-8")
    print(json.dumps(pool_stats, indent=2))
    print(f"Saved: {args.output / 'sweep.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["collect", "simulate", "all"])
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/evaluation_questions.json")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pool", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "sweep")
    args = parser.parse_args()
    config = deepcopy(load_config(args.config))
    if args.phase in {"collect", "all"}:
        collect(args, config)
    if args.phase in {"simulate", "all"}:
        simulate(args, config)


if __name__ == "__main__":
    main()
