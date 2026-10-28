"""CLI for held-out evaluation; gold SQL never enters the model prompt."""
import argparse
import csv
import hashlib
import json
import logging
import statistics
from pathlib import Path
import time

import sqlglot

from core.config import ROOT, load_config
from core.model_backend import create_backend
from core.pipeline import Pipeline
from core.sql_executor import SQLExecutor
from databases.create_student_db import DEFAULT_DB, create_database
from evaluation.metrics import execution_match, summarize


def evaluate(config: dict, dataset: Path, db_path: Path, output_dir: Path,
             limit: int | None = None, backend=None, *, profile_cuda: bool = False) -> dict:
    dataset = dataset.resolve()
    items = json.loads(dataset.read_text(encoding="utf-8"))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit phải dương.")
        items = items[:limit]
    if not items:
        raise ValueError("Dataset rỗng.")
    if db_path.resolve() == DEFAULT_DB.resolve():
        create_database()
    start_load = time.perf_counter()
    backend = backend or create_backend(config)
    load_ms = (time.perf_counter() - start_load) * 1000
    pipeline = Pipeline(config, backend)
    executor = SQLExecutor(config["execution"])
    cuda = None
    if profile_cuda:
        import torch
        if not torch.cuda.is_available():
            raise ValueError("--profile-cuda requires a working CUDA GPU")
        cuda = torch.cuda
    records = []
    for item in items:
        database = (dataset.parent / item["db_path"]).resolve() if item.get("db_path") else db_path
        gold = executor.execute(database, item["gold_sql"])
        if not gold.success:
            raise ValueError(f"Gold SQL lỗi ở {item['id']}: {gold.error}")
        if cuda:
            for device in range(cuda.device_count()):
                cuda.synchronize(device)
                cuda.reset_peak_memory_stats(device)
        result = pipeline.run(database, item["question"], item.get("evidence", ""))
        peaks = {}
        if cuda:
            for device in range(cuda.device_count()):
                cuda.synchronize(device)
                peaks[str(device)] = cuda.max_memory_allocated(device) / (1024 ** 2)
        chosen = result.selected
        correct = execution_match(chosen.execution if chosen else None, gold, item["gold_sql"])
        exact = bool(chosen and sqlglot.parse_one(chosen.sql, read="sqlite").sql() ==
                     sqlglot.parse_one(item["gold_sql"], read="sqlite").sql())
        error_type = None if correct else "wrong_result"
        if not chosen:
            failures = [c.execution.error_type if c.execution else
                        c.validation.error_type if c.validation else "generation" for c in result.candidates]
            error_type = failures[-1] if failures else "no_candidate"
        records.append({
            "id": item["id"], "db_id": item.get("db_id"), "db_path": str(database),
            "cuda_peak_allocated_mb_by_device": peaks, "difficulty": item.get("difficulty"),
            "correct": correct, "exact_match": exact, "error_type": error_type,
            "candidate_count": len(result.candidates),
            "valid_count": sum(bool(c.execution and c.execution.success) for c in result.candidates),
            "elapsed_ms": result.elapsed_ms,
            "early_stop": result.decision.reason == "consensus" and len(result.candidates) < min(
                config["sampling"]["max_candidates"], sum(config["sampling"]["rounds"])
            ),
            "correction_attempts": result.correction_attempts,
            "correction_successes": result.correction_successes,
            "prediction": result.public_dict(),
        })
    calls = [call for r in records for call in r["prediction"]["schema_retrieval"].get("generation", {}).get("calls", [])]
    summary = {"backend": backend.name, "mode": config["sampling"]["mode"],
               "dataset": dataset.name, "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
               "seed": config["seed"], "model_load_ms": load_ms,
               "mock_notice": "Fixture only; not model accuracy" if backend.name == "mock" else "",
               "schema_mode": config.get("retrieval", {}).get("mode", "bm25") if config["validation"]["schema_selection"] else "full",
               "execution_metric": "project_execution_match_not_official_BIRD_scorer",
               "mean_model_input_tokens_per_call": statistics.mean(c["input_tokens"] for c in calls) if calls else None,
               "total_output_tokens": sum(c["output_tokens"] for c in calls) if calls else None,
               "context_overflows": sum(c["context_overflow"] for c in calls) if calls else None,
               "mean_retrieval_ms": statistics.mean(r["prediction"]["schema_retrieval"].get("latency_ms", 0) for r in records),
               "mean_prompt_characters": statistics.mean(r["prediction"]["schema_retrieval"]["prompt_characters"] for r in records),
               "peak_cuda_allocated_mb_sum_device_peaks": max((sum(r["cuda_peak_allocated_mb_by_device"].values()) for r in records), default=0) if cuda else None,
               **summarize(records)}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--backend", choices=["mock", "huggingface"])
    parser.add_argument("--mode", choices=["greedy", "fixed_sc", "adaptive_sc"])
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/evaluation_questions.json")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--model-id")
    parser.add_argument("--retrieval", choices=["full", "bm25", "dense", "hybrid"])
    parser.add_argument("--profile-cuda", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.backend:
        config["model"]["backend"] = args.backend
    if args.mode:
        config["sampling"]["mode"] = args.mode
    if args.device:
        config["model"]["device"] = args.device
    if args.model_id:
        config["model"]["model_id"] = args.model_id
    if args.retrieval:
        config["validation"]["schema_selection"] = args.retrieval != "full"
        if args.retrieval != "full":
            config.setdefault("retrieval", {})["mode"] = args.retrieval
    logging.basicConfig(level=config["logging_level"])
    output = args.output or ROOT / "results" / time.strftime("%Y%m%d-%H%M%S")
    print(json.dumps(evaluate(config, args.dataset, args.db, output, args.limit, profile_cuda=args.profile_cuda), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
