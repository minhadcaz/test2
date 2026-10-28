"""Evaluate schema retrieval on official BIRD metadata, without loading an LLM/DB."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import statistics
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import sqlglot

from core.prompt_builder import build_prompt
from core.schema_selector import SchemaSelector
from evaluation.schema_retrieval import load_questions, load_schemas, reference_schema


def summarize(records: list[dict]) -> dict:
    scored = [r for r in records if r.get("label_error") is None]
    def mean(key):
        return statistics.mean(r[key] for r in scored) if scored else None
    table_gold = sum(r["gold_table_count"] for r in scored)
    column_gold = sum(r["gold_column_count"] for r in scored)
    full_tokens = sum(r["full_prompt_tokens"] for r in records)
    latencies = sorted(r["retrieval_ms"] for r in records)
    def percentile(p):
        return latencies[min(len(latencies) - 1, int((len(latencies) - 1) * p))] if latencies else None
    return {
        "questions": len(records), "scored_questions": len(scored),
        "label_errors": len(records) - len(scored),
        "table_recall_macro": mean("table_recall"),
        "column_recall_macro": mean("column_recall"),
        "table_recall_micro": sum(r["kept_gold_tables"] for r in scored) / max(table_gold, 1),
        "column_recall_micro": sum(r["kept_gold_columns"] for r in scored) / max(column_gold, 1),
        "all_tables_covered": mean("all_tables_covered"),
        "all_columns_covered": mean("all_columns_covered"),
        "all_references_covered": mean("all_references_covered"),
        "mean_selected_tables": statistics.mean(r["selected_table_count"] for r in records),
        "mean_selected_columns": statistics.mean(r["selected_column_count"] for r in records),
        "mean_full_prompt_tokens": full_tokens / len(records),
        "mean_selected_prompt_tokens": statistics.mean(r["selected_prompt_tokens"] for r in records),
        "prompt_token_reduction_micro": 1 - sum(r["selected_prompt_tokens"] for r in records) / max(full_tokens, 1),
        "schema_token_reduction_micro": 1 - sum(r["selected_schema_tokens"] for r in records) / max(sum(r["full_schema_tokens"] for r in records), 1),
        "retrieval_ms_mean": statistics.mean(latencies),
        "retrieval_ms_p50": percentile(.5), "retrieval_ms_p95": percentile(.95),
        "fallback_count": sum(bool(r["retrieval"].get("fallback_reason")) for r in records),
        "mandatory_over_budget_count": sum(r["retrieval"].get("mandatory_columns_over_budget", 0) > 0 for r in records),
    }


def evaluate(data: Path, schemas_path: Path, output: Path, *, mode: str = "bm25",
             top_k_tables: int = 8, top_k_columns: int = 48, max_bridge_tables: int = 2,
             fallback: str = "full", use_evidence: bool = True, limit: int | None = None,
             tokenizer=None, tokenizer_name: str | None = None, retriever_options: dict | None = None) -> dict:
    questions, schemas = load_questions(data), load_schemas(schemas_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        questions = questions[:limit]
    missing = {q["db_id"] for q in questions} - set(schemas)
    if missing:
        raise ValueError(f"Schema missing for databases: {sorted(missing)}")
    if mode not in {"full", "bm25", "dense", "hybrid"}:
        raise ValueError("Unknown retrieval mode")
    options = dict(retriever_options or {})
    selector = SchemaSelector(mode="bm25" if mode == "full" else mode,
                              top_k_tables=top_k_tables, top_k_columns=top_k_columns,
                              max_bridge_tables=max_bridge_tables, fallback=fallback, **options)
    if tokenizer is None:
        def count(text):
            return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
        counter_name = "regex_proxy_not_model_tokens"
    else:
        def count(text):
            return len(tokenizer.encode(text, add_special_tokens=False))
        counter_name = tokenizer_name or getattr(tokenizer, "name_or_path", "provided_tokenizer")
    start = perf_counter()
    records = []
    for index, item in enumerate(questions):
        schema = schemas[item["db_id"]]
        # Gold SQL is accessed only below, after the retrieval call.
        evidence = item.get("evidence", "") if use_evidence else ""
        if mode == "full":
            selected, metadata = schema, {"mode": "full", "latency_ms": 0.0}
        else:
            selected = selector.select(item["question"], schema, evidence)
            metadata = dict(selector.last_metadata)
        full_prompt = build_prompt(item["question"], schema, evidence)
        selected_prompt = build_prompt(item["question"], selected, evidence)
        record = {
            "question_id": item.get("question_id", item.get("id", index)), "db_id": item["db_id"],
            "difficulty": item.get("difficulty", "unknown"), "selected_tables": list(selected),
            "selected_columns": {t: [c["name"] for c in v["columns"]] for t, v in selected.items()},
            "selected_table_count": len(selected),
            "selected_column_count": sum(len(v["columns"]) for v in selected.values()),
            "full_prompt_tokens": count(full_prompt), "selected_prompt_tokens": count(selected_prompt),
            "full_schema_tokens": count(build_prompt("", schema, "")) - count(build_prompt("", {}, "")),
            "selected_schema_tokens": count(build_prompt("", selected, "")) - count(build_prompt("", {}, "")),
            "retrieval_ms": metadata["latency_ms"], "retrieval": metadata, "label_error": None,
        }
        try:
            gold_tables, gold_columns = reference_schema(item.get("SQL") or item.get("sql") or item["gold_sql"], schema)
            kept_tables = {t.casefold() for t in selected}
            kept_columns = {(t.casefold(), c["name"].casefold()) for t, v in selected.items() for c in v["columns"]}
            covered_tables = gold_tables & kept_tables
            covered_columns = gold_columns & kept_columns
            record.update({
                "gold_table_count": len(gold_tables), "gold_column_count": len(gold_columns),
                "kept_gold_tables": len(covered_tables), "kept_gold_columns": len(covered_columns),
                "table_recall": len(covered_tables) / len(gold_tables) if gold_tables else 1.0,
                "column_recall": len(covered_columns) / len(gold_columns) if gold_columns else 1.0,
                "all_tables_covered": int(gold_tables <= kept_tables),
                "all_columns_covered": int(gold_columns <= kept_columns),
                "all_references_covered": int(gold_tables <= kept_tables and gold_columns <= kept_columns),
                "missing_tables": sorted(gold_tables - kept_tables),
                "missing_columns": sorted(".".join(c) for c in gold_columns - kept_columns),
            })
        except Exception as exc:
            record["label_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    groups = defaultdict(list)
    for record in records:
        groups[record["db_id"]].append(record)
    report = {
        "task": "known_database_schema_retrieval", "reference_labels": "derived_from_gold_sql_not_official_link_annotations",
        "mode": mode, "use_evidence": use_evidence,
        "budgets": {"top_k_tables": top_k_tables, "top_k_columns": top_k_columns,
                    "max_bridge_tables": max_bridge_tables, "fallback": fallback},
        "retriever_options": options, "token_counter": counter_name,
        "token_scope": "raw build_prompt text, excluding model chat template",
        "python_version": platform.python_version(), "sqlglot_version": sqlglot.__version__,
        "dataset_sha256": hashlib.sha256(Path(data).read_bytes()).hexdigest(),
        "schemas_sha256": hashlib.sha256(Path(schemas_path).read_bytes()).hexdigest(),
        "elapsed_seconds": perf_counter() - start,
        "summary": summarize(records), "by_database": {db: summarize(rows) for db, rows in groups.items()},
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "details.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--schemas", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=["full", "bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--top-k-tables", type=int, default=8)
    parser.add_argument("--top-k-columns", type=int, default=48)
    parser.add_argument("--max-bridge-tables", type=int, default=2)
    parser.add_argument("--fallback", choices=["full", "top_k"], default="full")
    parser.add_argument("--no-evidence", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tokenizer", help="Optional model/tokenizer path; reports actual tokenizer counts")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-revision")
    parser.add_argument("--cache-dir", default=".cache/schema_embeddings")
    parser.add_argument("--allow-downloads", action="store_true")
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=not args.allow_downloads)
    report = evaluate(args.data, args.schemas, args.output, mode=args.mode,
                      top_k_tables=args.top_k_tables, top_k_columns=args.top_k_columns,
                      max_bridge_tables=args.max_bridge_tables, fallback=args.fallback,
                      use_evidence=not args.no_evidence, limit=args.limit,
                      tokenizer=tokenizer, tokenizer_name=args.tokenizer,
                      retriever_options={"embedding_model": args.embedding_model,
                                         "embedding_revision": args.embedding_revision,
                                         "cache_dir": args.cache_dir,
                                         "local_files_only": not args.allow_downloads})
    print(json.dumps(report["summary"], indent=2))
    if report["summary"]["label_errors"]:
        raise SystemExit("Reference-label errors: see details.jsonl; results are incomplete")


if __name__ == "__main__":
    main()
