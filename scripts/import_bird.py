"""Convert a local BIRD JSON/JSONL + SQLite package without modifying databases."""
import argparse
import json
import os
from pathlib import Path


def import_bird(questions: Path, database_root: Path, output: Path) -> list[dict]:
    text = questions.read_text(encoding="utf-8")
    items = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(items, list) or not items:
        raise ValueError("Expected a nonempty JSON array or JSONL of BIRD questions")
    root = database_root.resolve(strict=True)
    records, seen = [], set()
    for index, item in enumerate(items):
        db_id = item["db_id"]
        path = (root / db_id / (db_id + ".sqlite")).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing SQLite database for item {index} ({db_id}).")
        gold = item.get("SQL") or item.get("sql") or item.get("gold_sql")
        if not isinstance(gold, str) or not gold.strip():
            raise ValueError(f"Missing gold SQL for item {index}.")
        identity = f"bird_{item.get('question_id', index)}"
        if identity in seen:
            raise ValueError(f"Duplicate BIRD question ID: {identity}")
        seen.add(identity)
        records.append({"id": identity, "db_id": db_id,
                        "db_path": os.path.relpath(path, output.resolve().parent),
                        "question": item["question"], "evidence": item.get("evidence", ""),
                        "gold_sql": gold, "difficulty": item.get("difficulty", "unknown"),
                        "tags": [], "source": "BIRD (user supplied)", "language": "en"})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = import_bird(args.questions, args.database_root, args.output)
    print(f"Imported {len(records)} questions; database files were not modified.")


if __name__ == "__main__":
    main()
