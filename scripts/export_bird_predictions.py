"""Export predictions and gold in exactly the input order for the official BIRD scorer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def flatten_sql(sql: str) -> str:
    """Collapse formatting whitespace while preserving quoted SQL literals."""
    out, pending_space, quote = [], False, None
    index = 0
    while index < len(sql):
        char = sql[index]
        if quote:
            out.append(char)
            if char == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    out.append(sql[index + 1])
                    index += 1
                else:
                    quote = None
        elif char in "'\"`":
            if pending_space and out and out[-1] != " ":
                out.append(" ")
            pending_space = False
            quote = char
            out.append(char)
        elif char.isspace():
            pending_space = True
        else:
            if pending_space and out and out[-1] != " ":
                out.append(" ")
            pending_space = False
            out.append(char)
        index += 1
    return "".join(out).strip()


def export(dataset: Path, predictions: Path, output: Path) -> int:
    items = json.loads(dataset.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not items or len(items) != len(rows):
        raise ValueError("Dataset/prediction counts differ; export the exact evaluated subset first")
    if len({item["id"] for item in items}) != len(items) or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate question IDs")
    by_id = {row["id"]: row for row in rows}
    if set(by_id) != {item["id"] for item in items}:
        raise ValueError("Dataset/prediction IDs differ")
    predicted, gold = {}, []
    for index, item in enumerate(items):
        db = item["db_id"]
        row = by_id[item["id"]]
        if row.get("db_id") != db:
            raise ValueError(f"Database mismatch for {item['id']}")
        selected = row["prediction"].get("selected")
        # Preserve failures in the denominator: a valid query guaranteed to fail.
        sql = selected["sql"] if selected else 'SELECT * FROM "__bird_missing_prediction__"'
        # Scorer gold files have one SQL per line; normalize only whitespace outside quotes.
        import sqlglot
        sql = flatten_sql(sqlglot.parse_one(sql, read="sqlite").sql(dialect="sqlite", pretty=False))
        reference = flatten_sql(sqlglot.parse_one(item["gold_sql"], read="sqlite").sql(dialect="sqlite", pretty=False))
        if any(c in sql + reference + db for c in ("\n", "\r", "\t")):
            raise ValueError("SQL literals or database ID contain newlines/tabs; export requires manual handling")
        predicted[str(index)] = f"{sql}\t----- bird -----\t{db}"
        gold.append(f"{reference}\t{db}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "predict_mini_dev.json").write_text(json.dumps(predicted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "mini_dev_gold.sql").write_text("\n".join(gold) + "\n", encoding="utf-8")
    (output / "mini_dev_difficulty.jsonl").write_text("\n".join(
        json.dumps({"question_id": i, "db_id": item["db_id"], "difficulty": item.get("difficulty", "unknown")})
        for i, item in enumerate(items)
    ) + "\n", encoding="utf-8")
    (output / "order.json").write_text(json.dumps([item["id"] for item in items], indent=2) + "\n", encoding="utf-8")
    return len(items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(f"Exported {export(args.dataset, args.predictions, args.output)} predictions in dataset order")


if __name__ == "__main__":
    main()
