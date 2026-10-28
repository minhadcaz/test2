"""BIRD schema adapters and SQL-derived reference labels, independent of model inference."""
from __future__ import annotations

import json
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope


def load_questions(path: Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset must be a nonempty JSON array or JSONL")
    seen = set()
    for i, row in enumerate(rows):
        for key in ("question", "db_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Row {i}: missing {key}")
        if not (row.get("SQL") or row.get("sql") or row.get("gold_sql")):
            raise ValueError(f"Row {i}: missing reference SQL")
        identity = row.get("question_id", row.get("id", i))
        if identity in seen:
            raise ValueError(f"Duplicate question id: {identity}")
        seen.add(identity)
    return rows


def parse_schema_ddl(ddl: str) -> dict:
    """Keep structural metadata; omit source comments/value examples by design."""
    schema = {}
    for statement in sqlglot.parse(ddl, read="sqlite"):
        if not isinstance(statement, exp.Create) or not isinstance(statement.this, exp.Schema):
            continue
        name = statement.this.this.name
        columns, foreign_keys, primary = [], [], set()
        def unwrap(items):
            for item in items:
                if isinstance(item, exp.Constraint):
                    yield from unwrap(item.expressions)
                else:
                    yield item
        for item in unwrap(statement.this.expressions):
            if isinstance(item, exp.ColumnDef):
                is_primary = any(isinstance(c.kind, exp.PrimaryKeyColumnConstraint)
                                 for c in item.constraints)
                columns.append({"name": item.name, "type": item.kind.sql(dialect="sqlite") if item.kind else "TEXT",
                                "pk": int(is_primary)})
                for constraint in item.constraints:
                    if isinstance(constraint.kind, exp.Reference):
                        target = constraint.kind.this
                        foreign_keys.append({"column": item.name, "table": target.this.name,
                                             "target": target.expressions[0].name if target.expressions else None})
            elif isinstance(item, exp.PrimaryKey):
                primary.update(c.name for c in item.expressions)
            elif isinstance(item, exp.ForeignKey):
                target = item.args["reference"].this
                targets = target.expressions
                for index, column in enumerate(item.expressions):
                    foreign_keys.append({"column": column.name, "table": target.this.name,
                                         "target": targets[index].name if index < len(targets) else None})
        for column in columns:
            if column["name"] in primary:
                column["pk"] = 1
        schema[name] = {"columns": columns, "foreign_keys": foreign_keys, "samples": []}
    if not schema:
        raise ValueError("No CREATE TABLE definitions in schema")
    return schema


def load_schemas(path: Path) -> dict:
    """Support canonical Mini-Dev DDL dictionaries and BIRD/Spider tables.json."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and all(isinstance(v, str) for v in raw.values()):
        return {db: parse_schema_ddl(ddl) for db, ddl in raw.items()}
    if isinstance(raw, list):
        databases = {}
        for db in raw:
            tables = db["table_names_original"]
            columns = db["column_names_original"]
            primary = set(db.get("primary_keys", []))
            schema = {name: {"columns": [], "foreign_keys": [], "samples": []} for name in tables}
            for index, (table_index, name) in enumerate(columns):
                if table_index >= 0:
                    schema[tables[table_index]]["columns"].append({
                        "name": name, "type": db.get("column_types", ["TEXT"] * len(columns))[index],
                        "pk": int(index in primary)})
            for source, target in db.get("foreign_keys", []):
                source_table, source_name = columns[source]
                target_table, target_name = columns[target]
                schema[tables[source_table]]["foreign_keys"].append({
                    "column": source_name, "table": tables[target_table], "target": target_name})
            databases[db["db_id"]] = schema
        return databases
    raise ValueError("Expected Mini-Dev {db_id: DDL} or BIRD/Spider tables.json list")


def reference_schema(sql: str, schema: dict) -> tuple[set[str], set[tuple[str, str]]]:
    """Resolve aliases/scopes; expand SELECT *, but COUNT(*) needs no individual column.

    These are references in one gold SQL, not official schema-linking annotations.
    Unknown/unresolved references raise rather than silently inflate recall.
    """
    catalog = {t: {c["name"]: "UNKNOWN" for c in table["columns"]} for t, table in schema.items()}
    parsed = sqlglot.parse_one(sql, read="sqlite")
    if any(str(join.args.get("method", "")).upper() == "NATURAL" for join in parsed.find_all(exp.Join)):
        raise ValueError("NATURAL JOIN implicit-key labels are not supported; provide audited explicit JOIN references")
    tree = qualify(parsed, dialect="sqlite", schema=catalog,
                   validate_qualify_columns=True, identify=False)
    known = {t.casefold(): {c["name"].casefold() for c in table["columns"]} for t, table in schema.items()}
    tables, columns = set(), set()
    for scope in traverse_scope(tree):
        for source in scope.sources.values():
            if isinstance(source, exp.Table):
                name = source.name.casefold()
                if name not in known:
                    raise ValueError(f"Unresolved base table: {source.name}")
                tables.add(name)
        for column in scope.columns:
            if isinstance(column.this, exp.Star):
                continue
            if (not column.table and isinstance(scope.expression, exp.SetOperation)
                    and column.name.casefold() in {name.casefold() for name in scope.expression.named_selects}):
                # UNION/INTERSECT/EXCEPT ORDER BY resolves against output aliases.
                continue
            current: Scope | None = scope
            source = None
            while current:
                source = next((value for alias, value in current.sources.items()
                               if alias.casefold() == column.table.casefold()), None)
                if source is not None:
                    break
                current = current.parent
            if isinstance(source, exp.Table):
                table, name = source.name.casefold(), column.name.casefold()
                if name not in known[table]:
                    raise ValueError(f"Unresolved column: {table}.{name}")
                columns.add((table, name))
            elif source is None:
                raise ValueError(f"Unresolved column source: {column.sql()}")
            # Derived/CTE output columns are resolved by visiting the inner scopes.
    return tables, columns
