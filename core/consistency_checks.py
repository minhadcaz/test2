"""Gold-free plausibility checks on a candidate's SQL and result, used to filter votes."""
from functools import lru_cache

import sqlglot
from sqlglot import exp

from core.schema_inspector import open_readonly, quote_identifier

CHECKS = ("degenerate", "ungrounded", "bad_join")


class ConsistencyChecker:
    """Flags results that are empty/all-NULL, string literals absent from the compared column,
    and JOINs equating key columns that no foreign key links."""

    def __init__(self, db_path):
        self.db_path = db_path
        connection = open_readonly(db_path)
        try:
            tables = [row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            self.names = {table.lower(): table for table in tables}
            self.columns, self.keys, self.links = {}, set(), set()
            for table in tables:
                info = connection.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
                self.columns[table.lower()] = {row[1].lower(): row[1] for row in info}
                self.keys |= {(table.lower(), row[1].lower()) for row in info if row[5]}
                for row in connection.execute(f"PRAGMA foreign_key_list({quote_identifier(table)})"):
                    source, target = (table.lower(), row[3].lower()), (row[2].lower(), (row[4] or "id").lower())
                    self.keys |= {source, target}
                    self.links.add(frozenset({source, target}))
        finally:
            connection.close()

    @lru_cache(maxsize=4096)
    def _value_exists(self, table: str, column: str, value: str) -> bool:
        connection = open_readonly(self.db_path)
        try:
            return connection.execute(
                f"SELECT 1 FROM {quote_identifier(self.names[table])} "
                f"WHERE {quote_identifier(self.columns[table][column])} = ? LIMIT 1", (value,)
            ).fetchone() is not None
        finally:
            connection.close()

    def _resolver(self, tree):
        aliases = {t.alias_or_name.lower(): t.name.lower() for t in tree.find_all(exp.Table)
                   if t.name.lower() in self.columns}

        def resolve(column: exp.Column):
            name = column.name.lower()
            if column.table:
                table = aliases.get(column.table.lower())
                return table if table and name in self.columns[table] else None
            owners = {t for t in aliases.values() if name in self.columns[t]}
            return owners.pop() if len(owners) == 1 else None
        return resolve

    def check_sql(self, sql: str) -> dict:
        flags = {"ungrounded": False, "bad_join": False}
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
        except sqlglot.errors.SqlglotError:
            return flags
        resolve = self._resolver(tree)
        comparisons = [(n.this, [n.expression]) for n in tree.find_all(exp.EQ)]
        comparisons += [(n.expression, [n.this]) for n in tree.find_all(exp.EQ)]
        comparisons += [(n.this, n.expressions) for n in tree.find_all(exp.In)]
        for column, values in comparisons:
            literals = [v.this for v in values if isinstance(v, exp.Literal) and v.is_string]
            if not isinstance(column, exp.Column) or not literals or len(literals) != len(values):
                continue
            table = resolve(column)
            if table and not any(self._value_exists(table, column.name.lower(), v) for v in literals):
                flags["ungrounded"] = True
        for join in tree.find_all(exp.Join):
            condition = join.args.get("on")
            for eq in condition.find_all(exp.EQ) if condition else []:
                if not (isinstance(eq.this, exp.Column) and isinstance(eq.expression, exp.Column)):
                    continue
                left = (resolve(eq.this), eq.this.name.lower())
                right = (resolve(eq.expression), eq.expression.name.lower())
                if (left[0] and right[0] and left[0] != right[0]
                        and left in self.keys and right in self.keys
                        and frozenset({left, right}) not in self.links):
                    flags["bad_join"] = True
        return flags

    @staticmethod
    def check_result(rows: list) -> dict:
        return {"degenerate": not rows or all(v is None for row in rows for v in row)}

    def check(self, sql: str, rows: list) -> dict:
        return self.check_result(rows) | self.check_sql(sql)
