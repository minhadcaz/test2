"""Retrieve tables/columns while preserving a bounded number of join paths."""
from __future__ import annotations

from collections import deque
from time import perf_counter

from core.schema_retriever import SchemaRetriever, normalize, tokenize  # noqa: F401


class SchemaSelector:
    def __init__(self, mode: str = "bm25", top_k_tables: int = 8,
                 top_k_columns: int = 48, max_bridge_tables: int = 2,
                 fallback: str = "full", **retriever_options):
        if top_k_tables < 1 or top_k_columns < 1 or max_bridge_tables < 0:
            raise ValueError("Retrieval budgets must be positive (bridges may be zero)")
        if fallback not in {"full", "top_k"}:
            raise ValueError("retrieval.fallback must be full or top_k")
        self.top_k_tables = top_k_tables
        self.top_k_columns = top_k_columns
        self.max_bridge_tables = max_bridge_tables
        self.fallback = fallback
        self.retriever = SchemaRetriever(mode=mode, **retriever_options)
        self.last_metadata: dict = {}

    @staticmethod
    def _path(graph: dict, start: str, target: str) -> list[str]:
        queue = deque([[start]])
        visited = {start}
        while queue:
            path = queue.popleft()
            if path[-1] == target:
                return path
            for neighbor in sorted(graph[path[-1]]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return []

    def select(self, question: str, schema: dict, evidence: str = "") -> dict:
        start = perf_counter()
        ranking, metadata = self.retriever.rank(f"{question}\n{evidence}", schema)
        matched = bool(ranking and ranking[0][1] > 0)
        if not matched and self.fallback == "full":
            result = schema
            metadata.update({"fallback_reason": "no_retrieval_match", "bridge_tables": [],
                             "mandatory_columns_over_budget": 0, "disconnected_pairs": []})
        else:
            table_scores = {t: 0.0 for t in schema}
            column_scores = {}
            for (table, column), score in ranking:
                table_scores[table] = max(table_scores[table], score)
                if column is not None:
                    column_scores[(table, column)] = score
            seeds = sorted(schema, key=lambda t: (-table_scores[t], t))[:self.top_k_tables]
            selected = set(seeds)
            graph = {t: set() for t in schema}
            table_names = {t.casefold(): t for t in schema}
            column_names = {t: {c["name"].casefold(): c["name"] for c in info["columns"]}
                            for t, info in schema.items()}
            foreign_keys = {t: [] for t in schema}
            for t, info in schema.items():
                for original in info.get("foreign_keys", []):
                    target_table = table_names.get(original["table"].casefold())
                    if target_table is None:
                        continue
                    target_column = original.get("target")
                    if target_column is None:
                        primary = [c["name"] for c in schema[target_table]["columns"] if c.get("pk")]
                        target_column = primary[0] if len(primary) == 1 else None
                    fk = {**original, "table": target_table,
                          "column": column_names[t].get(original["column"].casefold(), original["column"]),
                          "target": column_names[target_table].get(str(target_column).casefold(), target_column)}
                    foreign_keys[t].append(fk)
                    if fk["table"] in schema:
                        graph[t].add(fk["table"])
                        graph[fk["table"]].add(t)
            bridges, disconnected = set(), []
            # Connect seeds to a growing component; never expand an entire FK component.
            connected = {seeds[0]} if seeds else set()
            for target in seeds[1:]:
                if target in connected:
                    continue
                paths = [self._path(graph, source, target) for source in sorted(connected)]
                paths = [path for path in paths if path]
                path = min(paths, key=lambda p: (len(p), p)) if paths else []
                additions = set(path) - selected
                if path and len(bridges | additions) <= self.max_bridge_tables:
                    selected.update(additions)
                    bridges.update(additions)
                    connected.update(path)
                else:
                    disconnected.append(target)
            mandatory = {t: {c["name"] for c in schema[t]["columns"] if c.get("pk")} for t in selected}
            for t in selected:
                for fk in foreign_keys[t]:
                    target = fk["table"]
                    if target in selected:
                        mandatory[t].add(fk["column"])
                        # SQLite FK target may be implicit (REFERENCES parent).
                        if fk.get("target"):
                            mandatory[target].add(fk["target"])
            result, over_budget = {}, 0
            query_words = set(tokenize(question + " " + evidence))
            for t, info in schema.items():
                if t not in selected:
                    continue
                columns = info["columns"]
                # Exact column mentions (including evidence) are never dropped.
                explicit = {c["name"] for c in columns if
                            set(tokenize(c["name"])) and set(tokenize(c["name"])) <= query_words}
                keep = mandatory[t] | explicit
                ordered = sorted(columns, key=lambda c: (-column_scores.get((t, c["name"]), 0), c["name"]))
                for column in ordered:
                    if len(keep) >= self.top_k_columns:
                        break
                    keep.add(column["name"])
                indexes = [i for i, c in enumerate(columns) if c["name"] in keep]
                projected = [columns[i].copy() for i in indexes]
                over_budget += max(0, len(projected) - self.top_k_columns)
                result[t] = {**info, "columns": projected,
                             "samples": [[row[i] for i in indexes] for row in info.get("samples", [])],
                             "foreign_keys": [fk.copy() for fk in foreign_keys[t]
                                              if fk["table"] in selected]}
            metadata.update({"fallback_reason": None if matched else "no_match_top_k",
                             "seed_tables": seeds, "bridge_tables": sorted(bridges),
                             "mandatory_columns_over_budget": over_budget,
                             "disconnected_pairs": disconnected})
        full_columns = sum(len(t["columns"]) for t in schema.values())
        kept_columns = sum(len(t["columns"]) for t in result.values())
        metadata.update({"full_tables": len(schema), "selected_tables": len(result),
                         "full_columns": full_columns, "selected_columns": kept_columns,
                         "column_reduction": 1 - kept_columns / max(full_columns, 1),
                         "latency_ms": (perf_counter() - start) * 1000})
        self.last_metadata = metadata
        return result
