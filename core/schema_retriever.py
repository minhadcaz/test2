"""Small CPU BM25 index, with an optional cached sentence-transformer index.

The index contains schema metadata only. Gold SQL is never an input to retrieval.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

STOP_WORDS = set("a an the is are was were of for to in on at by and or with from that this which what how many show list find all each number total please give me".split())
ALIASES = {
    "students": "sinh vien student", "classes": "lop class",
    "departments": "khoa department", "subjects": "mon hoc tin chi subject",
    "enrollments": "diem dang ky hoc ky enrollment",
}


def normalize(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.casefold())
                   if unicodedata.category(c) != "Mn").replace("đ", "d")


def tokenize(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    words = re.findall(r"[^\W_]+", normalize(text))
    # A small, deterministic plural normalization; no downloaded language resources.
    return [w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w
            for w in words if w not in STOP_WORDS]


def schema_documents(schema: dict) -> tuple[list[tuple[str, str | None]], list[str]]:
    keys, documents = [], []
    for table_name, table in schema.items():
        context = f"{table_name} {ALIASES.get(table_name.casefold(), '')} {table.get('description', '')}"
        columns = table.get("columns", [])
        keys.append((table_name, None))
        documents.append(context + " " + " ".join(str(c['name']) for c in columns))
        for column in columns:
            keys.append((table_name, column["name"]))
            documents.append(f"{context} {column['name']} {column.get('description', '')}")
    return keys, documents


class BM25Index:
    def __init__(self, documents: list[str]):
        self.documents = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(d.values()) for d in self.documents]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.document_frequency = Counter(w for d in self.documents for w in d)

    def scores(self, query: str) -> list[float]:
        scores = [0.0] * len(self.documents)
        for word in set(tokenize(query)):
            frequency = self.document_frequency[word]
            if not frequency:
                continue
            idf = math.log(1 + (len(scores) - frequency + 0.5) / (frequency + 0.5))
            for i, document in enumerate(self.documents):
                tf = document[word]
                if tf:
                    length = self.lengths[i] / max(self.average_length, 1)
                    scores[i] += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length))
        return scores


class SchemaRetriever:
    """Reuse one instance across questions to amortize index construction."""
    def __init__(self, mode: str = "bm25", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embedding_revision: str | None = None, cache_dir: str | None = None,
                 local_files_only: bool = True, device: str = "cpu"):
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("retrieval.mode must be bm25, dense, or hybrid")
        self.mode = mode
        self.embedding_model = embedding_model
        self.embedding_revision = embedding_revision
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.local_files_only = local_files_only
        self.device = device
        self._indexes: dict = {}
        self._model = None

    def _encoder(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Dense retrieval requires requirements-retrieval.txt") from exc
            self._model = SentenceTransformer(
                self.embedding_model, revision=self.embedding_revision,
                local_files_only=self.local_files_only, device=self.device,
            )
        return self._model

    def rank(self, query: str, schema: dict) -> tuple[list, dict]:
        keys, documents = schema_documents(schema)
        fingerprint = hashlib.sha256(json.dumps(
            [keys, documents, self.embedding_model, self.embedding_revision],
            ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        hit = fingerprint in self._indexes
        if not hit:
            entry = {"bm25": BM25Index(documents)}
            if self.mode != "bm25" and documents:
                import numpy as np
                path = self.cache_dir / f"{fingerprint}.npy" if self.cache_dir else None
                if path and path.exists():
                    matrix = np.load(path, allow_pickle=False)
                    if matrix.ndim != 2 or matrix.shape[0] != len(documents):
                        raise ValueError(f"Invalid embedding cache: {path}")
                else:
                    matrix = self._encoder().encode(documents, normalize_embeddings=True,
                                                    show_progress_bar=False)
                    if path:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(path, matrix, allow_pickle=False)
                entry["dense"] = matrix
            self._indexes[fingerprint] = entry
        entry = self._indexes[fingerprint]
        lexical = entry["bm25"].scores(query)
        scores = lexical
        if self.mode != "bm25" and documents:
            vector = self._encoder().encode([query], normalize_embeddings=True,
                                            show_progress_bar=False)[0]
            semantic = (entry["dense"] @ vector).tolist()
            if self.mode == "dense":
                scores = semantic
            else:
                scores = [0.0] * len(keys)
                # Reciprocal-rank fusion avoids comparing incompatible score scales.
                # Zero BM25 matches contribute nothing, instead of arbitrary rank votes.
                for values in (lexical, semantic):
                    ordered = sorted(range(len(keys)), key=lambda i: (-values[i], i))
                    for rank, index in enumerate(ordered, 1):
                        if values[index] > 0:
                            scores[index] += 1 / (60 + rank)
        ranking = sorted(zip(keys, scores), key=lambda pair: (-pair[1], pair[0][0], pair[0][1] or ""))
        return ranking, {"index_cache_hit": hit, "index_fingerprint": fingerprint,
                         "lexical_matches": sum(v > 0 for v in lexical), "mode": self.mode}
