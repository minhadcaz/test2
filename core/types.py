"""Serializable records shared by all pipeline modules."""
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    is_safe: bool = False
    is_valid: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    schema_score: float = 0.0
    error_type: str | None = None


@dataclass
class ExecutionResult:
    success: bool = False
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    normalized_result: str | None = None
    unordered_result: str | None = None
    ordered_result: str | None = None
    row_count: int = 0
    truncated: bool = False
    elapsed_ms: float = 0.0
    error: str | None = None
    error_type: str | None = None


@dataclass
class Candidate:
    id: int
    raw_text: str = ""
    sql: str = ""
    generation_ms: float = 0.0
    generation_parameters: dict = field(default_factory=dict)
    error: str | None = None
    validation: ValidationResult | None = None
    execution: ExecutionResult | None = None
    group: str | None = None
    vote_score: float = 0.0
    score: float = 0.0
    generation_score: float = 0.5
    corrections: list[dict] = field(default_factory=list)

    def public_dict(self) -> dict:
        record = asdict(self)
        record.pop("raw_text", None)
        return record


@dataclass
class StoppingDecision:
    should_stop: bool
    reason: str
    confidence: float
    margin: float
    candidates_used: int


@dataclass
class PipelineResult:
    question: str
    backend: str
    mode: str
    candidates: list[Candidate]
    selected: Candidate | None
    decision: StoppingDecision
    trace: list[dict]
    elapsed_ms: float
    explanation: str
    correction_attempts: int = 0
    correction_successes: int = 0
    schema_retrieval: dict = field(default_factory=dict)

    def public_dict(self) -> dict:
        return {
            "question": self.question, "backend": self.backend, "mode": self.mode,
            "candidates": [c.public_dict() for c in self.candidates],
            "selected": self.selected.public_dict() if self.selected else None,
            "decision": asdict(self.decision), "trace": self.trace,
            "elapsed_ms": self.elapsed_ms, "explanation": self.explanation,
            "correction_attempts": self.correction_attempts,
            "correction_successes": self.correction_successes,
            "schema_retrieval": self.schema_retrieval,
        }
