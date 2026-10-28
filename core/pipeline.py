"""Orchestrate baselines, adaptive rounds, validation, execution and bounded repair."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import logging
import time

from core.adaptive_sampler import decide
from core.candidate_ranker import rank_candidates
from core.config import validate_config
from core.model_backend import create_backend
from core.prompt_builder import build_prompt
from core.result_grouper import group_candidates
from core.schema_inspector import SchemaInspector
from core.schema_selector import SchemaSelector
from core.sql_corrector import correct_candidate
from core.sql_executor import SQLExecutor
from core.sql_generator import generate_candidates
from core.sql_validator import SQLValidator
from core.types import PipelineResult, ValidationResult

LOGGER = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: dict, backend=None, correction_backend=None):
        validate_config(config)
        self.config = deepcopy(config)
        self.backend = backend or create_backend(config)
        self.correction_backend = correction_backend or self.backend
        self.validator = SQLValidator()
        self.executor = SQLExecutor(config["execution"])
        self.schema_selector = SchemaSelector(**config.get("retrieval", {}))

    def _check(self, candidate, db_path):
        if candidate.error:
            candidate.validation = ValidationResult(errors=[candidate.error], error_type="generation")
            return
        candidate.validation = self.validator.validate(
            candidate.sql, db_path, self.config["validation"]["schema"]
        )
        if candidate.validation.is_safe and candidate.validation.is_valid:
            candidate.execution = self.executor.execute(db_path, candidate.sql)

    def run(self, db_path, question: str, evidence: str = "") -> PipelineResult:
        # Streamlit shares cached model objects: keep seed and all rounds atomic per request.
        with getattr(self.backend, "lock", nullcontext()):
            return self._run(db_path, question, evidence)

    def _run(self, db_path, question: str, evidence: str) -> PipelineResult:
        if not question.strip() or len(question) > 4000 or len(evidence) > 8000:
            raise ValueError("Câu hỏi rỗng/quá dài hoặc evidence quá dài.")
        start = time.perf_counter()
        sample = self.config["sampling"]
        deadline = start + sample["time_budget_seconds"]
        schema = SchemaInspector(**self.config["schema"]).inspect(db_path)
        if self.config["validation"]["schema_selection"]:
            selected_schema = self.schema_selector.select(question, schema, evidence)
            retrieval_metadata = dict(self.schema_selector.last_metadata)
        else:
            selected_schema = schema
            retrieval_metadata = {"mode": "full", "latency_ms": 0.0,
                                  "full_tables": len(schema), "selected_tables": len(schema),
                                  "full_columns": sum(len(t["columns"]) for t in schema.values()),
                                  "selected_columns": sum(len(t["columns"]) for t in schema.values())}
        prompt = build_prompt(question, selected_schema, evidence)
        self.backend.start_request(self.config["seed"])
        if self.correction_backend is not self.backend:
            self.correction_backend.start_request(self.config["seed"])
        mode = sample["mode"]
        rounds = [1] if mode == "greedy" else [sample["fixed_k"]] if mode == "fixed_sc" else sample["rounds"]
        parameters = dict(self.config["generation"])
        if mode == "greedy":
            parameters["temperature"] = 0.0
        candidates, trace = [], []
        attempts = successes = 0
        decision = decide({}, 0, 0, sample)
        for round_index, requested in enumerate(rounds, 1):
            if time.perf_counter() >= deadline:
                decision = decide(group_candidates(candidates), len(candidates),
                                  time.perf_counter() - start, sample)
                break
            count = min(requested, sample["max_candidates"] - len(candidates))
            if count <= 0:
                break
            new = generate_candidates(self.backend, prompt, count, parameters, len(candidates), deadline)
            for candidate in new:
                self._check(candidate, db_path)
                correction = self.config["correction"]
                for _ in range(correction["per_candidate"]):
                    failed = not (candidate.execution and candidate.execution.success)
                    if not (correction["enabled"] and failed and candidate.validation.is_safe
                            and candidate.validation.error_type != "safety"
                            and attempts < correction["total_budget"]
                            and time.perf_counter() < deadline):
                        break
                    if candidate.execution and candidate.execution.error_type in {"timeout", "result_limit", "safety"}:
                        break
                    previous = candidate.public_dict()
                    repaired = correct_candidate(self.correction_backend, candidate, question,
                                                 selected_schema, evidence, parameters, deadline)
                    if repaired is None:
                        break
                    attempts += 1
                    self._check(repaired, db_path)
                    successful = bool(repaired.execution and repaired.execution.success)
                    successes += int(successful)
                    repaired.corrections = candidate.corrections + [{
                        "original_sql": candidate.sql, "validation": previous["validation"],
                        "execution_error": candidate.execution.error if candidate.execution else None,
                        "repaired_sql": repaired.sql, "success": successful,
                    }]
                    repaired.generation_ms += candidate.generation_ms
                    candidate = repaired
                candidates.append(candidate)
            groups = group_candidates(candidates)
            decision = decide(groups, len(candidates), time.perf_counter() - start, sample)
            trace.append({"round": round_index, "requested": count, "generated": len(new),
                          "valid": sum(len(group) for group in groups.values()),
                          "group_sizes": sorted([len(g) for g in groups.values()], reverse=True),
                          "elapsed_ms": (time.perf_counter() - start) * 1000,
                          **asdict(decision)})
            if decision.reason in {"time_budget", "max_candidates"} or (
                mode == "adaptive_sc" and decision.should_stop
            ):
                break
            if not new:
                break
        if decision.reason not in {"time_budget", "max_candidates"}:
            if mode != "adaptive_sc":
                decision.reason = "greedy_complete" if mode == "greedy" else "fixed_complete"
            elif decision.reason != "consensus":
                decision.reason = "rounds_exhausted"
        decision.should_stop = True
        selected = rank_candidates(candidates, self.config["ranking"])
        explanation = (
            f"Chọn ứng viên {selected.id}: chạy thành công, tỷ lệ phiếu {selected.vote_score:.0%}, "
            f"điểm {selected.score:.3f}. Đồng thuận không bảo đảm đúng ý câu hỏi."
        ) if selected else "Không có SQL thực thi thành công trong ngân sách hiện tại."
        LOGGER.info("backend=%s mode=%s candidates=%d reason=%s", self.backend.name, mode,
                    len(candidates), decision.reason)
        result = PipelineResult(question, self.backend.name, mode, candidates, selected, decision,
                                trace, (time.perf_counter() - start) * 1000, explanation, attempts, successes)
        result.schema_retrieval = {**retrieval_metadata, "prompt_characters": len(prompt)}
        metrics = getattr(self.backend, "request_metrics", None)
        if metrics is not None:
            result.schema_retrieval["generation"] = deepcopy(metrics)
        return result
