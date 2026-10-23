"""Transformers local generation with small batches and cooperative time limits."""
import logging
import threading
import time

from core.model_backend import ModelBackend
from core.config import ROOT
from core.types import Candidate

LOGGER = logging.getLogger(__name__)


class HFModelBackend(ModelBackend):
    name = "huggingface"

    def __init__(self, config: dict):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError("Cài PyTorch và requirements-model.txt để dùng model thật.") from error
        self.torch = torch
        self.lock = threading.RLock()
        self.config = config
        model_config = config["model"]
        self.warnings = []
        requested = model_config["device"]
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
        if self.device == "auto":
            self.device = "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA chưa khả dụng. Chọn CPU hoặc kiểm tra driver/PyTorch.")
        dtype_name = model_config["dtype"]
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if dtype_name != "auto" and dtype_name not in dtype_map:
            raise ValueError("dtype phải là auto, float32, float16 hoặc bfloat16.")
        dtype = dtype_map.get(dtype_name)
        if self.device == "cpu":
            dtype = torch.float32
        elif dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise ValueError("GPU không hỗ trợ BF16; chọn dtype auto.")
        kwargs = {"torch_dtype": dtype, "trust_remote_code": False,
                  "cache_dir": str(ROOT / ".cache" / "huggingface" / "hub"),
                  "local_files_only": model_config.get("local_files_only", False)}
        quantization = model_config["quantization"]
        if quantization != "none":
            if self.device == "cpu":
                self.warnings.append("Quantization bị tắt trên CPU; dùng FP32.")
            else:
                try:
                    import bitsandbytes  # noqa: F401
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=quantization == "4bit", load_in_8bit=quantization == "8bit",
                        bnb_4bit_compute_dtype=dtype,
                    )
                    kwargs["device_map"] = {"": 0}
                except (ImportError, OSError, RuntimeError):
                    self.warnings.append("bitsandbytes không khả dụng; dùng dtype thường.")
        model_id = model_config["model_id"]
        local_model = ROOT / "models" / model_id.split("/")[-1]
        if (local_model / "config.json").is_file():
            model_id = str(local_model)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=False,
                cache_dir=kwargs["cache_dir"],
                local_files_only=model_config.get("local_files_only", False),
            )
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            except (ImportError, ValueError, RuntimeError) as error:
                if "quantization_config" not in kwargs or "out of memory" in str(error).lower():
                    raise
                kwargs.pop("quantization_config")
                kwargs.pop("device_map", None)
                self.warnings.append("Backend quantization không tương thích; tải dtype thường.")
                self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            if "device_map" not in kwargs:
                self.model.to(self.device)
            self.model.eval()
            self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        except Exception as error:
            raise RuntimeError(
                "Không tải được model. Kiểm tra model ID, cache, kết nối và bộ nhớ; "
                "xem README để chọn CPU hoặc giảm cấu hình."
            ) from error
        if not self.tokenizer.chat_template:
            self.warnings.append("Tokenizer thiếu chat template; dùng prompt văn bản thuần.")
        for warning in self.warnings:
            LOGGER.warning(warning)

    def start_request(self, seed: int) -> None:
        from transformers import set_seed
        set_seed(seed)

    def generate(self, prompt: str, n: int, parameters: dict,
                 deadline: float | None = None) -> list[Candidate]:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class DeadlineStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return bool(deadline and time.perf_counter() >= deadline)

        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        ) if self.tokenizer.chat_template else prompt
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_length = inputs["input_ids"].shape[-1]
        context = getattr(self.model.config, "max_position_embeddings", 32768)
        if input_length > parameters["max_input_tokens"] or input_length + parameters["max_new_tokens"] > context:
            return [Candidate(0, error="Prompt vượt context; giảm row samples hoặc chọn schema.")]
        inputs = inputs.to(self.model.device)
        candidates = []
        sampled = parameters.get("temperature", 0) > 0
        batch_size = parameters["batch_size"] if sampled else 1
        while len(candidates) < n:
            if deadline and time.perf_counter() >= deadline:
                break
            size = min(batch_size, n - len(candidates))
            kwargs = {
                "max_new_tokens": parameters["max_new_tokens"], "do_sample": sampled,
                "use_cache": True,
                "num_return_sequences": size, "pad_token_id": self.tokenizer.pad_token_id,
                "stopping_criteria": StoppingCriteriaList([DeadlineStop()]),
            }
            if sampled:
                kwargs.update(temperature=parameters["temperature"], top_p=parameters["top_p"])
            else:
                kwargs.update(temperature=1.0, top_p=1.0, top_k=50)
            start = time.perf_counter()
            try:
                with self.lock, self.torch.inference_mode():
                    output = self.model.generate(**inputs, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000 / size
                for sequence in output:
                    candidates.append(Candidate(
                        len(candidates), raw_text=self.tokenizer.decode(
                            sequence[input_length:], skip_special_tokens=True
                        ), generation_ms=elapsed, generation_parameters=dict(parameters),
                    ))
            except RuntimeError as error:
                if "out of memory" in str(error).lower() and batch_size > 1:
                    # Batching is what makes self-consistency affordable; shrink it before giving up.
                    self.torch.cuda.empty_cache()
                    batch_size = max(1, size // 2)
                    LOGGER.warning("CUDA hết bộ nhớ với batch %d; giảm còn %d.", size, batch_size)
                    continue
                message = "CUDA hết bộ nhớ; giảm batch/max_new_tokens hoặc chọn CPU." if (
                    "out of memory" in str(error).lower()
                ) else "Model generation thất bại; kiểm tra cấu hình và dependencies."
                candidates.append(Candidate(len(candidates), error=message,
                                            generation_ms=(time.perf_counter() - start) * 1000))
                break
        return candidates
