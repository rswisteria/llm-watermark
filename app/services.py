from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterator, Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    TextIteratorStreamer,
)

from app.config import ServiceSettings
from app.schemas import DetectionResponse, TokenDetail
from app.token_pieces import token_pieces
from watermark import (
    DetectionStep,
    IncrementalScorer,
    WatermarkConfig,
    WatermarkLogitsProcessor,
)


class ModelNotReadyError(RuntimeError):
    pass


class GenerationQueueFullError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationEvent:
    kind: Literal["token", "done", "error"]
    payload: dict


class GenerationLease:
    def __init__(self, slots: threading.BoundedSemaphore) -> None:
        self._slots = slots
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if not self._released:
                self._released = True
                self._slots.release()


def _inconclusive_response(
    config: WatermarkConfig, token_count: int = 0, tokens: list[TokenDetail] | None = None
) -> DetectionResponse:
    return DetectionResponse(
        verdict="inconclusive",
        num_tokens=token_count,
        green_count=0,
        z_score=0.0,
        p_value=1.0,
        threshold=config.z_threshold,
        tokens=tokens,
    )


def _detection_response(
    result, config: WatermarkConfig, tokens: list[TokenDetail] | None = None
) -> DetectionResponse:
    if result.token_count < 25:
        verdict = "inconclusive"
    else:
        verdict = "watermarked" if result.z_score > config.z_threshold else "not_watermarked"
    return DetectionResponse(
        verdict=verdict,
        num_tokens=result.token_count,
        green_count=result.green_count,
        z_score=result.z_score,
        p_value=result.p_value,
        threshold=config.z_threshold,
        tokens=tokens,
    )


def _steps_to_tokens(steps: list[DetectionStep], pieces: list[str]) -> list[TokenDetail]:
    return [
        TokenDetail(
            index=step.index,
            id=step.token_id,
            text=piece,
            green=step.is_green,
            t=step.scored,
            green_count=step.green_count,
            z=step.z_score,
        )
        for step, piece in zip(steps, pieces)
    ]


def _classify_token_ids(
    token_ids: list[int],
    vocab_size: int,
    config: WatermarkConfig,
    tokenizer=None,
    include_tokens: bool = False,
) -> DetectionResponse:
    scorer = IncrementalScorer(vocab_size, config)
    steps = [scorer.push(token_id) for token_id in token_ids]
    tokens = None
    if include_tokens and tokenizer is not None:
        tokens = _steps_to_tokens(steps, token_pieces(tokenizer, token_ids))
    if scorer.scored == 0:
        return _inconclusive_response(config, max(0, len(token_ids) - 1), tokens)
    return _detection_response(scorer.result(), config, tokens)


class DetectionService:
    def __init__(self, tokenizer, config: WatermarkConfig) -> None:
        self.tokenizer = tokenizer
        self.config = config

    def is_ready(self) -> bool:
        return self.tokenizer is not None

    def _token_ids(self, text: str) -> list[int]:
        input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        token_ids = [int(i) for i in input_ids.tolist()]
        if token_ids and token_ids[0] == getattr(self.tokenizer, "bos_token_id", None):
            token_ids = token_ids[1:]
        return token_ids

    def classify(self, text: str, include_tokens: bool = False) -> DetectionResponse:
        if not self.is_ready():
            raise ModelNotReadyError("tokenizer is not ready")
        return _classify_token_ids(
            self._token_ids(text),
            len(self.tokenizer),
            self.config,
            tokenizer=self.tokenizer,
            include_tokens=include_tokens,
        )


class GenerationService:
    def __init__(
        self,
        settings: ServiceSettings,
        config: WatermarkConfig,
        tokenizer=None,
        model=None,
    ) -> None:
        self.settings = settings
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.detection = DetectionService(tokenizer, config)
        self._loading_started = False
        self._loading_error: Exception | None = None
        self._loading_lock = threading.Lock()
        self._capacity = threading.BoundedSemaphore(2)
        self._generation = threading.Semaphore(1)

    def start_loading(self) -> None:
        with self._loading_lock:
            if self._loading_started or self.model is not None:
                return
            self._loading_started = True
            threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.settings.model_name)
            self.tokenizer = tokenizer
            self.detection.tokenizer = tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                self.settings.model_name,
                torch_dtype=torch.bfloat16,
            )
            self.model = model.to("cpu")
            self.model.eval()
        except Exception as exc:
            self._loading_error = exc

    def health(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def reserve_slot(self) -> GenerationLease:
        if not self._capacity.acquire(blocking=False):
            raise GenerationQueueFullError("generation queue is full")
        return GenerationLease(self._capacity)

    def begin(
        self, prompt: str, max_new_tokens: int, seed: int | None
    ) -> Iterator[GenerationEvent]:
        if not self.health():
            raise ModelNotReadyError("model is not ready")
        lease = self.reserve_slot()
        self._generation.acquire()
        try:
            yield from self._generate(prompt, max_new_tokens, seed)
        finally:
            self._generation.release()
            lease.release()

    def _generate(
        self, prompt: str, max_new_tokens: int, seed: int | None
    ) -> Iterator[GenerationEvent]:
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if apply_chat_template is not None:
            inputs = apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        result_holder: dict = {}

        def worker() -> None:
            try:
                if seed is not None:
                    torch.manual_seed(seed)
                result_holder["output"] = self.model.generate(
                    **inputs,
                    streamer=streamer,
                    logits_processor=LogitsProcessorList(
                        [WatermarkLogitsProcessor(len(self.tokenizer), self.config)]
                    ),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                )
            except Exception as exc:
                result_holder["error"] = exc
            finally:
                streamer.end()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        for fragment in streamer:
            if fragment:
                yield GenerationEvent("token", {"text": fragment})
        thread.join()

        if "error" in result_holder:
            yield GenerationEvent("error", {"message": "generation failed"})
            return

        try:
            output = result_holder.get("output")
            output_ids = output[0].tolist() if hasattr(output, "__getitem__") else list(output)
            prompt_ids = inputs["input_ids"]
            prompt_length = int(prompt_ids.shape[-1]) if hasattr(prompt_ids, "shape") else len(prompt_ids[0])
            token_ids = [int(token_id) for token_id in output_ids[prompt_length:]]
            special_ids = set(getattr(self.tokenizer, "all_special_ids", []))
            token_ids = [token_id for token_id in token_ids if token_id not in special_ids]
            full_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            response = _classify_token_ids(token_ids, len(self.tokenizer), self.config)
        except Exception:
            yield GenerationEvent("error", {"message": "generation failed"})
            return
        yield GenerationEvent(
            "done",
            {"full_text": full_text, "detection": response.model_dump()},
        )
