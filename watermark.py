from dataclasses import dataclass
from math import erfc, floor, isfinite, sqrt
from numbers import Integral
from typing import Optional, Sequence

import torch
from transformers import LogitsProcessor


MAX_HASH_KEY = (2**64 - 1) // (2**31 - 1)


@dataclass(frozen=True)
class WatermarkConfig:
    gamma: float = 0.25
    delta: float = 2.0
    hash_key: int = 15485863
    z_threshold: float = 4.0
    ignore_repeated_bigrams: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if not isfinite(self.delta) or self.delta < 0.0:
            raise ValueError("delta must be non-negative")
        if (
            isinstance(self.hash_key, bool)
            or not isinstance(self.hash_key, Integral)
            or not 0 < self.hash_key <= MAX_HASH_KEY
        ):
            raise ValueError(f"hash_key must be an integer between 1 and {MAX_HASH_KEY}")
        if not isfinite(self.z_threshold) or self.z_threshold < 0.0:
            raise ValueError("z_threshold must be non-negative")


class GreenListGenerator:
    def __init__(self, vocab_size: int, config: WatermarkConfig) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.vocab_size = vocab_size
        self.config = config

    def green_list(
        self,
        previous_token_id: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        target_device = torch.device(device or "cpu")
        generator = torch.Generator(device=target_device)
        generator.manual_seed(self.config.hash_key * int(previous_token_id))
        permutation = torch.randperm(
            self.vocab_size,
            generator=generator,
            device=target_device,
        )
        green_size = floor(self.vocab_size * self.config.gamma)
        return permutation[:green_size]


class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size: int, config: WatermarkConfig) -> None:
        self.config = config
        self.generator = GreenListGenerator(vocab_size, config)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        logits_vocab_size = scores.shape[-1]
        watermark_vocab_size = self.generator.vocab_size
        if logits_vocab_size < watermark_vocab_size:
            raise ValueError(
                "logits vocabulary size "
                f"{logits_vocab_size} is smaller than watermark vocabulary size "
                f"{watermark_vocab_size}"
            )

        result = scores.clone()
        for batch_index in range(input_ids.shape[0]):
            previous_token_id = int(input_ids[batch_index, -1].item())
            green_ids = self.generator.green_list(
                previous_token_id,
                device=scores.device,
            )
            result[batch_index, green_ids] += self.config.delta
        if logits_vocab_size > watermark_vocab_size:
            result[:, watermark_vocab_size:] = -float("inf")
        return result


@dataclass(frozen=True)
class DetectionResult:
    token_count: int
    green_count: int
    green_fraction: float
    z_score: float
    p_value: float
    is_watermarked: bool


@dataclass(frozen=True)
class DetectionStep:
    index: int
    token_id: int
    previous_id: Optional[int]
    is_green: Optional[bool]
    scored: int
    green_count: int
    z_score: float


class IncrementalScorer:
    """Scores tokens one at a time with the same bigram rule as WatermarkDetector."""

    def __init__(self, vocab_size: int, config: WatermarkConfig) -> None:
        self.config = config
        self.generator = GreenListGenerator(vocab_size, config)
        self._previous_id: Optional[int] = None
        self._index = 0
        self._scored = 0
        self._green_count = 0
        self._seen_bigrams: set = set()

    @property
    def scored(self) -> int:
        return self._scored

    @property
    def green_count(self) -> int:
        return self._green_count

    def _z_score(self) -> float:
        if self._scored == 0:
            return 0.0
        expected = self.config.gamma * self._scored
        return (self._green_count - expected) / sqrt(
            self._scored * self.config.gamma * (1 - self.config.gamma)
        )

    def push(self, token_id: int) -> DetectionStep:
        token_id = int(token_id)
        is_green: Optional[bool] = None
        if self._previous_id is not None:
            bigram = (self._previous_id, token_id)
            if not (self.config.ignore_repeated_bigrams and bigram in self._seen_bigrams):
                self._seen_bigrams.add(bigram)
                green_ids = self.generator.green_list(bigram[0])
                is_green = bool(token_id in green_ids.tolist())
                self._scored += 1
                self._green_count += int(is_green)
        step = DetectionStep(
            index=self._index,
            token_id=token_id,
            previous_id=self._previous_id,
            is_green=is_green,
            scored=self._scored,
            green_count=self._green_count,
            z_score=self._z_score(),
        )
        self._index += 1
        self._previous_id = token_id
        return step

    def result(self) -> DetectionResult:
        if self._scored == 0:
            raise ValueError("text must contain at least one scorable token")
        z_score = self._z_score()
        p_value = 0.5 * erfc(z_score / sqrt(2.0))
        return DetectionResult(
            token_count=self._scored,
            green_count=self._green_count,
            green_fraction=self._green_count / self._scored,
            z_score=z_score,
            p_value=p_value,
            is_watermarked=z_score > self.config.z_threshold,
        )


class WatermarkDetector:
    def __init__(self, vocab_size: int, config: WatermarkConfig, tokenizer=None) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

    def detect_token_ids_detailed(
        self, token_ids: Sequence[int]
    ) -> tuple[DetectionResult, list[DetectionStep]]:
        if len(token_ids) < 2:
            raise ValueError("text must contain at least two tokens")
        scorer = IncrementalScorer(self.vocab_size, self.config)
        steps = [scorer.push(token_id) for token_id in token_ids]
        return scorer.result(), steps

    def detect_token_ids(self, token_ids: Sequence[int]) -> DetectionResult:
        result, _ = self.detect_token_ids_detailed(token_ids)
        return result

    def detect(self, text: str) -> DetectionResult:
        if self.tokenizer is None:
            raise ValueError("a tokenizer is required to detect text")
        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0]
        token_ids = input_ids.tolist()
        if token_ids and token_ids[0] == self.tokenizer.bos_token_id:
            token_ids = token_ids[1:]
        return self.detect_token_ids(token_ids)
