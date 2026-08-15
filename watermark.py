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


class WatermarkDetector:
    def __init__(self, vocab_size: int, config: WatermarkConfig, tokenizer=None) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.generator = GreenListGenerator(vocab_size, config)

    def detect_token_ids(self, token_ids: Sequence[int]) -> DetectionResult:
        if len(token_ids) < 2:
            raise ValueError("text must contain at least two tokens")

        green_count = 0
        scored = 0
        seen_bigrams = set()
        for index in range(1, len(token_ids)):
            bigram = (int(token_ids[index - 1]), int(token_ids[index]))
            if self.config.ignore_repeated_bigrams and bigram in seen_bigrams:
                continue
            seen_bigrams.add(bigram)
            scored += 1
            green_ids = self.generator.green_list(bigram[0])
            green_count += int(bigram[1] in green_ids.tolist())

        if scored == 0:
            raise ValueError("text must contain at least one scorable token")

        expected = self.config.gamma * scored
        z_score = (green_count - expected) / sqrt(
            scored * self.config.gamma * (1 - self.config.gamma)
        )
        p_value = 0.5 * erfc(z_score / sqrt(2.0))
        return DetectionResult(
            token_count=scored,
            green_count=green_count,
            green_fraction=green_count / scored,
            z_score=z_score,
            p_value=p_value,
            is_watermarked=z_score > self.config.z_threshold,
        )

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
