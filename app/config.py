import os
from dataclasses import dataclass
from typing import Mapping

from app.schemas import WatermarkOverride
from watermark import MAX_HASH_KEY, WatermarkConfig


@dataclass(frozen=True)
class ServiceSettings:
    model_name: str = "Qwen/Qwen3-4B"
    hash_key: int = 0
    gamma: float = 0.25
    delta: float = 2.0
    z_threshold: float = 4.0
    max_new_tokens: int = 4096

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("WM_MAX_NEW_TOKENS must be a positive integer")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ServiceSettings":
        values = os.environ if environ is None else environ
        raw_hash_key = values.get("WM_HASH_KEY")
        try:
            if raw_hash_key is None:
                raise ValueError
            hash_key = int(raw_hash_key)
            if not 0 < hash_key <= MAX_HASH_KEY:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("WM_HASH_KEY is required and must be a valid integer") from None

        try:
            max_new_tokens = int(values.get("WM_MAX_NEW_TOKENS", cls.max_new_tokens))
        except (TypeError, ValueError):
            raise ValueError("WM_MAX_NEW_TOKENS must be a positive integer") from None
        if max_new_tokens <= 0:
            raise ValueError("WM_MAX_NEW_TOKENS must be a positive integer")

        return cls(
            model_name=values.get("WM_MODEL_NAME", cls.model_name),
            hash_key=hash_key,
            gamma=float(values.get("WM_GAMMA", cls.gamma)),
            delta=float(values.get("WM_DELTA", cls.delta)),
            z_threshold=float(values.get("WM_Z_THRESHOLD", cls.z_threshold)),
            max_new_tokens=max_new_tokens,
        )

    def watermark_config(self, override: WatermarkOverride | None = None) -> WatermarkConfig:
        override = override or WatermarkOverride()
        return WatermarkConfig(
            gamma=self.gamma if override.gamma is None else override.gamma,
            delta=self.delta if override.delta is None else override.delta,
            hash_key=self.hash_key if override.hash_key is None else override.hash_key,
            z_threshold=self.z_threshold,
        )

    def max_tokens(self, requested: int | None) -> int:
        if requested is None:
            return self.max_new_tokens
        return max(1, min(requested, self.max_new_tokens))
