from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class WatermarkOverride(BaseModel):
    """Optional per-request watermark parameters for the learning UI.

    Missing values fall back to the server settings; validation happens in
    WatermarkConfig so the rules stay in one place.
    """

    model_config = ConfigDict(extra="forbid")

    hash_key: int | None = None
    gamma: float | None = None
    delta: float | None = None


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int | None = None
    seed: int | None = None
    watermark: WatermarkOverride | None = None
    inspect: bool = False

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty")
        if len(value) > 2000:
            raise ValueError("prompt must not exceed 2000 characters")
        return value

    @field_validator("max_new_tokens")
    @classmethod
    def max_new_tokens_must_be_positive(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("max_new_tokens must be at least 1")
        return value

    @field_validator("seed")
    @classmethod
    def seed_must_be_non_negative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("seed must be non-negative")
        return value


class DetectRequest(BaseModel):
    text: str
    include_tokens: bool = False
    watermark: WatermarkOverride | None = None

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        if len(value) > 10000:
            raise ValueError("text must not exceed 10000 characters")
        return value


class TokenDetail(BaseModel):
    index: int
    id: int
    text: str
    green: bool | None
    t: int
    green_count: int
    z: float


class DetectionResponse(BaseModel):
    verdict: Literal["watermarked", "not_watermarked", "inconclusive"]
    num_tokens: int
    green_count: int
    z_score: float
    p_value: float
    threshold: float
    tokens: list[TokenDetail] | None = None


class CandidateDetail(BaseModel):
    id: int
    text: str
    raw: float
    adjusted: float
    green: bool
    prob: float


class StepInspection(BaseModel):
    index: int
    chosen_id: int
    candidates: list[CandidateDetail]


class HealthResponse(BaseModel):
    model_loaded: bool
    model_name: str
    gamma: float
    delta: float
    z_threshold: float
