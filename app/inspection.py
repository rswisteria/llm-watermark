from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from transformers import LogitsProcessor

from watermark import WatermarkLogitsProcessor

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    id: int
    raw: float
    adjusted: float
    green: bool
    prob: float


@dataclass
class StepRecord:
    index: int
    candidates: list[Candidate] = field(default_factory=list)
    chosen_id: int | None = None


class InspectingProcessor(LogitsProcessor):
    """Wraps WatermarkLogitsProcessor and records per-step candidate logits.

    The returned scores are exactly what the inner processor returns; recording
    is a side effect limited to the first ``max_steps`` calls. ``prob`` is the
    softmax of adjusted logits divided by ``temperature`` over the whole
    vocabulary (before top-k / top-p truncation).
    """

    def __init__(
        self,
        inner: WatermarkLogitsProcessor,
        temperature: float = 0.7,
        top_n: int = 10,
        max_steps: int = 400,
    ) -> None:
        self.inner = inner
        self.temperature = temperature
        self.top_n = top_n
        self.max_steps = max_steps
        self.records: list[StepRecord] = []
        self.broken = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        adjusted_scores = self.inner(input_ids, scores)
        if not self.broken and len(self.records) < self.max_steps:
            try:
                raw = scores[0].detach().clone()
                self.records.append(self._record(len(self.records), input_ids, raw, adjusted_scores[0]))
            except Exception:  # recording must never break generation
                logger.exception(
                    "candidate recording failed; disabling inspection for this generation"
                )
                self.broken = True
                self.records.clear()
        return adjusted_scores

    def _record(self, index: int, input_ids, raw: torch.Tensor, adjusted: torch.Tensor) -> StepRecord:
        vocab = self.inner.generator.vocab_size
        raw_v = raw[:vocab].float()
        adj_v = adjusted[:vocab].float()
        k = min(self.top_n, vocab)
        ids: list[int] = []
        for i in torch.topk(raw_v, k).indices.tolist() + torch.topk(adj_v, k).indices.tolist():
            if i not in ids:
                ids.append(i)
        probs = torch.softmax(adj_v / self.temperature, dim=-1)
        previous_id = int(input_ids[0, -1].item())
        green_mask = torch.zeros(vocab, dtype=torch.bool)
        green_mask[self.inner.generator.green_list(previous_id)] = True
        candidates = [
            Candidate(
                id=i,
                raw=float(raw_v[i]),
                adjusted=float(adj_v[i]),
                green=bool(green_mask[i]),
                prob=float(probs[i]),
            )
            for i in ids
        ]
        candidates.sort(key=lambda c: c.adjusted, reverse=True)
        return StepRecord(index=index, candidates=candidates)
