import math

import pytest
import torch

from app.inspection import InspectingProcessor, StepRecord
from watermark import GreenListGenerator, WatermarkConfig, WatermarkLogitsProcessor


def make_scores(vocab=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, vocab, generator=g)


def test_inspecting_processor_returns_same_adjusted_logits_as_inner():
    config = WatermarkConfig(hash_key=17, gamma=0.5, delta=2.0)
    inner = WatermarkLogitsProcessor(32, config)
    processor = InspectingProcessor(inner, temperature=0.7)
    input_ids = torch.tensor([[1, 2, 3]])
    scores = make_scores()

    expected = WatermarkLogitsProcessor(32, config)(input_ids, scores.clone())
    actual = processor(input_ids, scores.clone())

    assert torch.allclose(actual, expected)


def test_inspecting_processor_records_candidates_with_softmax_probs():
    config = WatermarkConfig(hash_key=17, gamma=0.5, delta=2.0)
    inner = WatermarkLogitsProcessor(32, config)
    processor = InspectingProcessor(inner, temperature=0.7, top_n=5)
    input_ids = torch.tensor([[1, 2, 3]])
    scores = make_scores()

    adjusted = processor(input_ids, scores.clone())

    assert len(processor.records) == 1
    record = processor.records[0]
    assert isinstance(record, StepRecord)
    assert record.index == 0
    assert record.chosen_id is None
    ids = [c.id for c in record.candidates]
    assert 5 <= len(ids) <= 10
    assert len(set(ids)) == len(ids)
    # 元 logit 上位 5 と補正後上位 5 を必ず含む
    for i in torch.topk(scores[0], 5).indices.tolist():
        assert i in ids
    for i in torch.topk(adjusted[0], 5).indices.tolist():
        assert i in ids
    green_ids = set(GreenListGenerator(32, config).green_list(3).tolist())
    probs = torch.softmax(adjusted[0] / 0.7, dim=-1)
    for c in record.candidates:
        assert c.raw == pytest.approx(float(scores[0, c.id]))
        assert c.adjusted == pytest.approx(float(adjusted[0, c.id]))
        assert c.green == (c.id in green_ids)
        assert c.prob == pytest.approx(float(probs[c.id]), rel=1e-5)
        if c.green:
            assert c.adjusted == pytest.approx(c.raw + 2.0)
        else:
            assert c.adjusted == pytest.approx(c.raw)


def test_inspecting_processor_marks_green_even_when_delta_is_zero():
    config = WatermarkConfig(hash_key=17, gamma=0.5, delta=0.0)
    processor = InspectingProcessor(WatermarkLogitsProcessor(32, config))
    processor(torch.tensor([[7]]), make_scores())
    green_ids = set(GreenListGenerator(32, config).green_list(7).tolist())
    record = processor.records[0]
    for c in record.candidates:
        assert c.green == (c.id in green_ids)
        assert c.adjusted == pytest.approx(c.raw)


def test_inspecting_processor_stops_recording_after_max_steps():
    config = WatermarkConfig(hash_key=17)
    processor = InspectingProcessor(WatermarkLogitsProcessor(32, config), max_steps=3)
    for step in range(5):
        processor(torch.tensor([[step + 1]]), make_scores(seed=step))
    assert [r.index for r in processor.records] == [0, 1, 2]
