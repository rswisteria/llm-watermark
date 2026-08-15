import math
from argparse import Namespace

import pytest
import torch

from watermark import (
    DetectionResult,
    GreenListGenerator,
    WatermarkConfig,
    WatermarkDetector,
    WatermarkLogitsProcessor,
)


def test_green_list_is_reproducible_for_same_previous_token():
    config = WatermarkConfig(hash_key=17)
    generator = GreenListGenerator(vocab_size=32, config=config)

    first = generator.green_list(previous_token_id=5)
    second = generator.green_list(previous_token_id=5)

    assert torch.equal(first, second)
    assert len(first) == math.floor(32 * config.gamma)


def test_logits_processor_adds_delta_only_to_green_tokens():
    config = WatermarkConfig(gamma=0.25, delta=2.0, hash_key=17)
    generator = GreenListGenerator(vocab_size=16, config=config)
    processor = WatermarkLogitsProcessor(vocab_size=16, config=config)
    input_ids = torch.tensor([[3]])
    scores = torch.zeros((1, 16))

    result = processor(input_ids, scores)
    green = set(generator.green_list(3).tolist())

    assert all(result[0, token].item() == 2.0 for token in green)
    assert all(result[0, token].item() == 0.0 for token in range(16) if token not in green)


def test_logits_processor_rejects_smaller_logits_vocabulary():
    processor = WatermarkLogitsProcessor(vocab_size=8, config=WatermarkConfig())

    with pytest.raises(
        ValueError,
        match="logits vocabulary size.*watermark vocabulary size",
    ):
        processor(torch.tensor([[3]]), torch.zeros((1, 7)))


def test_logits_processor_masks_padded_logits_outside_watermark_vocabulary():
    config = WatermarkConfig(gamma=0.25, delta=2.0, hash_key=17)
    processor = WatermarkLogitsProcessor(vocab_size=8, config=config)
    generator = GreenListGenerator(vocab_size=8, config=config)

    result = processor(torch.tensor([[3]]), torch.zeros((1, 10)))
    green = set(generator.green_list(3).tolist())

    assert all(result[0, token].item() == 2.0 for token in green)
    assert all(result[0, token].item() == 0.0 for token in range(8) if token not in green)
    assert torch.isneginf(result[0, 8:]).all()


def test_detector_scores_a_green_only_sequence_as_watermarked():
    config = WatermarkConfig(gamma=0.25, hash_key=17, z_threshold=4.0)
    generator = GreenListGenerator(vocab_size=64, config=config)
    token_ids = [7]
    for _ in range(40):
        token_ids.append(int(generator.green_list(token_ids[-1])[0]))

    result = WatermarkDetector(vocab_size=64, config=config).detect_token_ids(token_ids)

    assert isinstance(result, DetectionResult)
    assert result.token_count == 40
    assert result.green_count == 40
    assert result.z_score > 4.0
    assert result.p_value < 3e-5
    assert result.is_watermarked is True


def test_detector_uses_the_documented_z_score_and_p_value_formula():
    config = WatermarkConfig(gamma=0.25, hash_key=17)
    generator = GreenListGenerator(vocab_size=64, config=config)
    token_ids = [7]
    for _ in range(16):
        token_ids.append(int(generator.green_list(token_ids[-1])[0]))

    result = WatermarkDetector(vocab_size=64, config=config).detect_token_ids(token_ids)
    expected_z = (result.green_count - config.gamma * result.token_count) / math.sqrt(
        result.token_count * config.gamma * (1 - config.gamma)
    )
    expected_p = 0.5 * math.erfc(expected_z / math.sqrt(2.0))

    assert result.z_score == pytest.approx(expected_z)
    assert result.p_value == pytest.approx(expected_p)


def test_detector_rejects_a_sequence_exactly_at_the_strict_z_threshold():
    config = WatermarkConfig(gamma=0.5, z_threshold=4.0)
    generator = GreenListGenerator(vocab_size=16, config=config)
    token_ids = [0]
    for _ in range(16):
        token_ids.append(int(generator.green_list(token_ids[-1])[0]))

    result = WatermarkDetector(vocab_size=16, config=config).detect_token_ids(token_ids)

    assert result.z_score == pytest.approx(4.0)
    assert result.is_watermarked is False


def test_detector_accepts_a_sequence_above_the_strict_z_threshold():
    config = WatermarkConfig(gamma=0.5, z_threshold=4.0)
    generator = GreenListGenerator(vocab_size=16, config=config)
    token_ids = [0]
    for _ in range(17):
        token_ids.append(int(generator.green_list(token_ids[-1])[0]))

    result = WatermarkDetector(vocab_size=16, config=config).detect_token_ids(token_ids)

    assert result.z_score > 4.0
    assert result.is_watermarked is True


def test_invalid_configuration_and_short_sequence_are_rejected():
    with pytest.raises(ValueError, match="gamma"):
        WatermarkConfig(gamma=0.0)

    with pytest.raises(ValueError, match="at least two"):
        WatermarkDetector(vocab_size=8, config=WatermarkConfig()).detect_token_ids([1])


@pytest.mark.parametrize("field_name", ["delta", "z_threshold"])
def test_non_finite_numeric_configuration_is_rejected(field_name):
    with pytest.raises(ValueError, match=field_name):
        WatermarkConfig(**{field_name: math.nan})


@pytest.mark.parametrize("hash_key", [1.0, 2**64])
def test_non_integer_or_unsafe_hash_key_is_rejected(hash_key):
    with pytest.raises(ValueError, match="hash_key"):
        WatermarkConfig(hash_key=hash_key)


def test_repeated_bigrams_are_counted_once_when_requested():
    config = WatermarkConfig(gamma=0.5, hash_key=17, ignore_repeated_bigrams=True)
    generator = GreenListGenerator(vocab_size=16, config=config)
    first = int(generator.green_list(2)[0])
    token_ids = [2, first, first, first]

    result = WatermarkDetector(vocab_size=16, config=config).detect_token_ids(token_ids)
    unique_bigrams = {(2, first), (first, first)}
    expected_green_count = sum(
        second in generator.green_list(previous).tolist()
        for previous, second in unique_bigrams
    )

    assert result.token_count == 2
    assert result.green_count == expected_green_count


import demo
from demo import GeneratedText, detect_generated_text, format_detection


def test_format_detection_includes_required_metrics():
    result = DetectionResult(10, 8, 0.8, 3.0, 0.001, False)
    output = format_detection(result)
    assert "T=10" in output
    assert "green=8" in output
    assert "z=3.000" in output
    assert "p=" in output
    assert "透かしなし" in output


def test_generated_text_detection_uses_original_ids_when_round_trip_differs(
    monkeypatch, capsys
):
    class FakeTokenizer:
        def __len__(self):
            return 32

        def __call__(self, text, return_tensors, add_special_tokens):
            assert text == "decoded text"
            assert return_tensors == "pt"
            assert add_special_tokens is False
            return {"input_ids": torch.tensor([[9, 10]])}

    expected = DetectionResult(1, 1, 1.0, 2.0, 0.1, False)

    class DetectorThatRejectsTextDetection:
        def __init__(self, vocab_size, config, tokenizer):
            assert vocab_size == 32
            assert tokenizer is fake_tokenizer

        def detect(self, text):
            raise AssertionError("mismatched IDs must not use text detection")

        def detect_token_ids(self, token_ids):
            assert tuple(token_ids) == (5, 6)
            return expected

    fake_tokenizer = FakeTokenizer()
    monkeypatch.setattr(demo, "WatermarkDetector", DetectorThatRejectsTextDetection)

    result = detect_generated_text(
        fake_tokenizer,
        GeneratedText(text="decoded text", token_ids=(5, 6)),
        WatermarkConfig(),
    )

    assert result is expected
    assert "ラウンドトリップ" in capsys.readouterr().out


def test_detector_only_main_path_does_not_load_a_model(monkeypatch, capsys):
    args = Namespace(
        model="fake-model",
        max_new_tokens=1,
        seed=0,
        gamma=0.25,
        delta=2.0,
        hash_key=17,
        z_threshold=4.0,
        ignore_repeated_bigrams=False,
        detect="detector-only text",
        detect_file=None,
    )
    expected = DetectionResult(1, 1, 1.0, 2.0, 0.1, False)

    monkeypatch.setattr(demo, "parse_args", lambda: args)
    monkeypatch.setattr(demo, "load_tokenizer", lambda model_name: object())
    monkeypatch.setattr(demo, "detect_text", lambda tokenizer, text, config: expected)

    def fail_if_called(model_name):
        raise AssertionError("detector-only path must not load the model")

    monkeypatch.setattr(demo, "load_model", fail_if_called)

    demo.main()

    assert "T=1" in capsys.readouterr().out
