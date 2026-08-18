import inspect
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from app import services as services_module
from app.config import ServiceSettings
from app.services import (
    DetectionService,
    GenerationQueueFullError,
    GenerationEvent,
    GenerationService,
    ModelNotReadyError,
)
from transformers import LogitsProcessorList
from watermark import WatermarkConfig, WatermarkLogitsProcessor


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_documentation_provides_secret_free_environment_template():
    env_example = (REPOSITORY_ROOT / ".env.example").read_text()
    env_lines = env_example.splitlines()
    hash_key_assignments = [
        line.strip() for line in env_lines if line.strip().startswith("WM_HASH_KEY=")
    ]

    assert hash_key_assignments == ["WM_HASH_KEY="]
    for required_default in (
        "WM_MODEL_NAME=Qwen/Qwen3-4B",
        "WM_GAMMA=0.25",
        "WM_DELTA=2.0",
        "WM_Z_THRESHOLD=4.0",
        "WM_MAX_NEW_TOKENS=4096",
    ):
        assert required_default in env_lines


def test_documentation_provides_localhost_uvicorn_startup_command():
    readme = (REPOSITORY_ROOT / "README.md").read_text()

    assert "uvicorn app.main:app --host 127.0.0.1 --port 8000" in readme


class FakeTokenizer:
    bos_token_id = 0
    all_special_ids = [0]

    def __len__(self):
        return 16

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        return {"input_ids": torch.tensor([[1, 2]])}

    def decode(self, token_ids, skip_special_tokens=True):
        ids = [int(token_id) for token_id in token_ids]
        return "continuation:" + ",".join(str(token_id) for token_id in ids) + " "


class PieceTokenizer(FakeTokenizer):
    """decode が ID ごとに 'w<id>' を返す（判定 API のトークン表示テスト用）。"""

    def __init__(self, ids):
        self.ids = ids

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        return {"input_ids": torch.tensor([self.ids])}

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(f"w{int(i)}" for i in token_ids)


class ChatTemplateTokenizer(FakeTokenizer):
    def __init__(self):
        self.chat_template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.chat_template_kwargs = {"messages": messages, **kwargs}
        return {"input_ids": torch.tensor([[1, 2]])}


class FakeModel:
    def __init__(self, tokenizer, continuation_ids=None):
        self.tokenizer = tokenizer
        self.continuation_ids = (
            list(range(3, 28)) if continuation_ids is None else continuation_ids
        )
        self.last_generation = None

    def generate(self, input_ids, streamer, **kwargs):
        self.last_generation = kwargs
        streamer.put(input_ids)
        midpoint = len(self.continuation_ids) // 2
        streamer.put(torch.tensor([self.continuation_ids[:midpoint]]))
        streamer.put(torch.tensor([self.continuation_ids[midpoint:]]))
        streamer.end()
        return torch.tensor([[1, 2, *self.continuation_ids]])


class InspectableFakeModel(FakeModel):
    """Calls each logits processor once per continuation token, like generate() would."""

    def generate(self, input_ids, streamer, **kwargs):
        self.last_generation = kwargs
        processors = kwargs.get("logits_processor") or []
        streamer.put(input_ids)
        history = list(input_ids[0].tolist())
        for token_id in self.continuation_ids:
            scores = torch.zeros(1, 16)
            scores[0, token_id] = 5.0
            for processor in processors:
                scores = processor(torch.tensor([history]), scores)
            history.append(token_id)
            streamer.put(torch.tensor([token_id]))
        streamer.end()
        return torch.tensor([[1, 2, *self.continuation_ids]])


class QwenLikeModel(FakeModel):
    def generate(self, input_ids, streamer, **kwargs):
        if "enable_thinking" in kwargs:
            raise TypeError("enable_thinking is not a generate argument")
        return super().generate(input_ids, streamer, **kwargs)


def make_generation_service():
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer)
    config = WatermarkConfig(hash_key=17)
    settings = ServiceSettings(hash_key=17)
    return GenerationService(
        settings=settings,
        config=config,
        tokenizer=tokenizer,
        model=model,
    ), model


def make_inspectable_service(continuation_ids=None):
    tokenizer = FakeTokenizer()
    if continuation_ids is None:
        # FakeModel's default range(3, 28) would overflow InspectableFakeModel's
        # (1, 16) scores tensor, which is sized to FakeTokenizer's 16-token vocab.
        continuation_ids = list(range(3, 15))
    model = InspectableFakeModel(tokenizer, continuation_ids)
    return GenerationService(
        settings=ServiceSettings(hash_key=17), config=WatermarkConfig(hash_key=17),
        tokenizer=tokenizer, model=model,
    ), model


def test_generation_without_inspect_has_null_steps():
    service, model = make_inspectable_service()
    events = list(service.begin("prompt", max_new_tokens=12, seed=None))
    assert events[-1].payload["steps"] is None
    assert "logits_processor" in model.last_generation


def test_generation_with_inspect_returns_steps_aligned_with_tokens():
    service, model = make_inspectable_service(continuation_ids=[*range(3, 12), 0])  # 0 は special
    events = list(service.begin("prompt", max_new_tokens=12, seed=None, inspect=True))
    done = events[-1].payload
    tokens = done["detection"]["tokens"]
    steps = done["steps"]
    assert steps is not None
    assert len(steps) == len(tokens) == 9
    for step, tok in zip(steps, tokens):
        assert step["index"] == tok["index"]
        assert step["chosen_id"] == tok["id"]
        assert step["candidates"]
        chosen = [c for c in step["candidates"] if c["id"] == tok["id"]]
        assert chosen and chosen[0]["raw"] == pytest.approx(5.0)
        assert all(set(c) == {"id", "text", "raw", "adjusted", "green", "prob"} for c in step["candidates"])
    from app.inspection import InspectingProcessor
    assert isinstance(model.last_generation["logits_processor"][0], InspectingProcessor)


def test_generation_with_inspect_and_zero_delta_still_records():
    service, model = make_inspectable_service()
    events = list(service.begin("prompt", max_new_tokens=12, seed=None,
                                config=WatermarkConfig(hash_key=17, delta=0.0), inspect=True))
    assert events[-1].payload["steps"] is not None
    assert "logits_processor" in model.last_generation


def test_generation_with_inspect_disables_steps_and_logs_when_recording_fails(monkeypatch, caplog):
    from app.inspection import InspectingProcessor

    def broken_record(self, index, input_ids, raw, adjusted):
        raise RuntimeError("boom")

    monkeypatch.setattr(InspectingProcessor, "_record", broken_record)

    service, model = make_inspectable_service()
    with caplog.at_level(logging.ERROR, logger="app.inspection"):
        events = list(service.begin("prompt", max_new_tokens=12, seed=None, inspect=True))

    done = events[-1].payload
    assert done["steps"] is None
    assert done["detection"]["tokens"]
    assert "logits_processor" in model.last_generation

    assert any(
        "candidate recording failed" in record.message for record in caplog.records
    )


def test_generate_endpoint_forwards_inspect_flag():
    client, fake_service, _ = make_api_client()
    client.post("/api/generate", json={"prompt": "テスト"})
    client.post("/api/generate", json={"prompt": "テスト", "inspect": True})
    assert [c["inspect"] for c in fake_service.calls[-2:]] == [False, True]


def test_detection_service_returns_inconclusive_for_short_text():
    tokenizer = FakeTokenizer()
    config = WatermarkConfig(hash_key=17)

    result = DetectionService(tokenizer=tokenizer, config=config).classify("short")

    assert result.verdict == "inconclusive"


def test_detection_service_returns_tokens_only_when_requested():
    from watermark import GreenListGenerator

    config = WatermarkConfig(hash_key=17)
    generator = GreenListGenerator(16, config)
    ids = [3]
    for _ in range(30):
        ids.append(int(generator.green_list(ids[-1])[0]))
    service = DetectionService(tokenizer=PieceTokenizer(ids), config=config)

    plain = service.classify("x")
    detailed = service.classify("x", include_tokens=True)

    assert plain.tokens is None
    assert plain.verdict == "watermarked"
    assert detailed.tokens is not None
    assert len(detailed.tokens) == len(ids)
    assert detailed.tokens[0].model_dump() == {
        "index": 0, "id": 3, "text": "w3", "green": None, "t": 0, "green_count": 0, "z": 0.0,
    }
    assert detailed.tokens[1].green is True
    assert detailed.tokens[-1].t == detailed.num_tokens
    assert detailed.tokens[-1].z == pytest.approx(detailed.z_score)
    assert "tokens" not in plain.model_dump(exclude_none=True)


def test_detection_service_short_text_with_tokens_is_inconclusive_and_has_pieces():
    config = WatermarkConfig(hash_key=17)
    service = DetectionService(tokenizer=PieceTokenizer([3, 4, 5]), config=config)
    result = service.classify("x", include_tokens=True)
    assert result.verdict == "inconclusive"
    assert [t.text for t in result.tokens] == ["w3", "w4", "w5"]


def test_detection_service_tokenize_returns_pieces():
    service = DetectionService(tokenizer=PieceTokenizer([3, 4, 5]), config=WatermarkConfig(hash_key=17))
    result = service.tokenize("x")
    assert result.count == 3
    assert [t.model_dump() for t in result.tokens] == [
        {"index": 0, "id": 3, "text": "w3"}, {"index": 1, "id": 4, "text": "w4"}, {"index": 2, "id": 5, "text": "w5"},
    ]


def test_detection_service_tokenize_requires_tokenizer():
    service = DetectionService(tokenizer=None, config=WatermarkConfig(hash_key=17))
    with pytest.raises(ModelNotReadyError):
        service.tokenize("x")


def test_generation_service_emits_tokens_then_done(monkeypatch):
    service, model = make_generation_service()
    seed_calls = []
    monkeypatch.setattr(torch, "manual_seed", lambda seed: seed_calls.append(seed))

    events = list(service.begin("prompt", max_new_tokens=12, seed=7))

    assert [event.kind for event in events] == ["token", "token", "done"]
    assert events[-1].payload["full_text"] == (
        "continuation:" + ",".join(str(token_id) for token_id in range(3, 28)) + " "
    )
    assert events[-1].payload["detection"]["threshold"] == 4.0
    assert events[-1].payload["detection"]["num_tokens"] == 24
    assert model.last_generation["max_new_tokens"] == 12
    assert model.last_generation["do_sample"] is True
    assert model.last_generation["temperature"] == 0.7
    assert model.last_generation["top_p"] == 0.8
    assert model.last_generation["top_k"] == 20
    assert "enable_thinking" not in model.last_generation
    assert seed_calls == [7]
    assert isinstance(model.last_generation["logits_processor"], LogitsProcessorList)
    assert isinstance(model.last_generation["logits_processor"][0], WatermarkLogitsProcessor)


def test_qwen_generation_disables_thinking_in_chat_template_not_generate():
    tokenizer = ChatTemplateTokenizer()
    model = QwenLikeModel(tokenizer, continuation_ids=[3, 4, 5])
    service = GenerationService(
        settings=ServiceSettings(hash_key=17),
        config=WatermarkConfig(hash_key=17),
        tokenizer=tokenizer,
        model=model,
    )

    events = list(service.begin("prompt", max_new_tokens=12, seed=None))

    assert events[-1].kind == "done"
    assert tokenizer.chat_template_kwargs["messages"] == [
        {"role": "user", "content": "prompt"}
    ]
    assert tokenizer.chat_template_kwargs["enable_thinking"] is False
    assert "enable_thinking" not in model.last_generation


def test_generation_service_done_event_marks_short_continuation_inconclusive():
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer, continuation_ids=[3, 4])
    service = GenerationService(
        settings=ServiceSettings(hash_key=17),
        config=WatermarkConfig(hash_key=17),
        tokenizer=tokenizer,
        model=model,
    )

    events = list(service.begin("prompt", max_new_tokens=12, seed=None))

    assert events[-1].kind == "done"
    assert events[-1].payload["full_text"] == "continuation:3,4 "
    assert events[-1].payload["detection"]["verdict"] == "inconclusive"
    assert events[-1].payload["detection"]["num_tokens"] == 1


def test_generation_service_one_token_continuation_emits_inconclusive_done():
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer, continuation_ids=[3])
    service = GenerationService(
        settings=ServiceSettings(hash_key=17),
        config=WatermarkConfig(hash_key=17),
        tokenizer=tokenizer,
        model=model,
    )

    events = list(service.begin("prompt", max_new_tokens=12, seed=None))

    assert events[-1].kind == "done"
    assert events[-1].payload["detection"]["verdict"] == "inconclusive"
    assert events[-1].payload["detection"]["num_tokens"] == 0


def test_generation_service_with_no_continuation_emits_inconclusive_done():
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer, continuation_ids=[])
    service = GenerationService(
        settings=ServiceSettings(hash_key=17),
        config=WatermarkConfig(hash_key=17),
        tokenizer=tokenizer,
        model=model,
    )

    events = list(service.begin("prompt", max_new_tokens=12, seed=None))

    assert events[-1].kind == "done"
    assert events[-1].payload["detection"]["verdict"] == "inconclusive"
    assert events[-1].payload["detection"]["num_tokens"] == 0


def test_generation_stream_carries_live_token_details_matching_final_detection():
    service, _ = make_generation_service()

    events = list(service.begin("prompt", max_new_tokens=12, seed=None))

    token_events = [e for e in events if e.kind == "token"]
    live = [tok for e in token_events for tok in e.payload["tokens"]]
    done = events[-1].payload["detection"]

    assert all("text" in e.payload for e in token_events)
    assert [tok["index"] for tok in live] == list(range(25))
    assert live[0]["green"] is None and live[0]["t"] == 0
    assert live[-1]["t"] == done["num_tokens"] == 24
    assert live[-1]["z"] == pytest.approx(done["z_score"])
    assert done["tokens"] is not None
    assert [tok["id"] for tok in done["tokens"]] == [tok["id"] for tok in live]
    assert [tok["z"] for tok in done["tokens"]] == pytest.approx([tok["z"] for tok in live])


def test_generation_stream_skips_special_tokens_in_live_scoring():
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer, continuation_ids=[*range(3, 30), 0])  # 0 は special
    service = GenerationService(
        settings=ServiceSettings(hash_key=17), config=WatermarkConfig(hash_key=17),
        tokenizer=tokenizer, model=model,
    )
    events = list(service.begin("prompt", max_new_tokens=12, seed=None))
    live = [tok for e in events if e.kind == "token" for tok in e.payload["tokens"]]
    done = events[-1].payload["detection"]
    assert 0 not in [tok["id"] for tok in live]
    assert live[-1]["z"] == pytest.approx(done["z_score"])
    assert len(done["tokens"]) == len(live) == 27


def test_generation_stream_logs_and_recovers_when_live_scoring_fails(monkeypatch, caplog):
    class BrokenTokenPieceBuilder(services_module.TokenPieceBuilder):
        def push(self, token_ids):
            raise RuntimeError("boom")

    # Rebind the name looked up inside GenerationService._generate only; the
    # separate TokenPieceBuilder used by the `done` event's token_pieces()
    # helper (defined in app.token_pieces) is untouched, so that path keeps
    # working even though live scoring is broken.
    monkeypatch.setattr(services_module, "TokenPieceBuilder", BrokenTokenPieceBuilder)

    service, _ = make_generation_service()
    with caplog.at_level(logging.ERROR, logger="app.services"):
        events = list(service.begin("prompt", max_new_tokens=12, seed=None))

    token_events = [e for e in events if e.kind == "token"]
    assert token_events
    assert all(e.payload["tokens"] == [] for e in token_events)

    assert events[-1].kind == "done"
    done = events[-1].payload["detection"]
    assert done["tokens"] is not None
    assert len(done["tokens"]) == done["num_tokens"] + 1

    assert any(
        "live watermark scoring failed" in record.message for record in caplog.records
    )


def test_generation_service_rejects_third_queued_request():
    service, _ = make_generation_service()

    first = service.reserve_slot()
    second = service.reserve_slot()
    with pytest.raises(GenerationQueueFullError):
        service.reserve_slot()
    first.release()
    second.release()


def test_settings_require_hash_key():
    with pytest.raises(ValueError, match="WM_HASH_KEY"):
        ServiceSettings.from_env({"WM_MODEL_NAME": "local-model"})


def test_settings_default_to_qwen3_4b():
    settings = ServiceSettings.from_env({"WM_HASH_KEY": "12345"})

    assert settings.model_name == "Qwen/Qwen3-4B"


@pytest.mark.parametrize("hash_key", [0, -1, 2**63])
def test_settings_reject_hash_key_outside_watermark_range(hash_key):
    with pytest.raises(ValueError, match="WM_HASH_KEY"):
        ServiceSettings.from_env({"WM_HASH_KEY": str(hash_key)})


@pytest.mark.parametrize("max_new_tokens", [0, -1])
def test_settings_reject_non_positive_max_new_tokens(max_new_tokens):
    with pytest.raises(ValueError, match="WM_MAX_NEW_TOKENS"):
        ServiceSettings.from_env(
            {
                "WM_HASH_KEY": "12345",
                "WM_MAX_NEW_TOKENS": str(max_new_tokens),
            }
        )


def test_settings_build_one_watermark_config_from_environment():
    settings = ServiceSettings.from_env(
        {
            "WM_MODEL_NAME": "local-model",
            "WM_HASH_KEY": "12345",
            "WM_GAMMA": "0.3",
            "WM_DELTA": "1.5",
            "WM_Z_THRESHOLD": "3.5",
            "WM_MAX_NEW_TOKENS": "350",
        }
    )
    config = settings.watermark_config()

    assert settings.model_name == "local-model"
    assert config.hash_key == 12345
    assert config.gamma == 0.3
    assert config.delta == 1.5
    assert config.z_threshold == 3.5
    assert settings.max_tokens(401) == 350


def test_generation_request_clamps_only_at_service_boundary():
    from app.schemas import GenerateRequest

    request = GenerateRequest(prompt="test", max_new_tokens=999, seed=0)

    assert request.max_new_tokens == 999
    assert ServiceSettings.from_env({"WM_HASH_KEY": "12345"}).max_tokens(
        request.max_new_tokens
    ) == 999
    assert ServiceSettings.from_env({"WM_HASH_KEY": "12345"}).max_tokens(99999) == 4096
    assert ServiceSettings.from_env(
        {"WM_HASH_KEY": "12345", "WM_MAX_NEW_TOKENS": "8000"}
    ).max_tokens(99999) == 8000


def test_generation_request_without_max_new_tokens_uses_configured_maximum():
    from app.schemas import GenerateRequest

    request = GenerateRequest(prompt="test")

    assert request.max_new_tokens is None
    assert ServiceSettings.from_env(
        {"WM_HASH_KEY": "12345", "WM_MAX_NEW_TOKENS": "4096"}
    ).max_tokens(request.max_new_tokens) == 4096


@pytest.mark.parametrize("max_new_tokens", [0, -1])
def test_generation_request_rejects_non_positive_max_new_tokens(max_new_tokens):
    from app.schemas import GenerateRequest

    with pytest.raises(ValueError):
        GenerateRequest(prompt="test", max_new_tokens=max_new_tokens)


class FakeApiDetectionService:
    def __init__(self):
        self.calls = []
        self.tokenize_calls = []
        self.next_error = None

    def is_ready(self):
        return True

    def tokenize(self, text):
        self.tokenize_calls.append(text)
        if self.next_error:
            raise self.next_error
        from app.schemas import TokenizeResponse, TokenizedToken
        return TokenizeResponse(count=1, tokens=[TokenizedToken(index=0, id=1, text=text)])

    def classify(self, text, include_tokens=False, config=None):
        self.calls.append((text, include_tokens, config))
        if self.next_error:
            raise self.next_error
        from app.schemas import DetectionResponse, TokenDetail

        tokens = None
        if include_tokens:
            tokens = [
                TokenDetail(index=0, id=1, text="a", green=None, t=0, green_count=0, z=0.0)
            ]
        return DetectionResponse(
            verdict="inconclusive" if len(text) < 10 else "not_watermarked",
            num_tokens=len(text),
            green_count=0,
            z_score=0.0,
            p_value=1.0,
            threshold=4.0,
            tokens=tokens,
        )


class FakeApiGenerationService:
    def __init__(self):
        self.calls = []
        self.next_error = None
        self.emit_error = False
        self.start_loading_calls = 0

    def start_loading(self):
        self.start_loading_calls += 1

    def health(self):
        return True

    def begin(self, prompt, max_new_tokens, seed, config=None, inspect=False):
        self.calls.append(
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "seed": seed,
                "config": config,
                "inspect": inspect,
            }
        )
        if self.next_error:
            raise self.next_error
        yield GenerationEvent("token", {"text": "断片"})
        if self.emit_error:
            yield GenerationEvent("error", {"message": "generation failed"})
            return
        yield GenerationEvent(
            "done",
            {
                "full_text": "全文",
                "detection": {
                    "verdict": "inconclusive",
                    "num_tokens": 0,
                    "green_count": 0,
                    "z_score": 0.0,
                    "p_value": 1.0,
                    "threshold": 4.0,
                },
            },
        )


def make_api_client():
    from app.config import ServiceSettings
    from app.main import create_app

    generation = FakeApiGenerationService()
    detection = FakeApiDetectionService()
    settings = ServiceSettings(model_name="test-model", hash_key=17)
    return (
        TestClient(
            create_app(settings, generation, detection),
            raise_server_exceptions=False,
        ),
        generation,
        detection,
    )


def test_detect_endpoint_resolves_watermark_override():
    client, _, fake_detection = make_api_client()
    client.post(
        "/api/detect",
        json={"text": "十分に長いテキストです", "watermark": {"hash_key": 99, "gamma": 0.5}},
    )
    config = fake_detection.calls[-1][2]
    assert config.hash_key == 99
    assert config.gamma == 0.5
    assert config.delta == 2.0          # サーバー既定
    assert config.z_threshold == 4.0    # 上書き不可


def test_detect_endpoint_without_watermark_passes_default_config():
    client, _, fake_detection = make_api_client()
    client.post("/api/detect", json={"text": "十分に長いテキストです"})
    config = fake_detection.calls[-1][2]
    assert config.hash_key == 17


def test_generate_endpoint_resolves_watermark_override():
    client, fake_service, _ = make_api_client()
    response = client.post(
        "/api/generate", json={"prompt": "テスト", "watermark": {"delta": 0}}
    )
    assert response.status_code == 200
    config = fake_service.calls[-1]["config"]
    assert config.delta == 0.0
    assert config.hash_key == 17


@pytest.mark.parametrize("path,body", [
    ("/api/detect", {"text": "十分に長いテキストです", "watermark": {"hash_key": 0}}),
    ("/api/detect", {"text": "十分に長いテキストです", "watermark": {"gamma": 1.5}}),
    ("/api/generate", {"prompt": "テスト", "watermark": {"delta": -1}}),
])
def test_invalid_watermark_override_is_400(path, body):
    client, _, _ = make_api_client()
    response = client.post(path, json=body)
    assert response.status_code == 400
    assert response.json() == {"detail": "invalid request"}


def test_detect_endpoint_rejects_unknown_watermark_field():
    client, _, _ = make_api_client()
    response = client.post(
        "/api/detect",
        json={"text": "十分に長いテキストです", "watermark": {"gama": 0.5}},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "invalid request"}


def test_tokenize_endpoint_forwards_text_and_returns_json():
    client, _, fake_detection = make_api_client()
    response = client.post("/api/tokenize", json={"text": "こんにちは"})
    assert response.status_code == 200
    assert response.json() == {"count": 1, "tokens": [{"index": 0, "id": 1, "text": "こんにちは"}]}
    assert fake_detection.tokenize_calls == ["こんにちは"]


@pytest.mark.parametrize("body", [{"text": ""}, {"text": "  "}, {"text": "あ" * 10001}, {}])
def test_tokenize_endpoint_rejects_invalid_input(body):
    client, _, _ = make_api_client()
    response = client.post("/api/tokenize", json=body)
    assert response.status_code == 400


def test_tokenize_endpoint_is_503_when_tokenizer_missing():
    client, _, fake_detection = make_api_client()
    fake_detection.next_error = ModelNotReadyError("no tokenizer")
    response = client.post("/api/tokenize", json={"text": "こんにちは"})
    assert response.status_code == 503


def test_module_exports_asgi_app_with_test_only_environment():
    import app.main as main
    from fastapi import FastAPI

    assert isinstance(main.app, FastAPI)


def test_module_startup_without_hash_key_fails():
    environment = os.environ.copy()
    environment.pop("WM_HASH_KEY", None)
    result = subprocess.run(
        [sys.executable, "-c", "import app.main"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "WM_HASH_KEY" in result.stderr


def test_generate_route_is_sync_for_blocking_generation_admission():
    from app.main import create_app

    client, _, _ = make_api_client()
    route = next(route for route in client.app.routes if route.path == "/api/generate")

    assert not inspect.iscoroutinefunction(route.endpoint)


def test_detection_before_tokenizer_publication_is_503():
    from app.main import create_app

    settings = ServiceSettings(model_name="test-model", hash_key=17)
    generation = FakeApiGenerationService()
    detection = DetectionService(tokenizer=None, config=settings.watermark_config())
    client = TestClient(create_app(settings, generation, detection))

    response = client.post("/api/detect", json={"text": "short"})

    assert response.status_code == 503


def test_detect_short_text_is_inconclusive():
    client, _, _ = make_api_client()
    response = client.post("/api/detect", json={"text": "短文"})
    assert response.status_code == 200
    assert response.json()["verdict"] == "inconclusive"


def test_detect_returns_expected_statistics():
    client, _, _ = make_api_client()
    response = client.post("/api/detect", json={"text": "a sufficiently long text"})
    assert response.status_code == 200
    assert response.json()["threshold"] == 4.0
    assert "hash_key" not in response.json()


def test_detect_endpoint_forwards_include_tokens():
    client, _, fake_detection = make_api_client()
    client.post("/api/detect", json={"text": "十分に長いテキストです"})
    client.post("/api/detect", json={"text": "十分に長いテキストです", "include_tokens": True})
    assert [(c[0], c[1]) for c in fake_detection.calls[-2:]] == [
        ("十分に長いテキストです", False),
        ("十分に長いテキストです", True),
    ]


def test_detect_response_omits_tokens_key_when_not_requested():
    client, _, _ = make_api_client()
    response = client.post("/api/detect", json={"text": "十分に長いテキストです"})
    assert "tokens" not in response.json()


def test_detect_response_includes_token_details_when_requested():
    client, _, _ = make_api_client()
    response = client.post(
        "/api/detect", json={"text": "十分に長いテキストです", "include_tokens": True}
    )
    tokens = response.json()["tokens"]
    assert tokens[0]["id"] == 1
    assert tokens[0]["green"] is None


def test_generate_stream_contains_token_and_done_events():
    client, fake_service, _ = make_api_client()
    response = client.post(
        "/api/generate", json={"prompt": "テスト", "max_new_tokens": 999, "seed": 4}
    )
    assert response.status_code == 200
    assert "event: token" in response.text
    assert "event: done" in response.text
    assert fake_service.calls[-1]["max_new_tokens"] == 999


def test_generate_without_max_new_tokens_uses_configured_maximum():
    client, fake_service, _ = make_api_client()
    response = client.post("/api/generate", json={"prompt": "テスト"})
    assert response.status_code == 200
    assert fake_service.calls[-1]["max_new_tokens"] == 4096


def test_generate_clamps_max_new_tokens_to_configured_maximum():
    client, fake_service, _ = make_api_client()
    response = client.post(
        "/api/generate", json={"prompt": "テスト", "max_new_tokens": 99999}
    )
    assert response.status_code == 200
    assert fake_service.calls[-1]["max_new_tokens"] == 4096


def test_generate_stream_forwards_error_event_after_tokens():
    client, fake_service, _ = make_api_client()
    fake_service.emit_error = True
    response = client.post("/api/generate", json={"prompt": "test"})
    assert response.status_code == 200
    assert "event: error" in response.text
    assert 'data: {"message":"generation failed"}' in response.text


def test_health_hides_hash_key():
    client, _, _ = make_api_client()
    body = client.get("/api/health").json()
    assert body["model_name"] == "test-model"
    assert body["model_loaded"] is True
    assert "hash_key" not in body


def test_invalid_input_is_400_not_422():
    client, _, _ = make_api_client()
    response = client.post("/api/detect", json={"text": ""})
    assert response.status_code == 400


def test_generate_non_positive_max_new_tokens_is_400():
    client, _, _ = make_api_client()

    response = client.post(
        "/api/generate", json={"prompt": "test", "max_new_tokens": 0}
    )

    assert response.status_code == 400


def test_generate_prompt_over_2000_characters_is_400():
    client, _, _ = make_api_client()
    response = client.post("/api/generate", json={"prompt": "x" * 2001})
    assert response.status_code == 400


def test_detect_text_over_10000_characters_is_400():
    client, _, _ = make_api_client()
    response = client.post("/api/detect", json={"text": "x" * 10001})
    assert response.status_code == 400


def test_queue_overflow_is_429():
    client, fake_service, _ = make_api_client()
    fake_service.next_error = GenerationQueueFullError()
    response = client.post("/api/generate", json={"prompt": "test"})
    assert response.status_code == 429


def test_model_not_ready_is_503():
    client, fake_service, _ = make_api_client()
    fake_service.next_error = ModelNotReadyError()
    response = client.post("/api/generate", json={"prompt": "test"})
    assert response.status_code == 503


def test_unexpected_generation_error_is_generic_500_and_logged(caplog):
    client, fake_service, _ = make_api_client()
    fake_service.next_error = RuntimeError("secret details hash_key=17")
    response = client.post("/api/generate", json={"prompt": "test"})
    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}
    assert "hash_key=17" not in response.text
    assert "secret details" not in caplog.text
    assert "hash_key=17" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_root_serves_static_index():
    client, _, _ = make_api_client()
    response = client.get("/")
    assert response.status_code == 200
    assert "<!doctype html>" in response.text.lower()


def test_static_page_exposes_tabs_and_service_only_warning():
    client, _, _ = make_api_client()

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "生成" in response.text
    assert "判定" in response.text
    assert "鍵比較" in response.text
    assert "改ざん実験" in response.text
    assert "本サービスで生成されたテキストのみ判定可能(一般的なAI生成判定器ではない)" in response.text
    assert "同一トークナイザー・同一キーが必要です。" in response.text


def test_lifespan_starts_injected_service_once():
    _, fake_service, _ = make_api_client()
    assert fake_service.start_loading_calls == 0
    with TestClient(__import__("app.main", fromlist=["create_app"]).create_app(
        ServiceSettings(model_name="test-model", hash_key=17), fake_service,
        FakeApiDetectionService()
    )):
        assert fake_service.start_loading_calls == 1


def test_watermark_config_applies_only_provided_overrides():
    from app.schemas import WatermarkOverride

    settings = ServiceSettings.from_env(
        {"WM_HASH_KEY": "12345", "WM_GAMMA": "0.3", "WM_DELTA": "1.5", "WM_Z_THRESHOLD": "3.5"}
    )
    base = settings.watermark_config()
    assert (base.hash_key, base.gamma, base.delta, base.z_threshold) == (12345, 0.3, 1.5, 3.5)

    partial = settings.watermark_config(WatermarkOverride(hash_key=99, delta=0.0))
    assert (partial.hash_key, partial.gamma, partial.delta, partial.z_threshold) == (99, 0.3, 0.0, 3.5)

    assert settings.watermark_config(WatermarkOverride()) == base
    assert settings.watermark_config(None) == base


@pytest.mark.parametrize(
    "override",
    [{"hash_key": 0}, {"hash_key": -1}, {"gamma": 1.0}, {"gamma": 0.0}, {"delta": -0.5}],
)
def test_watermark_config_rejects_invalid_override(override):
    from app.schemas import WatermarkOverride

    settings = ServiceSettings.from_env({"WM_HASH_KEY": "12345"})
    with pytest.raises(ValueError):
        settings.watermark_config(WatermarkOverride(**override))


def test_requests_accept_optional_watermark_override():
    from app.schemas import DetectRequest, GenerateRequest

    assert GenerateRequest(prompt="x").watermark is None
    assert DetectRequest(text="x").watermark is None
    request = GenerateRequest(prompt="x", watermark={"hash_key": 5, "gamma": 0.5})
    assert request.watermark.hash_key == 5
    assert request.watermark.gamma == 0.5
    assert request.watermark.delta is None


def test_generation_uses_override_config_for_processor_and_scoring():
    service, model = make_generation_service()
    override = WatermarkConfig(hash_key=99, gamma=0.5, delta=3.0)

    events = list(service.begin("prompt", max_new_tokens=12, seed=None, config=override))

    processor = model.last_generation["logits_processor"][0]
    assert isinstance(processor, WatermarkLogitsProcessor)
    assert processor.config == override
    done = events[-1].payload["detection"]
    live = [tok for e in events if e.kind == "token" for tok in e.payload["tokens"]]
    # 上書き鍵で採点した結果と一致する（既定鍵とは一般に異なる）
    from watermark import WatermarkDetector
    ids = [tok["id"] for tok in done["tokens"]]
    expected = WatermarkDetector(len(service.tokenizer), override).detect_token_ids(ids)
    assert done["z_score"] == pytest.approx(expected.z_score)
    assert live[-1]["z"] == pytest.approx(expected.z_score)


def test_generation_without_override_uses_service_config():
    service, model = make_generation_service()
    list(service.begin("prompt", max_new_tokens=12, seed=None))
    assert model.last_generation["logits_processor"][0].config == service.config


def test_generation_with_zero_delta_skips_logits_processor():
    service, model = make_generation_service()
    plain = WatermarkConfig(hash_key=17, delta=0.0)

    events = list(service.begin("prompt", max_new_tokens=12, seed=None, config=plain))

    assert "logits_processor" not in model.last_generation
    assert events[-1].kind == "done"
    assert events[-1].payload["detection"]["threshold"] == 4.0


def test_detection_service_uses_override_config():
    from watermark import GreenListGenerator

    base = WatermarkConfig(hash_key=17)
    other = WatermarkConfig(hash_key=23)
    generator = GreenListGenerator(16, base)
    ids = [3]
    for _ in range(40):
        ids.append(int(generator.green_list(ids[-1])[0]))
    service = DetectionService(tokenizer=PieceTokenizer(ids), config=base)

    with_base = service.classify("x")
    with_other = service.classify("x", config=other)

    assert with_base.verdict == "watermarked"
    assert with_other.green_count < with_base.green_count
    assert with_other.z_score < with_base.z_score
