import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

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
        "WM_MODEL_NAME=Qwen/Qwen3-1.7B",
        "WM_GAMMA=0.25",
        "WM_DELTA=2.0",
        "WM_Z_THRESHOLD=4.0",
        "WM_MAX_NEW_TOKENS=400",
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


def test_detection_service_returns_inconclusive_for_short_text():
    tokenizer = FakeTokenizer()
    config = WatermarkConfig(hash_key=17)

    result = DetectionService(tokenizer=tokenizer, config=config).classify("short")

    assert result.verdict == "inconclusive"


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
    ) == 400


@pytest.mark.parametrize("max_new_tokens", [0, -1])
def test_generation_request_rejects_non_positive_max_new_tokens(max_new_tokens):
    from app.schemas import GenerateRequest

    with pytest.raises(ValueError):
        GenerateRequest(prompt="test", max_new_tokens=max_new_tokens)


class FakeApiDetectionService:
    def __init__(self):
        self.calls = []
        self.next_error = None

    def is_ready(self):
        return True

    def classify(self, text):
        self.calls.append(text)
        if self.next_error:
            raise self.next_error
        from app.schemas import DetectionResponse

        return DetectionResponse(
            verdict="inconclusive" if len(text) < 10 else "not_watermarked",
            num_tokens=len(text),
            green_count=0,
            z_score=0.0,
            p_value=1.0,
            threshold=4.0,
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

    def begin(self, prompt, max_new_tokens, seed):
        self.calls.append({"prompt": prompt, "max_new_tokens": max_new_tokens, "seed": seed})
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


def test_generate_stream_contains_token_and_done_events():
    client, fake_service, _ = make_api_client()
    response = client.post(
        "/api/generate", json={"prompt": "テスト", "max_new_tokens": 999, "seed": 4}
    )
    assert response.status_code == 200
    assert "event: token" in response.text
    assert "event: done" in response.text
    assert fake_service.calls[-1]["max_new_tokens"] == 400


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
