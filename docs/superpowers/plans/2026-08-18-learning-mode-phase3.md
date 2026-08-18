# 学習モード Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成ステップごとの候補 logit を記録・可視化する「logit インスペクタ」、任意文のトークン分割を見せる「トークナイザー」タブ、低/高エントロピーの prompt プリセットを追加する。

**Architecture:** `app/inspection.py` に `InspectingProcessor`（既存 `WatermarkLogitsProcessor` をラップし補正後 logit をそのまま返しつつ候補を記録）を置く。`GenerationService._generate(..., inspect)` が `inspect` 時にそれを使い、`done` に `steps` を載せる。`/api/tokenize` は `DetectionService` の tokenizer を再利用。UI は Phase 1/2 の描画関数を再利用し、チップクリックでインスペクタ、トークナイザータブ、プリセット `<select>` を追加する。

**Tech Stack:** Python 3.14 / FastAPI / pydantic v2 / torch / transformers 5.x / pytest / 素の HTML+JS+SVG

**Spec:** `docs/superpowers/specs/2026-08-18-learning-mode-phase3-design.md`

## Global Constraints

- `InspectingProcessor` が返す補正後 logit は `WatermarkLogitsProcessor` と完全に一致する。記録は先頭 400 ステップ、候補は元上位 10 ∪ 補正後上位 10、`prob` は補正後 / temperature(0.7) の softmax（語彙全体）。
- `done.steps[i]` は `done.detection.tokens[i]` に対応（特殊トークン除外で整列）。`inspect=false` なら `steps` は `null`。記録失敗時も生成は継続し `steps` は `null`（ログあり）。
- `/api/tokenize`: 空文字 / 10,000 文字超 → 400、tokenizer 未ロード → 503。
- 外部ライブラリなし。動的文字列は `escapeHtml`。触るファイルは各タスクの Files のみ。
- テストは `.venv/bin/python -m pytest tests -q`（現在 96 passed）。既存を壊さない。
- コミットメッセージ末尾に以下を付ける:
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Q6hPFyZT8RVVUWRdZVQaeE
  ```

---

## File Structure

| File | Responsibility |
|---|---|
| `app/inspection.py` (新規) | `Candidate`, `StepRecord`, `InspectingProcessor` |
| `app/schemas.py` | `GenerateRequest.inspect`, `CandidateDetail`, `StepInspection`, `TokenizeRequest`, `TokenizedToken`, `TokenizeResponse` |
| `app/services.py` | `_generate(..., inspect)`, `done.steps` 構築、`DetectionService.tokenize(text)` |
| `app/main.py` | `inspect` の受け渡し、`POST /api/tokenize` |
| `app/static/index.html` | インスペクタパネル、トークナイザータブ、プリセット |
| `tests/test_inspection.py` (新規), `tests/test_api.py` | テスト |
| `README.md` | 追記 |

---

### Task 1: `InspectingProcessor`

**Files:**
- Create: `app/inspection.py`
- Test: `tests/test_inspection.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class Candidate: id: int; raw: float; adjusted: float; green: bool; prob: float
  @dataclass
  class StepRecord: index: int; candidates: list[Candidate]; chosen_id: int | None = None
  class InspectingProcessor(LogitsProcessor):
      def __init__(self, inner: WatermarkLogitsProcessor, temperature: float = 0.7, top_n: int = 10, max_steps: int = 400)
      records: list[StepRecord]
      def __call__(self, input_ids, scores) -> scores  # inner と同じ補正後 logit を返す
  ```

- [ ] **Step 1: 失敗するテストを書く**（`tests/test_inspection.py`）

```python
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
```

- [ ] **Step 2: 失敗確認**

Run: `.venv/bin/python -m pytest tests/test_inspection.py -q`
Expected: ModuleNotFoundError

- [ ] **Step 3: 実装**（`app/inspection.py`）

```python
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import LogitsProcessor

from watermark import WatermarkLogitsProcessor


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

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raw = scores[0].detach().clone()
        adjusted_scores = self.inner(input_ids, scores)
        if len(self.records) < self.max_steps:
            try:
                self.records.append(self._record(len(self.records), input_ids, raw, adjusted_scores[0]))
            except Exception:  # recording must never break generation
                pass
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
        green_ids = set(self.inner.generator.green_list(previous_id).tolist())
        candidates = [
            Candidate(
                id=i,
                raw=float(raw_v[i]),
                adjusted=float(adj_v[i]),
                green=i in green_ids,
                prob=float(probs[i]),
            )
            for i in ids
        ]
        candidates.sort(key=lambda c: c.adjusted, reverse=True)
        return StepRecord(index=index, candidates=candidates)
```

- [ ] **Step 4: テスト**

Run: `.venv/bin/python -m pytest tests -q`
Expected: 全件 PASS（96 + 4）

- [ ] **Step 5: コミット**

```bash
git add app/inspection.py tests/test_inspection.py
git commit -m "Add InspectingProcessor that records per-step candidate logits"
```

---

### Task 2: 生成に `inspect` を通し `done.steps` を返す

**Files:**
- Modify: `app/schemas.py`（`GenerateRequest.inspect`, `CandidateDetail`, `StepInspection`）
- Modify: `app/services.py`（`begin`/`_generate` に `inspect`、`done` に `steps`）
- Modify: `app/main.py`（`payload.inspect` を渡す）
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `InspectingProcessor`, `StepRecord`（Task 1）
- Produces:
  ```python
  class GenerateRequest: ... inspect: bool = False
  class CandidateDetail(BaseModel): id:int; text:str; raw:float; adjusted:float; green:bool; prob:float
  class StepInspection(BaseModel): index:int; chosen_id:int; candidates:list[CandidateDetail]
  GenerationService.begin(self, prompt, max_new_tokens, seed, config=None, inspect: bool = False)
  # done payload: {"full_text", "detection", "steps": list[dict] | None}
  ```

- [ ] **Step 1: 失敗するテストを書く**（`tests/test_api.py` 末尾）

`FakeModel` は `logits_processor` を呼ばないので、呼ぶフェイクを追加する（`FakeModel` の直後）:

```python
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
```

テスト:

```python
def make_inspectable_service(continuation_ids=None):
    tokenizer = FakeTokenizer()
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


def test_generate_endpoint_forwards_inspect_flag():
    client, fake_service, _ = make_api_client()
    client.post("/api/generate", json={"prompt": "テスト"})
    client.post("/api/generate", json={"prompt": "テスト", "inspect": True})
    assert [c["inspect"] for c in fake_service.calls[-2:]] == [False, True]
```

`FakeApiGenerationService.begin` を `def begin(self, prompt, max_new_tokens, seed, config=None, inspect=False):` にして `"inspect": inspect` を calls に含める。既存 `test_generation_service_emits_tokens_then_done` などは `FakeModel`（processor を呼ばない）のままで良い。

- [ ] **Step 2: 失敗確認**

Run: `.venv/bin/python -m pytest tests/test_api.py -q -k "inspect"`
Expected: FAIL/TypeError

- [ ] **Step 3: 実装**

`app/schemas.py`:
```python
class GenerateRequest(BaseModel):
    ...
    inspect: bool = False


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
```

`app/services.py`（import に `from app.inspection import InspectingProcessor` と `from app.schemas import CandidateDetail, StepInspection` を追加）:

```python
    def begin(self, prompt, max_new_tokens, seed, config=None, inspect: bool = False):
        ...
            yield from self._generate(prompt, max_new_tokens, seed, config or self.config, inspect)

    def _generate(self, prompt, max_new_tokens, seed, config, inspect: bool = False):
        ...
        inspector: InspectingProcessor | None = None
        def worker():
            nonlocal inspector
            ...
                if inspect:
                    inspector = InspectingProcessor(
                        WatermarkLogitsProcessor(len(self.tokenizer), config), temperature=0.7
                    )
                    generate_kwargs["logits_processor"] = LogitsProcessorList([inspector])
                elif config.delta > 0:
                    generate_kwargs["logits_processor"] = LogitsProcessorList(
                        [WatermarkLogitsProcessor(len(self.tokenizer), config)]
                    )
        ...
        # done 構築の try 内、response の後:
            steps = None
            if inspector is not None:
                try:
                    steps = _build_steps(inspector.records, output_ids[prompt_length:], special_ids, self.tokenizer)
                except Exception:
                    logger.exception("logit inspection failed; continuing without steps")
                    steps = None
        ...
        yield GenerationEvent("done", {"full_text": full_text, "detection": response.model_dump(), "steps": steps})
```

モジュール関数:

```python
def _build_steps(records, generated_ids, special_ids, tokenizer) -> list[dict]:
    """Align per-step records with non-special generated tokens.

    records[k] was captured before generated_ids[k] was sampled. Special tokens
    (e.g. EOS) are dropped from both sides so steps[i] matches detection.tokens[i].
    """
    steps: list[dict] = []
    index = 0
    for k, token_id in enumerate(generated_ids):
        token_id = int(token_id)
        if token_id in special_ids:
            continue
        if k < len(records):
            record = records[k]
            candidates = [
                CandidateDetail(
                    id=c.id, text=token_pieces(tokenizer, [c.id])[0], raw=c.raw,
                    adjusted=c.adjusted, green=c.green, prob=c.prob,
                )
                for c in record.candidates
            ]
            steps.append(StepInspection(index=index, chosen_id=token_id, candidates=candidates).model_dump())
        index += 1
    return steps
```

`app/main.py`:
```python
        iterator = iter(generation.begin(
            payload.prompt,
            resolved_settings.max_tokens(payload.max_new_tokens),
            payload.seed,
            _resolve_watermark(payload.watermark),
            inspect=payload.inspect,
        ))
```

- [ ] **Step 4: 全テスト**

Run: `.venv/bin/python -m pytest tests -q` → 全件 PASS

- [ ] **Step 5: 実機確認**

サーバー再起動（Phase 2 と同じ手順）後:
```bash
curl -sN http://127.0.0.1:8000/api/generate -H 'content-type: application/json' \
  -d '{"prompt":"春について一文で。","max_new_tokens":20,"inspect":true}' | grep '^event: done' -A1 | tail -c 1200
```
Expected: `"steps":[{"index":0,"chosen_id":…,"candidates":[{"id":…,"text":…,"raw":…,"adjusted":…,"green":…,"prob":…}]}]`、`steps` の長さ == `detection.tokens` の長さ。

- [ ] **Step 6: コミット**

```bash
git add app/schemas.py app/services.py app/main.py tests/test_api.py
git commit -m "Return per-step candidate logits when generation is inspected"
```

---

### Task 3: `/api/tokenize`

**Files:**
- Modify: `app/schemas.py`（`TokenizeRequest`, `TokenizedToken`, `TokenizeResponse`）
- Modify: `app/services.py`（`DetectionService.tokenize`）
- Modify: `app/main.py`（`POST /api/tokenize`）
- Test: `tests/test_api.py`

**Interfaces:**
- Produces:
  ```python
  class TokenizeRequest(BaseModel): text: str   # 空/10000 超 → ValueError
  class TokenizedToken(BaseModel): index:int; id:int; text:str
  class TokenizeResponse(BaseModel): count:int; tokens:list[TokenizedToken]
  DetectionService.tokenize(self, text: str) -> TokenizeResponse   # 未ロード → ModelNotReadyError
  ```

- [ ] **Step 1: 失敗するテストを書く**

```python
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
```

`FakeApiDetectionService` に `self.tokenize_calls = []` と:
```python
    def tokenize(self, text):
        self.tokenize_calls.append(text)
        if self.next_error:
            raise self.next_error
        from app.schemas import TokenizeResponse, TokenizedToken
        return TokenizeResponse(count=1, tokens=[TokenizedToken(index=0, id=1, text=text)])
```

- [ ] **Step 2: 失敗確認** — `-k tokenize` → FAIL

- [ ] **Step 3: 実装**

`app/schemas.py`:
```python
class TokenizeRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        if len(value) > 10000:
            raise ValueError("text must not exceed 10000 characters")
        return value


class TokenizedToken(BaseModel):
    index: int
    id: int
    text: str


class TokenizeResponse(BaseModel):
    count: int
    tokens: list[TokenizedToken]
```

`app/services.py`（`DetectionService`）:
```python
    def tokenize(self, text: str) -> TokenizeResponse:
        if not self.is_ready():
            raise ModelNotReadyError("tokenizer is not ready")
        ids = self._token_ids(text)
        pieces = token_pieces(self.tokenizer, ids)
        return TokenizeResponse(
            count=len(ids),
            tokens=[TokenizedToken(index=i, id=t, text=p) for i, (t, p) in enumerate(zip(ids, pieces))],
        )
```

`app/main.py`:
```python
    @app.post("/api/tokenize", response_model=TokenizeResponse)
    def tokenize(payload: TokenizeRequest):
        return detection.tokenize(payload.text)
```

- [ ] **Step 4: 全テスト** → PASS

- [ ] **Step 5: 実機確認**（再起動後）
```bash
curl -s http://127.0.0.1:8000/api/tokenize -H 'content-type: application/json' -d '{"text":"日本の四季は美しい。"}'
```

- [ ] **Step 6: コミット**
```bash
git add app/schemas.py app/services.py app/main.py tests/test_api.py
git commit -m "Add /api/tokenize endpoint"
```

---

### Task 4: UI — logit インスペクタ（生成タブ）

**Files:**
- Modify: `app/static/index.html`

**Interfaces:**
- Consumes: `done.steps`（Task 2）、Phase 1 の `tokenChipMarkup` / `renderTokenChips` / `escapeHtml`
- Produces (JS): `tokenChipMarkup` に `data-index` 属性、`renderInspector(el, step, delta)`、生成タブの `lastSteps`

- [ ] **Step 1: マークアップと CSS**

生成パネルの `<div class="viz">` の直前に:
```html
<div id="inspector" class="inspector" hidden>
  <h3 id="inspector-title">ステップの候補</h3>
  <p class="legend">チップをクリックすると、そのトークンを選ぶ直前の候補（元 logit 上位 10 ∪ 補正後上位 10）を表示します。確率は temperature 0.7 適用後の softmax（top-p 0.8 / top-k 20 の打ち切り前）。</p>
  <div id="inspector-body"></div>
</div>
```

CSS:
```css
.inspector { margin-top: 1.25rem; padding: .85rem 1rem; border: 1px solid var(--line); border-radius: .55rem; background: #fff; }
.inspector h3 { margin: 0 0 .3rem; font-size: 1rem; }
.cand { display: grid; grid-template-columns: 8rem 1fr 4.5rem 4.5rem 4rem; gap: .5rem; align-items: center; padding: .25rem 0; border-bottom: 1px solid #eef1f6; font-variant-numeric: tabular-nums; font-size: .9rem; }
.cand.head { color: var(--muted); font-size: .8rem; }
.cand.chosen { background: #fff8e6; }
.cand .tokcell { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre; overflow: hidden; text-overflow: ellipsis; }
.bar { position: relative; height: 12px; background: #eef1f6; border-radius: 3px; overflow: hidden; }
.bar .raw { position: absolute; left: 0; top: 0; bottom: 0; background: #8b95a7; }
.bar .boost { position: absolute; top: 0; bottom: 0; background: #087443; }
.tok.selected { outline: 2px solid var(--brand); }
```

- [ ] **Step 2: JS**

`tokenChipMarkup(tok)` の `<span class="tok …"` に `data-index="' + tok.index + '"` を追加（既存の呼び出しはすべて `index` を持つ）。

```js
const inspector = document.getElementById('inspector');
const inspectorTitle = document.getElementById('inspector-title');
const inspectorBody = document.getElementById('inspector-body');
let lastSteps = null;   // 生成ハンドラの done で payload.steps を保存、開始時に null
let lastDelta = null;   // done で settings.delta（/api/health の delta を settings に追加）

function renderInspector(step, delta) {
  if (!step) {
    inspectorTitle.textContent = 'このステップは記録されていません（先頭 400 ステップのみ）';
    inspectorBody.innerHTML = '';
    inspector.hidden = false;
    return;
  }
  const cands = step.candidates.slice();
  const min = Math.min(...cands.map((c) => Math.min(c.raw, c.adjusted)));
  const max = Math.max(...cands.map((c) => Math.max(c.raw, c.adjusted)));
  const span = (max - min) || 1;
  const pct = (v) => (((v - min) / span) * 100).toFixed(1) + '%';
  inspectorTitle.textContent = 'ステップ ' + step.index + ' の候補（選ばれたトークン: ID ' + step.chosen_id + '）';
  let html = '<div class="cand head"><span>候補</span><span>logit（灰: 元、緑: +δ）</span><span>元</span><span>補正後</span><span>確率</span></div>';
  cands.forEach((c) => {
    const chosen = c.id === step.chosen_id ? ' chosen' : '';
    const boostLeft = pct(Math.min(c.raw, c.adjusted));
    const boostWidth = (((Math.abs(c.adjusted - c.raw)) / span) * 100).toFixed(1) + '%';
    html += '<div class="cand' + chosen + '">' +
      '<span class="tokcell">' + escapeHtml(c.text === '' ? '·' : c.text) + ' <span class="muted">#' + c.id + '</span>' + (c.green ? ' <span class="tok green">G</span>' : '') + '</span>' +
      '<span class="bar"><span class="raw" style="width:' + pct(c.raw) + '"></span>' + (c.green && c.adjusted !== c.raw ? '<span class="boost" style="left:' + boostLeft + ';width:' + boostWidth + '"></span>' : '') + '</span>' +
      '<span>' + c.raw.toFixed(2) + '</span><span>' + c.adjusted.toFixed(2) + '</span><span>' + (c.prob * 100).toFixed(1) + '%</span></div>';
  });
  inspectorBody.innerHTML = html;
  inspector.hidden = false;
}

generatedTokens.addEventListener('click', (ev) => {
  const chip = ev.target.closest('.tok[data-index]');
  if (!chip || !lastSteps) return;
  generatedTokens.querySelectorAll('.tok.selected').forEach((el) => el.classList.remove('selected'));
  chip.classList.add('selected');
  const index = Number(chip.dataset.index);
  renderInspector(lastSteps.find((s) => s.index === index) || null, lastDelta);
});
```

生成ハンドラ: リクエスト body に `inspect: true`；開始時 `lastSteps = null; inspector.hidden = true;`；`done` で `lastSteps = payload.steps || null;` と、`lastSteps` があれば凡例に「チップをクリックすると候補を表示」と分かるよう `#generated-tokens` に `title` を付けるだけで良い。`settings` に `delta` を保持（`/api/health` の `h.delta`）。

- [ ] **Step 3: ブラウザで確認**（Playwright）: 生成 → チップクリック → 候補テーブルが出て、選ばれたトークンの行が強調、Green 行に緑バーの伸び。400 超のクリックは長文が必要なので省略可（コードレビューで確認）。

- [ ] **Step 4: コミット**
```bash
git add app/static/index.html
git commit -m "Add logit inspector panel to the generate tab"
```

---

### Task 5: UI — トークナイザータブ

**Files:**
- Modify: `app/static/index.html`

**Interfaces:**
- Consumes: `POST /api/tokenize`（Task 3）
- Produces (JS): `tokenizeWith(text)`, `lcsDiff(a, b)`, `renderTokenList(container, tokens, onlySet)`, `activateTab('tokenizer')`

- [ ] **Step 1: マークアップと CSS**

タブ: `<button class="tab" type="button" data-tab="tokenizer" aria-selected="false" aria-controls="tokenizer-panel">トークナイザー</button>`

パネル（改ざん実験の後）:
```html
<section id="tokenizer-panel" class="panel" data-panel="tokenizer" hidden>
  <h2>トークナイザーを見る</h2>
  <p class="intro">文章がどのようなトークンに分割され、どの ID になるかを表示します。テキスト B を入れると A との差分（変わったトークン）を色分けします。透かしの判定は「トークン列」に対して行われるため、分割が変わると判定にも影響します。</p>
  <label for="tokenize-a">テキスト A</label>
  <textarea id="tokenize-a" maxlength="10000" placeholder="例: 日本の四季は美しい。"></textarea>
  <p class="legend" id="tokenize-a-count"></p>
  <div id="tokenize-a-tokens" class="tokens toklist"></div>
  <label for="tokenize-b">テキスト B（任意・比較用）</label>
  <textarea id="tokenize-b" maxlength="10000" placeholder="A を少し書き換えた文章を入れると差分が分かります"></textarea>
  <p class="legend" id="tokenize-b-count"></p>
  <div id="tokenize-b-tokens" class="tokens toklist"></div>
  <p class="legend" id="tokenize-diff"></p>
  <p class="legend" id="tokenize-error"></p>
</section>
```

CSS:
```css
.toklist .tk { display: inline-block; margin: 2px 1px; padding: .1rem .3rem; border-radius: .3rem; background: #e9eef7; line-height: 1.3; white-space: pre; }
.toklist .tk:nth-child(even) { background: #d7e0f5; }
.toklist .tk small { display: block; color: var(--muted); font-size: .7rem; text-align: center; }
.toklist .tk.only-a { background: #ffd9d4; }
.toklist .tk.only-b { background: #d3f5df; }
```

- [ ] **Step 2: JS**

```js
const tokenizeA = document.getElementById('tokenize-a');
const tokenizeB = document.getElementById('tokenize-b');
const tokenizeState = { a: null, b: null, seq: 0, timer: null };

async function tokenizeWith(text) {
  const response = await fetch('/api/tokenize', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) });
  if (!response.ok) throw new Error('トークン化に失敗しました。');
  return response.json();
}

function lcsDiff(a, b) {           // a, b: id 配列 → {onlyA:Set<index>, onlyB:Set<index>, common:number}
  const n = a.length, m = b.length;
  if (n * m > 4_000_000) return null;
  const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const onlyA = new Set(), onlyB = new Set();
  let i = 0, j = 0, common = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { i++; j++; common++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { onlyA.add(i); i++; }
    else { onlyB.add(j); j++; }
  }
  while (i < n) onlyA.add(i++);
  while (j < m) onlyB.add(j++);
  return { onlyA, onlyB, common };
}

function renderTokenList(container, tokens, onlySet, cls) {
  container.innerHTML = tokens.map((t) =>
    '<span class="tk' + (onlySet && onlySet.has(t.index) ? ' ' + cls : '') + '" title="ID ' + t.id + '">' +
    escapeHtml(t.text === '' ? '·' : t.text) + '<small>' + t.id + '</small></span>').join('');
}

function renderTokenizer() {
  const a = tokenizeState.a, b = tokenizeState.b;
  document.getElementById('tokenize-a-count').textContent = a ? 'トークン数: ' + a.count : '';
  document.getElementById('tokenize-b-count').textContent = b ? 'トークン数: ' + b.count : '';
  const diffEl = document.getElementById('tokenize-diff');
  if (a && b) {
    const diff = lcsDiff(a.tokens.map((t) => t.id), b.tokens.map((t) => t.id));
    if (diff) {
      renderTokenList(document.getElementById('tokenize-a-tokens'), a.tokens, diff.onlyA, 'only-a');
      renderTokenList(document.getElementById('tokenize-b-tokens'), b.tokens, diff.onlyB, 'only-b');
      diffEl.textContent = '共通トークン ' + diff.common + ' / A のみ ' + diff.onlyA.size + ' / B のみ ' + diff.onlyB.size;
      return;
    }
    diffEl.textContent = '長すぎるため差分は計算しません。';
  } else diffEl.textContent = '';
  renderTokenList(document.getElementById('tokenize-a-tokens'), a ? a.tokens : []);
  renderTokenList(document.getElementById('tokenize-b-tokens'), b ? b.tokens : []);
}

async function runTokenize() {
  const seq = ++tokenizeState.seq;
  const ta = tokenizeA.value, tb = tokenizeB.value;
  try {
    const [ra, rb] = await Promise.all([ta.trim() ? tokenizeWith(ta) : null, tb.trim() ? tokenizeWith(tb) : null]);
    if (seq !== tokenizeState.seq) return;
    tokenizeState.a = ra; tokenizeState.b = rb;
    document.getElementById('tokenize-error').textContent = '';
    renderTokenizer();
  } catch (error) {
    if (seq !== tokenizeState.seq) return;
    document.getElementById('tokenize-error').textContent = error.message;
  }
}
function scheduleTokenize() { clearTimeout(tokenizeState.timer); tokenizeState.timer = setTimeout(runTokenize, 300); }
tokenizeA.addEventListener('input', scheduleTokenize);
tokenizeB.addEventListener('input', scheduleTokenize);
```

- [ ] **Step 3: ブラウザで確認**: A に文章 → チップ＋ID＋トークン数。B に 1 文字変えた文章 → 差分着色と件数。

- [ ] **Step 4: コミット**
```bash
git add app/static/index.html
git commit -m "Add tokenizer visualization tab"
```

---

### Task 6: UI — prompt プリセット

**Files:**
- Modify: `app/static/index.html`

- [ ] **Step 1: マークアップ**

生成タブの `<label for="prompt">` の直前と、鍵比較タブの `<label for="compare-prompt">` の直前に:
```html
<div class="preset-row">
  <label>プリセット <select class="preset-select" data-target="prompt"><option value="">（選択）</option></select></label>
  <span class="legend preset-hint"></span>
</div>
```
（鍵比較側は `data-target="compare-prompt"`）

CSS: `.preset-row { display: flex; gap: .75rem; align-items: center; flex-wrap: wrap; margin-bottom: .5rem; } .preset-row select { padding: .3rem .4rem; border: 1px solid #aeb9c9; border-radius: .4rem; font: inherit; }`

- [ ] **Step 2: JS**

```js
const PRESETS = [
  { group: '低エントロピー（z が伸びにくい）', hint: '定型的な出力では次のトークンがほぼ決まっていて Green を選ぶ余地がなく、z が伸びにくい', items: [
    '1 から 100 まで数字を改行区切りで書いてください。',
    'Python で FizzBuzz を書いてください。',
    '五十音（あいうえお…）を順に、間に空白を入れて書いてください。',
  ] },
  { group: '高エントロピー（z が伸びやすい）', hint: '自由度が高い文章では Green を選びやすく、z が伸びやすい', items: [
    '架空の街を舞台にした短い物語を 400 字程度で書いてください。',
    '日本の四季それぞれの魅力を段落ごとに説明してください。',
    '旅行の計画の立て方について、初心者向けにアドバイスしてください。',
  ] },
];
document.querySelectorAll('.preset-select').forEach((select) => {
  PRESETS.forEach((g, gi) => {
    const og = document.createElement('optgroup'); og.label = g.group;
    g.items.forEach((p, pi) => { const o = document.createElement('option'); o.value = gi + ':' + pi; o.textContent = p; og.appendChild(o); });
    select.appendChild(og);
  });
  select.addEventListener('change', () => {
    const [gi, pi] = select.value.split(':').map(Number);
    const hint = select.parentElement.parentElement.querySelector('.preset-hint');
    if (Number.isNaN(gi)) { hint.textContent = ''; return; }
    document.getElementById(select.dataset.target).value = PRESETS[gi].items[pi];
    hint.textContent = PRESETS[gi].hint;
  });
});
```

- [ ] **Step 3: ブラウザで確認**: 両タブでプリセット選択 → prompt に反映、ヒント表示。

- [ ] **Step 4: コミット**
```bash
git add app/static/index.html
git commit -m "Add low/high entropy prompt presets"
```

---

### Task 7: README 更新と最終確認

**Files:**
- Modify: `README.md`

- [ ] **Step 1:** 「Use the API」に:
```
- `POST /api/generate` accepts optional `inspect` (default `false`). When true, the `done` event also carries `steps`: for each of the first 400 generated tokens, the candidate tokens (top 10 by raw logit ∪ top 10 by watermark-adjusted logit) with `raw`, `adjusted`, `green` and `prob` (softmax of adjusted/temperature over the whole vocabulary, before top-k/top-p). `steps[i]` corresponds to `detection.tokens[i]`.
- `POST /api/tokenize` accepts `text` (1–10,000 characters) and returns the token ids and display pieces produced by the service tokenizer; it needs only the tokenizer, not the model weights.
```
「Use the browser UI」に:
```
On the 生成 tab, click a token chip after generation to open the logit inspector for that step. The トークナイザー tab shows how any text is split into tokens (with ids) and, given a second text, highlights the tokens that differ. Both the 生成 and 鍵比較 tabs offer low- and high-entropy prompt presets that illustrate when the watermark signal is weak or strong.
```

- [ ] **Step 2:** `.venv/bin/python -m pytest tests -q` → 全件 PASS；サーバー稼働確認。

- [ ] **Step 3:**
```bash
git add README.md
git commit -m "Document the logit inspector, tokenizer endpoint and prompt presets"
```
