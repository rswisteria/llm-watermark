# 学習モード Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成・判定 API に透かしパラメータ上書き（`watermark`）を追加し、UI に「鍵比較」タブ（鍵A / 鍵B / 透かしなし の順次生成と相互検出マトリクス）と「改ざん実験」タブ（編集のたびに z を再計算）を追加する。

**Architecture:** `WatermarkOverride` スキーマ → `ServiceSettings.watermark_config(override)` で解決済み `WatermarkConfig` を作り、`GenerationService.begin(..., config)` / `DetectionService.classify(..., config)` に渡す。`delta == 0` は LogitsProcessor を付けない。UI はサーバーを増やさず、既存 SSE / detect API をクライアントが順に呼んで 3 レーンと 3×3 マトリクスを組み立てる。Phase 1 の描画関数（`renderTokenChips` / `renderZChart` / `renderFormula` / `readSse`）を再利用。

**Tech Stack:** Python 3.14 / FastAPI / pydantic v2 / pytest / 素の HTML+JS+SVG（外部ライブラリなし）

**Spec:** `docs/superpowers/specs/2026-08-18-learning-mode-phase2-design.md`

## Global Constraints

- `watermark` は省略可能（既定 `None`）。省略値はサーバー設定を使う。`z_threshold` は上書き不可。不正値は `400 {"detail": "invalid request"}`。
- `delta == 0` のとき生成に `WatermarkLogitsProcessor` を付けない（`logits_processor` 引数自体を渡さない）。
- 同一リクエスト内では、ライブ判定・`done.detection`・`/api/detect` は同じ解決済み config を使う。
- 鍵はログ・レスポンス・ヘルスに出さない。
- 外部ライブラリなし。動的文字列は `escapeHtml` 経由。触るファイルは各タスクの Files のみ。
- テストは `.venv/bin/python -m pytest tests -q`（現在 78 passed）。既存を壊さない。
- コミットメッセージ末尾に以下を付ける:
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Q6hPFyZT8RVVUWRdZVQaeE
  ```

---

## File Structure

| File | Responsibility |
|---|---|
| `app/schemas.py` | `WatermarkOverride`、`GenerateRequest.watermark`、`DetectRequest.watermark` |
| `app/config.py` | `ServiceSettings.watermark_config(override=None)` — 上書きを解決して `WatermarkConfig` を返す |
| `app/services.py` | `GenerationService.begin(..., config=None)`、`_generate(..., config)`（delta=0 で processor なし）、`DetectionService.classify(..., config=None)` |
| `app/main.py` | 両エンドポイントで `watermark` を解決してサービスに渡す。`ValueError` → 400 |
| `app/static/index.html` | 「鍵比較」「改ざん実験」タブ、持ち込みボタン |
| `tests/test_api.py` | 上記のテスト |
| `README.md` | API 追記・タブ説明・リスク注記 |

---

### Task 1: `WatermarkOverride` スキーマと `watermark_config(override)`

**Files:**
- Modify: `app/schemas.py`
- Modify: `app/config.py:50-57`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces:
  ```python
  class WatermarkOverride(BaseModel):
      hash_key: int | None = None
      gamma: float | None = None
      delta: float | None = None
  class GenerateRequest: ... watermark: WatermarkOverride | None = None
  class DetectRequest:   ... watermark: WatermarkOverride | None = None
  ServiceSettings.watermark_config(self, override: WatermarkOverride | None = None) -> WatermarkConfig
      # None でない値だけ差し替え。WatermarkConfig.__post_init__ の検証で ValueError
  ```

- [ ] **Step 1: 失敗するテストを書く**（`tests/test_api.py` 末尾に追記）

```python
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
```

- [ ] **Step 2: 失敗確認**

Run: `.venv/bin/python -m pytest tests/test_api.py -q -k "watermark_config or optional_watermark"`
Expected: ImportError / AttributeError

- [ ] **Step 3: 実装**

`app/schemas.py`（`GenerateRequest` の前に追加、両リクエストにフィールド追加）:

```python
class WatermarkOverride(BaseModel):
    """Optional per-request watermark parameters for the learning UI.

    Missing values fall back to the server settings; validation happens in
    WatermarkConfig so the rules stay in one place.
    """

    hash_key: int | None = None
    gamma: float | None = None
    delta: float | None = None


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int | None = None
    seed: int | None = None
    watermark: WatermarkOverride | None = None
    ...

class DetectRequest(BaseModel):
    text: str
    include_tokens: bool = False
    watermark: WatermarkOverride | None = None
    ...
```

`app/config.py`:

```python
from app.schemas import WatermarkOverride   # 先頭 import に追加（循環なし: schemas は config を import しない）

    def watermark_config(self, override: WatermarkOverride | None = None) -> WatermarkConfig:
        override = override or WatermarkOverride()
        return WatermarkConfig(
            gamma=self.gamma if override.gamma is None else override.gamma,
            delta=self.delta if override.delta is None else override.delta,
            hash_key=self.hash_key if override.hash_key is None else override.hash_key,
            z_threshold=self.z_threshold,
        )
```

- [ ] **Step 4: 全テスト**

Run: `.venv/bin/python -m pytest tests -q`
Expected: 全件 PASS（78 + 7）

- [ ] **Step 5: コミット**

```bash
git add app/schemas.py app/config.py tests/test_api.py
git commit -m "Add optional per-request watermark override schema"
```

---

### Task 2: サービス層で上書き config を使う

**Files:**
- Modify: `app/services.py`（`DetectionService.classify`, `GenerationService.begin`, `_generate`）
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `WatermarkConfig`（既存）
- Produces:
  ```python
  DetectionService.classify(self, text: str, include_tokens: bool = False, config: WatermarkConfig | None = None)
  GenerationService.begin(self, prompt, max_new_tokens, seed, config: WatermarkConfig | None = None)
  GenerationService._generate(self, prompt, max_new_tokens, seed, config: WatermarkConfig)
  ```

- [ ] **Step 1: 失敗するテストを書く**（`tests/test_api.py` 末尾に追記。`FakeModel.generate` は `**kwargs` を `last_generation` に保存しているので `logits_processor` の有無と中身を見られる）

```python
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
```

- [ ] **Step 2: 失敗確認**

Run: `.venv/bin/python -m pytest tests/test_api.py -q -k "override_config or zero_delta or without_override"`
Expected: TypeError（`config` 引数なし）

- [ ] **Step 3: 実装**

`app/services.py`:

```python
class DetectionService:
    ...
    def classify(
        self,
        text: str,
        include_tokens: bool = False,
        config: WatermarkConfig | None = None,
    ) -> DetectionResponse:
        if not self.is_ready():
            raise ModelNotReadyError("tokenizer is not ready")
        return _classify_token_ids(
            self._token_ids(text),
            len(self.tokenizer),
            config or self.config,
            tokenizer=self.tokenizer,
            include_tokens=include_tokens,
        )


class GenerationService:
    ...
    def begin(
        self,
        prompt: str,
        max_new_tokens: int,
        seed: int | None,
        config: WatermarkConfig | None = None,
    ) -> Iterator[GenerationEvent]:
        if not self.health():
            raise ModelNotReadyError("model is not ready")
        lease = self.reserve_slot()
        self._generation.acquire()
        try:
            yield from self._generate(prompt, max_new_tokens, seed, config or self.config)
        finally:
            self._generation.release()
            lease.release()

    def _generate(
        self, prompt: str, max_new_tokens: int, seed: int | None, config: WatermarkConfig
    ) -> Iterator[GenerationEvent]:
        ...
        def worker() -> None:
            try:
                if seed is not None:
                    torch.manual_seed(seed)
                generate_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                )
                if config.delta > 0:
                    generate_kwargs["logits_processor"] = LogitsProcessorList(
                        [WatermarkLogitsProcessor(len(self.tokenizer), config)]
                    )
                result_holder["output"] = self.model.generate(
                    **inputs, streamer=streamer, **generate_kwargs
                )
            ...
        scorer = IncrementalScorer(vocab_size, config)          # self.config → config
        ...
            response = _classify_token_ids(
                token_ids, vocab_size, config,                  # self.config → config
                tokenizer=self.tokenizer, include_tokens=True,
            )
```

`_generate` 内の `self.config` 参照を **すべて** `config` に置き換える（`grep -n "self.config" app/services.py` で `_generate` 内に残っていないことを確認）。

- [ ] **Step 4: 全テスト**

Run: `.venv/bin/python -m pytest tests -q`
Expected: 全件 PASS

- [ ] **Step 5: コミット**

```bash
git add app/services.py tests/test_api.py
git commit -m "Let generation and detection services take a per-request watermark config"
```

---

### Task 3: エンドポイントで `watermark` を解決して渡す

**Files:**
- Modify: `app/main.py`（`detect`, `generate`）
- Test: `tests/test_api.py`（`FakeApiDetectionService.classify`, `FakeApiGenerationService.begin` のシグネチャ更新＋テスト追加）

**Interfaces:**
- Consumes: Task 1 の `watermark_config(override)`、Task 2 の `config` 引数
- Produces: `POST /api/generate` / `POST /api/detect` が `watermark` を受け付け、解決済み config をサービスに渡す。不正値は 400。

- [ ] **Step 1: 失敗するテストを書く**

`FakeApiDetectionService.classify` を `def classify(self, text, include_tokens=False, config=None):` にし `self.calls.append((text, include_tokens, config))` に変更。既存テストで `calls[-2:] == [("…", False), ("…", True)]` と比較している箇所は `[(c[0], c[1]) for c in fake_detection.calls[-2:]]` に直す。
`FakeApiGenerationService.begin` を `def begin(self, prompt, max_new_tokens, seed, config=None):` にし、`self.calls.append({... , "config": config})` に変更。

追記:

```python
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
```

- [ ] **Step 2: 失敗確認**

Run: `.venv/bin/python -m pytest tests/test_api.py -q -k "watermark_override or without_watermark"`
Expected: FAIL（config が渡らない / 400 にならない）

- [ ] **Step 3: 実装**（`app/main.py`）

```python
    def _resolve_watermark(override):
        try:
            return resolved_settings.watermark_config(override)
        except ValueError:
            raise InvalidWatermarkOverride() from None

    class InvalidWatermarkOverride(ValueError):
        pass

    @app.exception_handler(InvalidWatermarkOverride)
    async def invalid_watermark_handler(request: Request, exc: InvalidWatermarkOverride):
        return JSONResponse(status_code=400, content={"detail": "invalid request"})

    @app.post("/api/detect", response_model=DetectionResponse)
    async def detect(payload: DetectRequest):
        result = detection.classify(
            payload.text,
            include_tokens=payload.include_tokens,
            config=_resolve_watermark(payload.watermark),
        )
        ...（以降既存のまま）

    @app.post("/api/generate")
    def generate(payload: GenerateRequest):
        iterator = iter(generation.begin(
            payload.prompt,
            resolved_settings.max_tokens(payload.max_new_tokens),
            payload.seed,
            _resolve_watermark(payload.watermark),
        ))
        ...
```

`InvalidWatermarkOverride` クラスと `_resolve_watermark` は `create_app` の中（`app = FastAPI(...)` の後、ハンドラ登録の並び）に置く。鍵の値をログに出さないこと。

- [ ] **Step 4: 全テスト**

Run: `.venv/bin/python -m pytest tests -q`
Expected: 全件 PASS

- [ ] **Step 5: 実機確認**

```bash
kill $(lsof -tiTCP:8000 -sTCP:LISTEN); sleep 2
set -a; source .env; set +a
nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 > /private/tmp/claude-501/-Users-toyota-PycharmProjects-llm-watermark/ddf726b3-80ee-434e-a9ed-86c7557b06ed/scratchpad/uvicorn.log 2>&1 &
# model_loaded:true まで待つ
curl -s http://127.0.0.1:8000/api/detect -H 'content-type: application/json' \
  -d '{"text":"<Phase 1 で生成した透かし付き文章>","watermark":{"hash_key":123456}}'
```
Expected: 既定鍵では `watermarked` の文章が、別鍵では `z` が小さく `not_watermarked` になる。

- [ ] **Step 6: コミット**

```bash
git add app/main.py tests/test_api.py
git commit -m "Accept per-request watermark overrides on generate and detect endpoints"
```

---

### Task 4: UI「鍵比較」タブ

**Files:**
- Modify: `app/static/index.html`

**Interfaces:**
- Consumes: `POST /api/generate` (`watermark`, SSE `token.tokens` / `done.detection`), `POST /api/detect` (`include_tokens`, `watermark`), Phase 1 の `renderTokenChips` / `resetTokenView` / `renderZChart` / `renderFormula` / `readSse` / `escapeHtml` / `verdictLabel` / `settings`
- Produces (JS): `laneOverride(lane)`（レーンカードの入力から `watermark` オブジェクト or `undefined` を作る）、`runComparison()`、`renderMatrix()`、`showLaneUnderKey(row, col)`、`window.__wmTamperImport`（Task 5 が使うフック: `{text, watermark, label}` を受け取り改ざん実験タブへ持ち込む。Task 4 では `sendToTamper(payload)` を呼ぶボタンだけ置き、関数本体は Task 5 で定義。Task 4 時点では `typeof sendToTamper === 'function'` のときだけ有効化）

- [ ] **Step 1: マークアップと CSS**

タブに追加:
```html
<button class="tab" type="button" data-tab="compare" aria-selected="false" aria-controls="compare-panel">鍵比較</button>
```

判定パネルの後に:
```html
<section id="compare-panel" class="panel" data-panel="compare" hidden>
  <h2>鍵を変えて比較する</h2>
  <p class="intro">同じプロンプト・seed で「鍵A / 鍵B / 透かしなし」を順に生成し、各文章を各鍵で判定します。別の鍵で見ると Green 比率が γ に戻り、透かしは検出できません。</p>
  <label for="compare-prompt">プロンプト</label>
  <textarea id="compare-prompt" maxlength="2000" placeholder="例: 日本の四季について短く説明してください。"></textarea>
  <div class="compare-controls">
    <label>seed <input id="compare-seed" type="number" min="0" step="1"></label>
    <label>max_new_tokens <input id="compare-max-tokens" type="number" min="1" step="1" value="150"></label>
  </div>
  <div class="lanes" id="lanes">
    <div class="lane" data-lane="0">
      <h3>鍵A（サーバー既定）</h3>
      <div class="lane-params">
        <label>γ <input class="lane-gamma" type="number" min="0.01" max="0.99" step="0.01"></label>
        <label>δ <input class="lane-delta" type="number" min="0" step="0.1"></label>
      </div>
      <div class="tokens lane-tokens"></div>
      <svg class="chart lane-chart" viewBox="0 0 640 220" preserveAspectRatio="xMidYMid meet" role="img" aria-label="z 値の推移"></svg>
      <output class="result lane-result" hidden></output>
      <button class="lane-tamper" type="button" hidden>この文章で改ざん実験</button>
    </div>
    <div class="lane" data-lane="1">
      <h3>鍵B</h3>
      <div class="lane-params">
        <label>鍵 <input class="lane-key" type="number" min="1" step="1"></label>
        <label>γ <input class="lane-gamma" type="number" min="0.01" max="0.99" step="0.01"></label>
        <label>δ <input class="lane-delta" type="number" min="0" step="0.1"></label>
      </div>
      <div class="tokens lane-tokens"></div>
      <svg class="chart lane-chart" viewBox="0 0 640 220" preserveAspectRatio="xMidYMid meet" role="img" aria-label="z 値の推移"></svg>
      <output class="result lane-result" hidden></output>
      <button class="lane-tamper" type="button" hidden>この文章で改ざん実験</button>
    </div>
    <div class="lane" data-lane="2">
      <h3>透かしなし（δ = 0）</h3>
      <div class="lane-params"><span class="muted">既定鍵で判定します</span></div>
      <div class="tokens lane-tokens"></div>
      <svg class="chart lane-chart" viewBox="0 0 640 220" preserveAspectRatio="xMidYMid meet" role="img" aria-label="z 値の推移"></svg>
      <output class="result lane-result" hidden></output>
      <button class="lane-tamper" type="button" hidden>この文章で改ざん実験</button>
    </div>
  </div>
  <button id="compare-button" class="primary" type="button">3 レーンを順に生成</button>
  <p id="compare-status" class="legend"></p>
  <div id="matrix-wrap" hidden>
    <h3>相互検出マトリクス（行 = 文章、列 = 判定に使う鍵）</h3>
    <table id="matrix" class="matrix"></table>
    <p class="legend">セルをクリックすると、その文章をその鍵で着色して上のレーンに表示します。</p>
  </div>
</section>
```

CSS:
```css
.compare-controls, .lane-params { display: flex; gap: 1rem; flex-wrap: wrap; margin: .5rem 0; }
.compare-controls label, .lane-params label { display: flex; align-items: center; gap: .35rem; font-weight: 600; }
.compare-controls input, .lane-params input { width: 9rem; padding: .3rem .4rem; border: 1px solid #aeb9c9; border-radius: .4rem; font: inherit; }
.lanes { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; margin-top: 1rem; }
.lane { padding: .75rem; border: 1px solid var(--line); border-radius: .7rem; background: #fafbfe; }
.lane h3 { margin: 0 0 .4rem; font-size: 1rem; }
.lane .tokens { min-height: 5rem; font-size: .9rem; }
.lane .chart { height: 160px; margin-top: .5rem; }
.lane.active { outline: 2px solid var(--brand); }
.matrix { border-collapse: collapse; width: 100%; margin-top: .5rem; }
.matrix th, .matrix td { border: 1px solid var(--line); padding: .45rem .6rem; text-align: center; }
.matrix td button { width: 100%; background: #fff; border: 0; padding: .3rem; cursor: pointer; }
.matrix td.selected button { background: #dfe8ff; }
.matrix .wm { color: var(--success); font-weight: 650; }
.matrix .nowm { color: var(--danger); }
.muted { color: var(--muted); }
```

- [ ] **Step 2: JS**

`settings` 取得後に既定値をレーン入力へ流し込む（`/api/health` の then 内で `document.querySelectorAll('.lane-gamma').forEach(i => i.value = h.gamma)` 同様に δ）。`compare-seed` は初期化時に `Math.floor(Math.random() * 1e9)`、`lane-key` は `Math.floor(Math.random() * 8_000_000_000) + 1`。

```js
const lanes = Array.from(document.querySelectorAll('#lanes .lane')).map((el, i) => ({
  el, index: i,
  tokensEl: el.querySelector('.lane-tokens'),
  chartEl: el.querySelector('.lane-chart'),
  resultEl: el.querySelector('.lane-result'),
  tamperBtn: el.querySelector('.lane-tamper'),
  text: '', tokens: [], detection: null, cross: {}, watermark: undefined,
}));
const compareButton = document.getElementById('compare-button');
const compareStatus = document.getElementById('compare-status');
const matrixWrap = document.getElementById('matrix-wrap');
const matrix = document.getElementById('matrix');
const LANE_LABELS = ['鍵A（既定）', '鍵B', '透かしなし'];

function laneOverride(lane) {
  const el = lane.el;
  const wm = {};
  const key = el.querySelector('.lane-key');
  const gamma = el.querySelector('.lane-gamma');
  const delta = el.querySelector('.lane-delta');
  if (key && key.value !== '') wm.hash_key = Number(key.value);
  if (gamma && gamma.value !== '') wm.gamma = Number(gamma.value);
  if (delta && delta.value !== '') wm.delta = Number(delta.value);
  if (lane.index === 2) wm.delta = 0;
  return Object.keys(wm).length ? wm : undefined;
}

// 判定用: 「透かしなし」レーンの列は既定鍵（delta を除いた設定）で判定する
function detectOverride(lane) {
  const wm = Object.assign({}, lane.watermark || {});
  delete wm.delta;
  return Object.keys(wm).length ? wm : undefined;
}

async function generateLane(lane, prompt, seed, maxTokens) {
  lane.text = ''; lane.tokens = []; lane.detection = null; lane.cross = {};
  resetTokenView(lane.tokensEl);
  renderZChart(lane.chartEl, [], settings.z_threshold);
  lane.resultEl.hidden = true;
  lane.tamperBtn.hidden = true;
  lane.el.classList.add('active');
  const body = { prompt, seed, max_new_tokens: maxTokens };
  if (lane.watermark) body.watermark = lane.watermark;
  try {
    const response = await fetch('/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    await readSse(response, (event, payload) => {
      if (event === 'token') {
        lane.text += payload.text || '';
        lane.tokens.push(...(payload.tokens || []));
        renderTokenChips(lane.tokensEl, payload.tokens || []);
        renderZChart(lane.chartEl, lane.tokens, settings.z_threshold);
      }
      if (event === 'done') {
        lane.text = payload.full_text;
        lane.tokens = payload.detection.tokens || [];
        lane.detection = payload.detection;
        resetTokenView(lane.tokensEl);
        renderTokenChips(lane.tokensEl, lane.tokens);
        renderZChart(lane.chartEl, lane.tokens, payload.detection.threshold);
        showResult(lane.resultEl, '自己判定: ' + verdictLabel(payload.detection.verdict), payload.detection, payload.detection.verdict === 'watermarked' ? 'success' : '');
        lane.tamperBtn.hidden = typeof sendToTamper !== 'function';
      }
      if (event === 'error') { lane.resultEl.className = 'result error'; lane.resultEl.textContent = payload.message || '生成中にエラーが発生しました。'; lane.resultEl.hidden = false; }
    });
  } catch (error) {
    lane.resultEl.className = 'result error'; lane.resultEl.textContent = error.message; lane.resultEl.hidden = false;
  } finally {
    lane.el.classList.remove('active');
  }
}

async function detectWith(text, watermark) {
  const body = { text, include_tokens: true };
  if (watermark) body.watermark = watermark;
  const response = await fetch('/api/detect', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!response.ok) throw new Error('判定リクエストに失敗しました。');
  return response.json();
}

function matrixCell(det) {
  if (!det) return '<span class="muted">—</span>';
  const cls = det.verdict === 'watermarked' ? 'wm' : (det.verdict === 'not_watermarked' ? 'nowm' : 'muted');
  return '<span class="' + cls + '">z=' + Number(det.z_score).toFixed(2) + '<br>' + escapeHtml(verdictLabel(det.verdict)) + '</span>';
}

function renderMatrix() {
  let html = '<thead><tr><th>文章 ＼ 鍵</th>' + LANE_LABELS.map((l, i) => '<th>' + escapeHtml(i === 2 ? '既定鍵' : l) + '</th>').join('') + '</tr></thead><tbody>';
  lanes.forEach((row) => {
    html += '<tr><th>' + escapeHtml(LANE_LABELS[row.index] + ' の文章') + '</th>';
    lanes.forEach((col) => {
      const det = row.cross[col.index];
      const selected = row.selectedCol === col.index ? ' class="selected"' : '';
      html += '<td' + selected + '><button type="button" data-row="' + row.index + '" data-col="' + col.index + '"' + (det ? '' : ' disabled') + '>' + matrixCell(det) + '</button></td>';
    });
    html += '</tr>';
  });
  matrix.innerHTML = html + '</tbody>';
  matrixWrap.hidden = false;
}

function showLaneUnderKey(rowIndex, colIndex) {
  const row = lanes[rowIndex];
  const det = row.cross[colIndex];
  if (!det) return;
  row.selectedCol = colIndex;
  resetTokenView(row.tokensEl);
  renderTokenChips(row.tokensEl, det.tokens || []);
  renderZChart(row.chartEl, det.tokens || [], det.threshold);
  showResult(row.resultEl, LANE_LABELS[colIndex === 2 ? 0 : colIndex] + ' で判定: ' + verdictLabel(det.verdict), det, det.verdict === 'watermarked' ? 'success' : '');
  renderMatrix();
}

matrix.addEventListener('click', (ev) => {
  const btn = ev.target.closest('button[data-row]');
  if (btn) showLaneUnderKey(Number(btn.dataset.row), Number(btn.dataset.col));
});

async function runComparison() {
  const prompt = document.getElementById('compare-prompt').value.trim();
  if (!prompt) return;
  const seed = Number(document.getElementById('compare-seed').value) || 0;
  const maxTokens = Math.max(1, Number(document.getElementById('compare-max-tokens').value) || 150);
  compareButton.disabled = true;
  matrixWrap.hidden = true;
  lanes.forEach((lane) => { lane.watermark = laneOverride(lane); lane.selectedCol = lane.index; });
  try {
    for (const lane of lanes) {
      compareStatus.textContent = LANE_LABELS[lane.index] + ' を生成中…';
      await generateLane(lane, prompt, seed, maxTokens);
    }
    compareStatus.textContent = '相互検出を計算中…';
    const jobs = [];
    lanes.forEach((row) => lanes.forEach((col) => {
      if (!row.text) return;
      jobs.push(detectWith(row.text, detectOverride(col)).then((det) => { row.cross[col.index] = det; }).catch(() => { row.cross[col.index] = null; }));
    }));
    await Promise.all(jobs);
    renderMatrix();
    compareStatus.textContent = '完了。対角セルが自分の鍵での判定です。';
  } finally {
    compareButton.disabled = false;
  }
}
compareButton.addEventListener('click', runComparison);

lanes.forEach((lane) => lane.tamperBtn.addEventListener('click', () => {
  if (typeof sendToTamper === 'function') sendToTamper({ text: lane.text, watermark: detectOverride(lane), label: LANE_LABELS[lane.index] });
}));
```

注意: `sendToTamper` は Task 5 で `function sendToTamper(payload) {...}` として同じ IIFE 内に定義する（関数宣言は巻き上げられるので Task 4 の `typeof` チェックは Task 5 適用後 true になる）。`showResult` の第 1 引数は既存どおり `<output>` 要素。

- [ ] **Step 3: ブラウザで確認**（サーバーは Task 3 で再起動済み。index.html は再読込で反映）

Playwright MCP で: 鍵比較タブ → 短い prompt、`max_new_tokens` 40 → 実行 → 3 レーンが順に流れる → マトリクスが 3×3 で埋まり、対角（A の文章 × A の鍵、B×B）が `透かしあり`、非対角と「透かしなし」行が `透かしなし`/`判定不能` になる（40 トークンでは T≥25 で判定可能）→ 非対角セルをクリックするとレーンの着色が変わる。スクリーンショットをスクラッチパッドに保存。

- [ ] **Step 4: コミット**

```bash
git add app/static/index.html
git commit -m "Add key comparison tab with three lanes and cross-detection matrix"
```

---

### Task 5: UI「改ざん実験」タブと持ち込みボタン

**Files:**
- Modify: `app/static/index.html`

**Interfaces:**
- Consumes: `POST /api/detect` (`include_tokens`, `watermark`)、Phase 1 描画関数、Task 4 の `sendToTamper` フック（本タスクで定義）
- Produces (JS): `function sendToTamper({text, watermark, label})`、`levenshtein(a, b)`（2000 文字上限）、`scheduleTamperDetect()`

- [ ] **Step 1: マークアップと CSS**

タブ:
```html
<button class="tab" type="button" data-tab="tamper" aria-selected="false" aria-controls="tamper-panel">改ざん実験</button>
```

パネル（鍵比較の後）:
```html
<section id="tamper-panel" class="panel" data-panel="tamper" hidden>
  <h2>文章を書き換えて透かしが消える様子を見る</h2>
  <p class="intro">生成タブ・鍵比較タブの「この文章で改ざん実験」から持ち込むか、下に貼り付けてください。編集するたびに z を再計算します。</p>
  <p id="tamper-source" class="legend">持ち込み元: なし（既定鍵で判定）</p>
  <div class="tamper-stats" id="tamper-stats" hidden>
    <div><dt>元の z</dt><dd id="tamper-z0">-</dd></div>
    <div><dt>現在の z</dt><dd id="tamper-z1">-</dd></div>
    <div><dt>判定</dt><dd id="tamper-verdict">-</dd></div>
    <div><dt>編集文字数</dt><dd id="tamper-dist">-</dd></div>
  </div>
  <label for="tamper-text">文章（編集できます）</label>
  <textarea id="tamper-text" maxlength="10000" placeholder="ここに文章を貼り付けるか、他のタブから持ち込んでください"></textarea>
  <div class="compare-controls">
    <button id="tamper-reset" class="tab" type="button">元に戻す</button>
    <span id="tamper-error" class="muted"></span>
  </div>
  <div id="tamper-tokens" class="tokens"></div>
  <div class="viz">
    <svg id="tamper-chart" class="chart" viewBox="0 0 640 220" preserveAspectRatio="xMidYMid meet" role="img" aria-label="z 値の推移"></svg>
    <div id="tamper-formula" class="formula"></div>
  </div>
</section>
```

生成パネルの `<output id="generation-result">` の直後に:
```html
<button id="generate-tamper" class="tab" type="button" hidden>この文章で改ざん実験</button>
```

CSS:
```css
.tamper-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: .65rem; margin: .5rem 0 1rem; }
.tamper-stats div { padding: .45rem .65rem; border: 1px solid #d7e0ef; border-radius: .4rem; background: #fff; }
.tamper-stats dt { color: var(--muted); font-size: .82rem; }
.tamper-stats dd { margin: 0; font-variant-numeric: tabular-nums; }
```

- [ ] **Step 2: JS**

```js
const tamper = { original: '', watermark: undefined, originalDetection: null, seq: 0, timer: null };
const tamperText = document.getElementById('tamper-text');
const tamperTokens = document.getElementById('tamper-tokens');
const tamperChart = document.getElementById('tamper-chart');
const tamperFormula = document.getElementById('tamper-formula');
const tamperStats = document.getElementById('tamper-stats');
const tamperError = document.getElementById('tamper-error');
const generateTamperButton = document.getElementById('generate-tamper');

function activateTab(name) {
  tabs.forEach((item) => item.setAttribute('aria-selected', String(item.dataset.tab === name)));
  panels.forEach((panel) => { panel.hidden = panel.dataset.panel !== name; });
}
// 既存のタブクリックリスナーも activateTab(tab.dataset.tab) を呼ぶ形に置き換える

function levenshtein(a, b) {
  if (a.length > 2000 || b.length > 2000) return null;
  const prev = new Array(b.length + 1);
  for (let j = 0; j <= b.length; j++) prev[j] = j;
  for (let i = 1; i <= a.length; i++) {
    let diag = prev[0]; prev[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const tmp = prev[j];
      prev[j] = Math.min(prev[j] + 1, prev[j - 1] + 1, diag + (a[i - 1] === b[j - 1] ? 0 : 1));
      diag = tmp;
    }
  }
  return prev[b.length];
}

function renderTamper(det) {
  resetTokenView(tamperTokens);
  renderTokenChips(tamperTokens, det.tokens || []);
  renderZChart(tamperChart, det.tokens || [], det.threshold);
  renderFormula(tamperFormula, det, settings.gamma);
  const z0 = tamper.originalDetection ? Number(tamper.originalDetection.z_score).toFixed(3) : '-';
  document.getElementById('tamper-z0').textContent = z0;
  document.getElementById('tamper-z1').textContent = Number(det.z_score).toFixed(3);
  const v0 = tamper.originalDetection ? verdictLabel(tamper.originalDetection.verdict) : '-';
  document.getElementById('tamper-verdict').textContent = v0 + ' → ' + verdictLabel(det.verdict);
  const dist = levenshtein(tamper.original, tamperText.value);
  const lenDiff = Math.abs(tamperText.value.length - tamper.original.length);
  document.getElementById('tamper-dist').textContent = dist === null ? ('約 ' + lenDiff + '（長文のため文字数差）') : String(dist);
  tamperStats.hidden = false;
}

async function runTamperDetect() {
  const text = tamperText.value.trim();
  if (!text) { tamperStats.hidden = true; resetTokenView(tamperTokens); renderZChart(tamperChart, [], settings.z_threshold); tamperFormula.innerHTML = ''; return; }
  const seq = ++tamper.seq;
  try {
    const det = await detectWith(text, tamper.watermark);
    if (seq !== tamper.seq) return;           // 古い応答は捨てる
    if (!tamper.originalDetection) tamper.originalDetection = det;
    tamperError.textContent = '';
    renderTamper(det);
  } catch (error) {
    if (seq !== tamper.seq) return;
    tamperError.textContent = error.message;   // 前回の表示は残す
  }
}

function scheduleTamperDetect() {
  clearTimeout(tamper.timer);
  tamper.timer = setTimeout(runTamperDetect, 300);
}
tamperText.addEventListener('input', scheduleTamperDetect);

document.getElementById('tamper-reset').addEventListener('click', () => {
  tamperText.value = tamper.original;
  runTamperDetect();
});

function sendToTamper(payload) {
  tamper.original = payload.text || '';
  tamper.watermark = payload.watermark;
  tamper.originalDetection = null;
  tamperText.value = tamper.original;
  const keyLabel = payload.watermark && payload.watermark.hash_key !== undefined ? '鍵 …' + String(payload.watermark.hash_key).slice(-4) : '既定鍵';
  document.getElementById('tamper-source').textContent = '持ち込み元: ' + (payload.label || '貼り付け') + '（' + keyLabel + (payload.watermark && payload.watermark.gamma !== undefined ? '、γ=' + payload.watermark.gamma : '') + ' で判定）';
  activateTab('tamper');
  runTamperDetect();
}

// 生成タブからの持ち込み
let lastGeneration = null;   // 生成ハンドラの done で {text: payload.full_text} を保存し generateTamperButton.hidden = false にする
generateTamperButton.addEventListener('click', () => {
  if (lastGeneration) sendToTamper({ text: lastGeneration.text, watermark: undefined, label: '生成タブ' });
});
```

生成ハンドラの `done` 分岐に `lastGeneration = { text: payload.full_text }; generateTamperButton.hidden = false;` を追加。生成開始時に `generateTamperButton.hidden = true`。

貼り付け直接利用時（持ち込みなしで textarea に入力）は `tamper.original` が空なので、最初の detect 成功時に `if (!tamper.original) { tamper.original = text; }` として元文を確定させる（`runTamperDetect` の成功分岐、`originalDetection` セットの直前）。

- [ ] **Step 3: ブラウザで確認**

Playwright MCP で: 生成タブで短い生成 → 「この文章で改ざん実験」→ 改ざん実験タブが開き、元の z と現在の z が同値、判定が表示される → textarea の数語を書き換える → 300 ms 後に z が下がり編集文字数が増える → 「元に戻す」で元の z に戻る。鍵比較タブの B レーンから持ち込むと「鍵 …XXXX で判定」と表示される。スクリーンショットをスクラッチパッドに保存。

- [ ] **Step 4: コミット**

```bash
git add app/static/index.html
git commit -m "Add tamper experiment tab with live re-detection"
```

---

### Task 6: README 更新と最終確認

**Files:**
- Modify: `README.md`

- [ ] **Step 1: README 追記**

「Use the API」に:
```
- Both `POST /api/generate` and `POST /api/detect` accept an optional `watermark` object `{hash_key, gamma, delta}` (each field optional). Missing fields fall back to the server settings; `z_threshold` cannot be overridden. `delta: 0` generates without a watermark. Invalid values return `400`. This exists for the learning UI: because it lets a caller detect with any key, it turns the service into a key-estimation oracle — never expose it beyond localhost.
```
「Use the browser UI」に:
```
The 鍵比較 tab generates the same prompt and seed three times (server key, a second key, and no watermark), shows each lane's chips and z chart, and fills a 3×3 cross-detection matrix (rows are texts, columns are keys); click a cell to recolor that text under that key. The 改ざん実験 tab lets you edit a generated text and re-detects it after every change, showing the original and current z, verdict, and edit distance.
```

- [ ] **Step 2: 全テスト・実機**

Run: `.venv/bin/python -m pytest tests -q` → 全件 PASS。サーバー稼働確認 `curl -s http://127.0.0.1:8000/api/health`。

- [ ] **Step 3: コミット**

```bash
git add README.md
git commit -m "Document watermark overrides and the comparison and tamper tabs"
```
