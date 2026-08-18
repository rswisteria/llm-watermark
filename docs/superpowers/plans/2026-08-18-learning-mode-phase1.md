# 学習モード Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成・判定結果をトークン単位で Green/Red 着色し、z 値の推移グラフと数式パネルで透かしの仕組みを可視化する。

**Architecture:** `watermark.py` に逐次スコアラー（`IncrementalScorer`）を追加し、既存の一括検出もそれを使うように統一する。`app/services.py` ではストリーマーを拡張してトークン ID を SSE `token` イベントに載せ、逐次スコアラーで Green 判定・累積 z を付与する。`app/static/index.html` はトークンチップ・インライン SVG グラフ・数式パネルを追加する（外部ライブラリなし）。

**Tech Stack:** Python 3.14 / FastAPI / transformers 5.x (`TextIteratorStreamer`) / pytest / 素の HTML+JS+SVG

**Spec:** `docs/superpowers/specs/2026-08-18-learning-mode-phase1-design.md`

## Global Constraints

- 既存レスポンス形状は不変。`DetectionResponse.tokens` は省略可能で既定 `None`、`DetectRequest.include_tokens` は既定 `False`。
- ライブで表示する累積 z の最終値と `done` / `/api/detect` の最終 z は一致する（同じ `IncrementalScorer` と式を使う）。
- 先頭トークン（index 0）は `green: null, t: 0, z: 0.0`。
- UI は外部ライブラリを使わない（インライン SVG）。
- テストは `.venv/bin/python -m pytest tests -q` で実行し、既存 59 件を壊さない。
- コミットメッセージ末尾に以下を付ける:
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Q6hPFyZT8RVVUWRdZVQaeE
  ```

---

## File Structure

| File | Responsibility |
|---|---|
| `watermark.py` | `DetectionStep` データクラス、`IncrementalScorer`（bigram 単位で Green 判定・累積 z を返す）、`WatermarkDetector.detect_token_ids_detailed` |
| `app/token_pieces.py` (新規) | `TokenPieceBuilder`: トークン ID → 表示用文字列（マルチバイト断片のマージ、ストリーミング対応） |
| `app/streaming.py` (新規) | `IdRecordingStreamer`: `TextIteratorStreamer` を継承し、確定テキストと消費トークン ID をペアで流す |
| `app/schemas.py` | `TokenDetail`、`DetectionResponse.tokens`、`DetectRequest.include_tokens` |
| `app/services.py` | `DetectionService.classify(include_tokens)`、`GenerationService._generate` のライブ判定と `done` の `tokens` |
| `app/main.py` | `/api/detect` で `include_tokens` を渡す |
| `app/static/index.html` | チップ表示・表示切替・z グラフ・数式パネル |
| `tests/test_watermark.py`, `tests/test_api.py`, `tests/test_token_pieces.py`(新規), `tests/test_streaming.py`(新規) | テスト |
| `README.md` | API 変更の記載 |

---

### Task 1: `IncrementalScorer` と `detect_token_ids_detailed`

**Files:**
- Modify: `watermark.py` (`DetectionResult` の後、`WatermarkDetector` の前後)
- Test: `tests/test_watermark.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class DetectionStep:
      index: int            # 0 始まりのトークン位置
      token_id: int
      previous_id: int | None
      is_green: bool | None # 未採点なら None
      scored: int           # ここまでの採点数 T
      green_count: int      # 累積 g
      z_score: float        # 累積 z（scored == 0 なら 0.0）

  class IncrementalScorer:
      def __init__(self, vocab_size: int, config: WatermarkConfig) -> None
      def push(self, token_id: int) -> DetectionStep
      @property
      def scored(self) -> int
      @property
      def green_count(self) -> int
      def result(self) -> DetectionResult   # scored == 0 なら ValueError("text must contain at least one scorable token")

  class WatermarkDetector:
      def detect_token_ids_detailed(self, token_ids) -> tuple[DetectionResult, list[DetectionStep]]
  ```

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_watermark.py` の import に `DetectionStep, IncrementalScorer` を追加し、末尾に追記:

```python
def test_incremental_scorer_matches_batch_detector():
    config = WatermarkConfig(gamma=0.25, hash_key=17)
    generator = GreenListGenerator(vocab_size=64, config=config)
    token_ids = [7]
    for _ in range(30):
        token_ids.append(int(generator.green_list(token_ids[-1])[0]))
    token_ids[10] = 3  # 途中に赤も混ぜる

    detector = WatermarkDetector(vocab_size=64, config=config)
    batch = detector.detect_token_ids(token_ids)
    result, steps = detector.detect_token_ids_detailed(token_ids)

    assert result == batch
    assert len(steps) == len(token_ids)
    assert steps[0] == DetectionStep(
        index=0, token_id=7, previous_id=None, is_green=None,
        scored=0, green_count=0, z_score=0.0,
    )
    assert steps[1].previous_id == 7
    assert steps[1].is_green is True
    assert steps[1].scored == 1
    assert steps[-1].scored == batch.token_count
    assert steps[-1].green_count == batch.green_count
    assert steps[-1].z_score == pytest.approx(batch.z_score)
    assert [step.index for step in steps] == list(range(len(token_ids)))


def test_incremental_scorer_reports_z_after_each_token():
    config = WatermarkConfig(gamma=0.5, hash_key=17)
    scorer = IncrementalScorer(vocab_size=64, config=config)
    generator = GreenListGenerator(vocab_size=64, config=config)

    first = scorer.push(7)
    assert first.is_green is None and first.z_score == 0.0
    green_id = int(generator.green_list(7)[0])
    second = scorer.push(green_id)
    assert second.is_green is True
    assert second.scored == 1 and second.green_count == 1
    assert second.z_score == pytest.approx((1 - 0.5 * 1) / math.sqrt(1 * 0.5 * 0.5))
    with_result = scorer.result()
    assert with_result.token_count == 1 and with_result.green_count == 1


def test_incremental_scorer_result_requires_scored_token():
    scorer = IncrementalScorer(vocab_size=64, config=WatermarkConfig(hash_key=17))
    scorer.push(7)
    with pytest.raises(ValueError, match="scorable"):
        scorer.result()


def test_incremental_scorer_skips_repeated_bigrams_when_configured():
    config = WatermarkConfig(gamma=0.5, hash_key=17, ignore_repeated_bigrams=True)
    scorer = IncrementalScorer(vocab_size=64, config=config)
    scorer.push(1)
    a = scorer.push(2)
    scorer.push(1)
    b = scorer.push(2)
    assert a.is_green is not None
    assert b.is_green is None
    assert b.scored == a.scored + 1  # (2,1) は採点され (1,2) の再出現は採点されない
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/bin/python -m pytest tests/test_watermark.py -q`
Expected: ImportError (`DetectionStep`, `IncrementalScorer` が無い)

- [ ] **Step 3: 実装**

`watermark.py` の `DetectionResult` の直後に追加し、`WatermarkDetector.detect_token_ids` をスコアラー経由に書き換える:

```python
@dataclass(frozen=True)
class DetectionStep:
    index: int
    token_id: int
    previous_id: Optional[int]
    is_green: Optional[bool]
    scored: int
    green_count: int
    z_score: float


class IncrementalScorer:
    """Scores tokens one at a time with the same bigram rule as WatermarkDetector."""

    def __init__(self, vocab_size: int, config: WatermarkConfig) -> None:
        self.config = config
        self.generator = GreenListGenerator(vocab_size, config)
        self._previous_id: Optional[int] = None
        self._index = 0
        self._scored = 0
        self._green_count = 0
        self._seen_bigrams: set = set()

    @property
    def scored(self) -> int:
        return self._scored

    @property
    def green_count(self) -> int:
        return self._green_count

    def _z_score(self) -> float:
        if self._scored == 0:
            return 0.0
        expected = self.config.gamma * self._scored
        return (self._green_count - expected) / sqrt(
            self._scored * self.config.gamma * (1 - self.config.gamma)
        )

    def push(self, token_id: int) -> DetectionStep:
        token_id = int(token_id)
        is_green: Optional[bool] = None
        if self._previous_id is not None:
            bigram = (self._previous_id, token_id)
            if not (self.config.ignore_repeated_bigrams and bigram in self._seen_bigrams):
                self._seen_bigrams.add(bigram)
                green_ids = self.generator.green_list(bigram[0])
                is_green = bool(token_id in green_ids.tolist())
                self._scored += 1
                self._green_count += int(is_green)
        step = DetectionStep(
            index=self._index,
            token_id=token_id,
            previous_id=self._previous_id,
            is_green=is_green,
            scored=self._scored,
            green_count=self._green_count,
            z_score=self._z_score(),
        )
        self._index += 1
        self._previous_id = token_id
        return step

    def result(self) -> DetectionResult:
        if self._scored == 0:
            raise ValueError("text must contain at least one scorable token")
        z_score = self._z_score()
        p_value = 0.5 * erfc(z_score / sqrt(2.0))
        return DetectionResult(
            token_count=self._scored,
            green_count=self._green_count,
            green_fraction=self._green_count / self._scored,
            z_score=z_score,
            p_value=p_value,
            is_watermarked=z_score > self.config.z_threshold,
        )
```

`WatermarkDetector` を以下に置き換え（`__init__` に `vocab_size` を保持）:

```python
class WatermarkDetector:
    def __init__(self, vocab_size: int, config: WatermarkConfig, tokenizer=None) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.generator = GreenListGenerator(vocab_size, config)

    def detect_token_ids_detailed(
        self, token_ids: Sequence[int]
    ) -> tuple[DetectionResult, list[DetectionStep]]:
        if len(token_ids) < 2:
            raise ValueError("text must contain at least two tokens")
        scorer = IncrementalScorer(self.vocab_size, self.config)
        steps = [scorer.push(token_id) for token_id in token_ids]
        return scorer.result(), steps

    def detect_token_ids(self, token_ids: Sequence[int]) -> DetectionResult:
        result, _ = self.detect_token_ids_detailed(token_ids)
        return result

    def detect(self, text: str) -> DetectionResult:
        ...（既存のまま）
```

- [ ] **Step 4: テストが通ることを確認**

Run: `.venv/bin/python -m pytest tests -q`
Expected: 全件 PASS（既存 59 + 新規 4）

- [ ] **Step 5: コミット**

```bash
git add watermark.py tests/test_watermark.py
git commit -m "Add IncrementalScorer and per-token detection steps"
```

---

### Task 2: `TokenPieceBuilder`（トークン ID → 表示文字列）

**Files:**
- Create: `app/token_pieces.py`
- Test: `tests/test_token_pieces.py`

**Interfaces:**
- Produces:
  ```python
  class TokenPieceBuilder:
      def __init__(self, tokenizer) -> None
      def push(self, token_ids: Sequence[int]) -> list[str]   # 入力と同じ長さ。呼び出しをまたいで断片グループを保持する
  def token_pieces(tokenizer, token_ids: Sequence[int]) -> list[str]  # 使い捨ての builder で一括変換
  ```
- 仕様: 各 ID を `tokenizer.decode([id], skip_special_tokens=True)` で個別に decode する。結果に `�` を含む ID は「開いているグループ」に溜め、グループ全体の decode に `�` が含まれなくなった時点で、その文字列をグループ末尾の ID に割り当てる（先頭側は `""`）。グループが開いたまま `�` を含まない ID が来たら、そのグループの decode（`�` を含んだまま）＋当該 ID の piece を当該 ID に割り当ててグループを閉じる。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_token_pieces.py`:

```python
from app.token_pieces import TokenPieceBuilder, token_pieces


class ByteTokenizer:
    """1 ID = 1 byte。日本語 1 文字が 3 トークンに分かれる状況を再現する。"""

    def decode(self, token_ids, skip_special_tokens=True):
        return bytes(int(i) for i in token_ids).decode("utf-8", errors="replace")


def ids(text):
    return list(text.encode("utf-8"))


def test_ascii_tokens_map_one_to_one():
    assert token_pieces(ByteTokenizer(), ids("ab c")) == ["a", "b", " ", "c"]


def test_multibyte_fragments_are_merged_onto_last_token():
    pieces = token_pieces(ByteTokenizer(), ids("あb"))
    assert pieces == ["", "", "あ", "b"]


def test_open_group_survives_across_push_calls():
    builder = TokenPieceBuilder(ByteTokenizer())
    raw = ids("字")
    assert builder.push(raw[:2]) == ["", ""]
    assert builder.push(raw[2:]) == ["字"]


def test_unfinished_group_is_flushed_onto_next_clean_token():
    builder = TokenPieceBuilder(ByteTokenizer())
    raw = ids("字")
    assert builder.push(raw[:1]) == [""]
    flushed = builder.push(ids("x"))
    assert flushed[0].endswith("x")
    assert "�" in flushed[0]
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/bin/python -m pytest tests/test_token_pieces.py -q`
Expected: ModuleNotFoundError (`app.token_pieces`)

- [ ] **Step 3: 実装**

`app/token_pieces.py`:

```python
from __future__ import annotations

from typing import Sequence

REPLACEMENT = "�"


class TokenPieceBuilder:
    """Turns token ids into display strings, merging multi-byte fragments.

    Byte-level BPE tokenizers can split one character into several ids; decoding
    such an id alone yields U+FFFD. Fragments are held until the group decodes
    cleanly, and the merged text is assigned to the last id of the group.
    """

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self._open: list[int] = []

    def _decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def push(self, token_ids: Sequence[int]) -> list[str]:
        pieces: list[str] = []
        for token_id in token_ids:
            token_id = int(token_id)
            piece = self._decode([token_id])
            if REPLACEMENT in piece:
                self._open.append(token_id)
                merged = self._decode(self._open)
                if REPLACEMENT in merged:
                    pieces.append("")
                else:
                    pieces.append(merged)
                    self._open = []
            elif self._open:
                pieces.append(self._decode(self._open) + piece)
                self._open = []
            else:
                pieces.append(piece)
        return pieces


def token_pieces(tokenizer, token_ids: Sequence[int]) -> list[str]:
    return TokenPieceBuilder(tokenizer).push(token_ids)
```

- [ ] **Step 4: テストが通ることを確認**

Run: `.venv/bin/python -m pytest tests/test_token_pieces.py -q`
Expected: 4 passed

- [ ] **Step 5: コミット**

```bash
git add app/token_pieces.py tests/test_token_pieces.py
git commit -m "Add TokenPieceBuilder for per-token display strings"
```

---

### Task 3: `IdRecordingStreamer`

**Files:**
- Create: `app/streaming.py`
- Test: `tests/test_streaming.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class StreamChunk:
      text: str
      token_ids: list[int]

  class IdRecordingStreamer(TextIteratorStreamer):
      # イテレートすると StreamChunk を返す。put() で受け取った ID を記録し、
      # on_finalized_text のたびに「前回の確定以降に受け取った ID」を text と組にして流す。
      # skip_prompt=True のときは最初の put()（プロンプト）は記録しない。
  ```

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_streaming.py`:

```python
import torch

from app.streaming import IdRecordingStreamer, StreamChunk


class SpaceTokenizer:
    """decode は各 ID を "t<id> " にする（常に空白で終わるので毎回確定する）。"""

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(f"t{int(i)} " for i in token_ids)


class HoldingTokenizer:
    """空白を出さないので、end() まで確定しない。"""

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(f"t{int(i)}" for i in token_ids)


def drain(streamer):
    return list(streamer)


def test_prompt_ids_are_skipped_and_chunks_pair_text_with_ids():
    streamer = IdRecordingStreamer(SpaceTokenizer(), skip_prompt=True, skip_special_tokens=True)
    streamer.put(torch.tensor([[1, 2]]))          # prompt
    streamer.put(torch.tensor([3]))               # 1D like real generate
    streamer.put(torch.tensor([[4, 5]]))          # 2D like the fake model
    streamer.end()

    chunks = drain(streamer)
    assert chunks == [
        StreamChunk(text="t3 ", token_ids=[3]),
        StreamChunk(text="t4 t5 ", token_ids=[4, 5]),
    ]


def test_end_flushes_pending_ids_with_held_text():
    streamer = IdRecordingStreamer(HoldingTokenizer(), skip_prompt=True, skip_special_tokens=True)
    streamer.put(torch.tensor([[1]]))
    streamer.put(torch.tensor([7]))
    streamer.put(torch.tensor([8]))
    streamer.end()

    chunks = drain(streamer)
    # 空白が無いので put のたびに text="" で ID だけ流れ、end() で全文が確定する
    assert [c.token_ids for c in chunks] == [[7], [8], []]
    assert "".join(c.text for c in chunks) == "t7t8"
    assert chunks[-1].text == "t7t8"


def test_without_skip_prompt_first_put_is_recorded():
    streamer = IdRecordingStreamer(SpaceTokenizer(), skip_prompt=False)
    streamer.put(torch.tensor([[1, 2]]))
    streamer.end()
    assert drain(streamer)[0].token_ids == [1, 2]
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/bin/python -m pytest tests/test_streaming.py -q`
Expected: ModuleNotFoundError (`app.streaming`)

- [ ] **Step 3: 実装**

`app/streaming.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field

from transformers import TextIteratorStreamer


@dataclass(frozen=True)
class StreamChunk:
    text: str
    token_ids: list[int] = field(default_factory=list)


class IdRecordingStreamer(TextIteratorStreamer):
    """TextIteratorStreamer that yields StreamChunk(text, token_ids).

    ``token_ids`` are the ids received via ``put`` since the previous chunk was
    emitted, so downstream code can score tokens while text is still being held
    back by the word-boundary heuristic. The prompt is not recorded when
    ``skip_prompt`` is true.
    """

    def __init__(self, tokenizer, skip_prompt: bool = False, timeout=None, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, timeout=timeout, **decode_kwargs)
        self._pending_ids: list[int] = []

    def put(self, value):
        is_prompt = self.skip_prompt and self.next_tokens_are_prompt
        if not is_prompt:
            ids = value[0] if len(value.shape) > 1 else value
            self._pending_ids.extend(int(token_id) for token_id in ids.tolist())
        super().put(value)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        chunk = StreamChunk(text=text, token_ids=self._pending_ids)
        self._pending_ids = []
        self.text_queue.put(chunk, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)
```

- [ ] **Step 4: テストが通ることを確認**

Run: `.venv/bin/python -m pytest tests/test_streaming.py -q`
Expected: 3 passed

- [ ] **Step 5: コミット**

```bash
git add app/streaming.py tests/test_streaming.py
git commit -m "Add IdRecordingStreamer that pairs streamed text with token ids"
```

---

### Task 4: スキーマと `/api/detect` の `include_tokens`

**Files:**
- Modify: `app/schemas.py`
- Modify: `app/services.py` (`_detection_response`, `_classify_token_ids`, `DetectionService.classify`)
- Modify: `app/main.py` (`detect` エンドポイント)
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `WatermarkDetector.detect_token_ids_detailed`, `DetectionStep`（Task 1）、`token_pieces`（Task 2）
- Produces:
  ```python
  class TokenDetail(BaseModel):
      index: int; id: int; text: str; green: bool | None; t: int; green_count: int; z: float
  class DetectionResponse(BaseModel):  # 既存 + tokens: list[TokenDetail] | None = None
  class DetectRequest(BaseModel):      # 既存 + include_tokens: bool = False
  # services.py
  def _steps_to_tokens(steps: list[DetectionStep], pieces: list[str]) -> list[TokenDetail]
  def _classify_token_ids(token_ids, vocab_size, config, tokenizer=None, include_tokens=False) -> DetectionResponse
  DetectionService.classify(self, text: str, include_tokens: bool = False) -> DetectionResponse
  ```

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_api.py` に追記（`FakeTokenizer` の直後あたり）:

```python
class PieceTokenizer(FakeTokenizer):
    """decode が ID ごとに 'w<id>' を返す（判定 API のトークン表示テスト用）。"""

    def __init__(self, ids):
        self.ids = ids

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        return {"input_ids": torch.tensor([self.ids])}

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(f"w{int(i)}" for i in token_ids)


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


def test_detect_endpoint_forwards_include_tokens():
    client, _, fake_detection = make_api_client()
    client.post("/api/detect", json={"text": "十分に長いテキストです"})
    client.post("/api/detect", json={"text": "十分に長いテキストです", "include_tokens": True})
    assert fake_detection.calls[-2:] == [
        ("十分に長いテキストです", False),
        ("十分に長いテキストです", True),
    ]
```

`FakeApiDetectionService.classify` を `def classify(self, text, include_tokens=False):` にし、`self.calls.append((text, include_tokens))` に変える。既存テストで `fake_detection.calls` を文字列比較している箇所があれば `[c[0] for c in ...]` に直す（`grep -n "detection.calls" tests/test_api.py` で確認）。

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/bin/python -m pytest tests/test_api.py -q -k "tokens or include_tokens"`
Expected: FAIL（`include_tokens` 未対応 / `tokens` 属性なし）

- [ ] **Step 3: 実装**

`app/schemas.py`:

```python
class DetectRequest(BaseModel):
    text: str
    include_tokens: bool = False
    ...（validator は既存のまま）


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
```

`app/services.py`:

```python
from app.schemas import DetectionResponse, TokenDetail
from app.token_pieces import token_pieces
from watermark import (
    DetectionStep,
    IncrementalScorer,
    WatermarkConfig,
    WatermarkDetector,
    WatermarkLogitsProcessor,
)


def _steps_to_tokens(steps: list[DetectionStep], pieces: list[str]) -> list[TokenDetail]:
    return [
        TokenDetail(
            index=step.index,
            id=step.token_id,
            text=piece,
            green=step.is_green,
            t=step.scored,
            green_count=step.green_count,
            z=step.z_score,
        )
        for step, piece in zip(steps, pieces)
    ]


def _inconclusive_response(config, token_count=0, tokens=None):
    return DetectionResponse(
        verdict="inconclusive", num_tokens=token_count, green_count=0,
        z_score=0.0, p_value=1.0, threshold=config.z_threshold, tokens=tokens,
    )


def _detection_response(result, config, tokens=None):
    ...（既存の verdict 判定のまま、DetectionResponse(..., tokens=tokens) を返す）


def _classify_token_ids(
    token_ids: list[int],
    vocab_size: int,
    config: WatermarkConfig,
    tokenizer=None,
    include_tokens: bool = False,
) -> DetectionResponse:
    detector = WatermarkDetector(vocab_size, config)
    tokens = None
    if include_tokens and tokenizer is not None:
        scorer = IncrementalScorer(vocab_size, config)
        steps = [scorer.push(token_id) for token_id in token_ids]
        tokens = _steps_to_tokens(steps, token_pieces(tokenizer, token_ids))
    try:
        result = detector.detect_token_ids(token_ids)
    except ValueError as exc:
        if _is_short_detection_error(exc):
            return _inconclusive_response(config, max(0, len(token_ids) - 1), tokens)
        raise
    return _detection_response(result, config, tokens)


class DetectionService:
    ...
    def _token_ids(self, text: str) -> list[int]:
        input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        token_ids = [int(i) for i in input_ids.tolist()]
        if token_ids and token_ids[0] == getattr(self.tokenizer, "bos_token_id", None):
            token_ids = token_ids[1:]
        return token_ids

    def classify(self, text: str, include_tokens: bool = False) -> DetectionResponse:
        if not self.is_ready():
            raise ModelNotReadyError("tokenizer is not ready")
        return _classify_token_ids(
            self._token_ids(text), len(self.tokenizer), self.config,
            tokenizer=self.tokenizer, include_tokens=include_tokens,
        )
```

（`WatermarkDetector.detect(text)` は CLI 互換のため残すが、サービスは `_token_ids` + `_classify_token_ids` に統一する。`_inconclusive_response` を短文で返すときの `num_tokens` は従来の `classify` では 0 だったが、`max(0, len(token_ids)-1)` になる。既存テストが 0 を期待していないことを `grep -n "num_tokens" tests/test_api.py` で確認する。）

`app/main.py`:

```python
    @app.post("/api/detect", response_model=DetectionResponse, response_model_exclude_none=True)
    async def detect(payload: DetectRequest):
        return detection.classify(payload.text, include_tokens=payload.include_tokens)
```

`response_model_exclude_none=True` により `tokens` 未要求時はレスポンスにキー自体が現れない。

- [ ] **Step 4: テストが通ることを確認**

Run: `.venv/bin/python -m pytest tests -q`
Expected: 全件 PASS

- [ ] **Step 5: コミット**

```bash
git add app/schemas.py app/services.py app/main.py tests/test_api.py
git commit -m "Return per-token detection details from /api/detect on request"
```

---

### Task 5: 生成ストリームにライブ判定を載せる

**Files:**
- Modify: `app/services.py` (`GenerationService._generate`)
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `IdRecordingStreamer`/`StreamChunk`（Task 3）、`IncrementalScorer`（Task 1）、`TokenPieceBuilder`（Task 2）、`_steps_to_tokens`（Task 4）
- Produces: SSE `token` イベント payload `{"text": str, "tokens": [TokenDetail dict...]}`、`done` の `detection.tokens` が全トークン

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_api.py` に追記:

```python
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
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/bin/python -m pytest tests/test_api.py -q -k "live_token or special_tokens_in_live"`
Expected: FAIL（`KeyError: 'tokens'`）

- [ ] **Step 3: 実装**

`app/services.py` の import に `from app.streaming import IdRecordingStreamer` と `from app.token_pieces import TokenPieceBuilder, token_pieces` を追加し、`_generate` を以下に置き換える:

```python
    def _generate(
        self, prompt: str, max_new_tokens: int, seed: int | None
    ) -> Iterator[GenerationEvent]:
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if apply_chat_template is not None:
            inputs = apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
        streamer = IdRecordingStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        result_holder: dict = {}

        def worker() -> None:
            ...（既存のまま。streamer 変数名も同じ）

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        special_ids = set(getattr(self.tokenizer, "all_special_ids", []))
        vocab_size = len(self.tokenizer)
        scorer = IncrementalScorer(vocab_size, self.config)
        pieces = TokenPieceBuilder(self.tokenizer)
        for chunk in streamer:
            live_ids = [i for i in chunk.token_ids if i not in special_ids]
            tokens: list[dict] = []
            if live_ids:
                try:
                    steps = [scorer.push(i) for i in live_ids]
                    tokens = [t.model_dump() for t in _steps_to_tokens(steps, pieces.push(live_ids))]
                except Exception:
                    tokens = []
            if chunk.text or tokens:
                yield GenerationEvent("token", {"text": chunk.text, "tokens": tokens})
        thread.join()

        if "error" in result_holder:
            yield GenerationEvent("error", {"message": "generation failed"})
            return

        try:
            output = result_holder.get("output")
            output_ids = output[0].tolist() if hasattr(output, "__getitem__") else list(output)
            prompt_ids = inputs["input_ids"]
            prompt_length = int(prompt_ids.shape[-1]) if hasattr(prompt_ids, "shape") else len(prompt_ids[0])
            token_ids = [int(token_id) for token_id in output_ids[prompt_length:]]
            token_ids = [token_id for token_id in token_ids if token_id not in special_ids]
            full_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            response = _classify_token_ids(
                token_ids, vocab_size, self.config,
                tokenizer=self.tokenizer, include_tokens=True,
            )
        except Exception:
            yield GenerationEvent("error", {"message": "generation failed"})
            return
        yield GenerationEvent(
            "done",
            {"full_text": full_text, "detection": response.model_dump()},
        )
```

注意: 既存テスト `test_generation_service_emits_tokens_then_done` は `token` イベント 2 件を期待する。`FakeTokenizer.decode` は常に空白で終わるので各 `put` で text が確定し、`end()` の chunk は text も ID も空になるため件数は変わらない。

- [ ] **Step 4: テストが通ることを確認**

Run: `.venv/bin/python -m pytest tests -q`
Expected: 全件 PASS

- [ ] **Step 5: 実機で確認**

```bash
kill $(lsof -tiTCP:8000 -sTCP:LISTEN); sleep 2
set -a; source .env; set +a
nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 > /private/tmp/claude-501/-Users-toyota-PycharmProjects-llm-watermark/ddf726b3-80ee-434e-a9ed-86c7557b06ed/scratchpad/uvicorn.log 2>&1 &
# model_loaded:true まで待ってから
curl -sN http://127.0.0.1:8000/api/generate -H 'content-type: application/json' \
  -d '{"prompt":"春について3文で。","max_new_tokens":40}' | head -c 1500
```
Expected: `token` イベントの data に `"tokens":[{"index":0,...}]` が含まれ、日本語トークンの `text` に `�` が残らない。

- [ ] **Step 6: コミット**

```bash
git add app/services.py tests/test_api.py
git commit -m "Stream per-token watermark scoring during generation"
```

---

### Task 6: UI — トークンチップ表示と表示切替

**Files:**
- Modify: `app/static/index.html`

**Interfaces:**
- Consumes: SSE `token.tokens[]`、`done.detection.tokens[]`、`/api/detect` の `tokens[]`（`include_tokens: true`）
- Produces (JS): `renderTokenChips(container, tokens)`（追記モード）、`resetTokenView(container)`、`tokenChipMarkup(tok)`

- [ ] **Step 1: マークアップと CSS を追加**

`<style>` に追記:

```css
.view-toggle { display: flex; gap: .4rem; margin: 1.25rem 0 .5rem; }
.view-toggle button { padding: .35rem .7rem; border: 1px solid var(--line); background: #e9eef7; font-weight: 600; }
.view-toggle button[aria-pressed="true"] { border-color: var(--brand); background: var(--brand); color: #fff; }
.tokens { min-height: 7rem; padding: 1rem; border-radius: .55rem; background: #fff; border: 1px solid var(--line); line-height: 2; white-space: pre-wrap; word-break: break-all; }
.tok { display: inline; padding: .1rem .15rem; margin: 0 1px; border-radius: .25rem; border-bottom: 2px solid transparent; }
.tok.green { background: #d3f5df; border-bottom-color: #087443; }
.tok.red { background: #ffd9d4; border-bottom-color: #b42318; }
.tok.unscored { background: #e6e9ef; border-bottom-color: #8b95a7; }
.legend { margin: .4rem 0 0; color: var(--muted); font-size: .85rem; }
.legend .tok { padding: 0 .4rem; }
```

生成パネルの `<pre id="generated-text">` の直前に:

```html
<div class="view-toggle" role="group" aria-label="表示形式">
  <button type="button" data-view="chips" data-target="generate" aria-pressed="true">トークン表示</button>
  <button type="button" data-view="text" data-target="generate" aria-pressed="false">テキスト表示</button>
</div>
<div id="generated-tokens" class="tokens" aria-live="polite"></div>
<p class="legend"><span class="tok green">Green</span> <span class="tok red">Red</span> <span class="tok unscored">未採点（先頭）</span> — ホバーで前トークンID・T・累積 z を表示</p>
```

`<pre id="generated-text">` に `hidden` を付ける。判定パネルの `<button id="detect-button">` の直後に同様のトグル（`data-target="detect"`）と `<div id="detect-tokens" class="tokens" hidden></div>`、`<pre id="detect-text-view" hidden></pre>` を追加する。

- [ ] **Step 2: JS を追加**

`showResult` の後に:

```js
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function tokenChipMarkup(tok) {
  const cls = tok.green === null ? 'unscored' : (tok.green ? 'green' : 'red');
  const label = tok.green === null ? '未採点（先頭トークン）' : (tok.green ? 'Green' : 'Red');
  const title = (tok.green === null ? '' : ('前トークン ' + tok.prev_id_label + ' → ')) +
    'ID ' + tok.id + ' / ' + label + ' / T=' + tok.t + ' / g=' + tok.green_count + ' / z=' + Number(tok.z).toFixed(3);
  const text = tok.text === '' ? '·' : tok.text;
  return '<span class="tok ' + cls + '" title="' + escapeHtml(title) + '">' + escapeHtml(text) + '</span>';
}

function renderTokenChips(container, tokens) {
  // 前トークン ID はサーバから来ないので直前チップの id を引き継ぐ
  let prevId = container.dataset.lastId ? Number(container.dataset.lastId) : null;
  container.insertAdjacentHTML('beforeend', tokens.map((tok) => {
    const withPrev = Object.assign({ prev_id_label: prevId === null ? '-' : String(prevId) }, tok);
    prevId = tok.id;
    return tokenChipMarkup(withPrev);
  }).join(''));
  if (tokens.length) container.dataset.lastId = String(prevId);
}

function resetTokenView(container) {
  container.innerHTML = '';
  delete container.dataset.lastId;
}

document.querySelectorAll('.view-toggle button').forEach((button) => button.addEventListener('click', () => {
  const target = button.dataset.target;
  const view = button.dataset.view;
  document.querySelectorAll('.view-toggle button[data-target="' + target + '"]').forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
  const chips = document.getElementById(target === 'generate' ? 'generated-tokens' : 'detect-tokens');
  const text = document.getElementById(target === 'generate' ? 'generated-text' : 'detect-text-view');
  chips.hidden = view !== 'chips';
  text.hidden = view !== 'text';
}));
```

生成ハンドラを更新:

```js
const generatedTokens = document.getElementById('generated-tokens');
...
generatedText.textContent = '';
resetTokenView(generatedTokens);
...
if (event === 'token') {
  generatedText.textContent += payload.text || '';
  renderTokenChips(generatedTokens, payload.tokens || []);
}
if (event === 'done') {
  generatedText.textContent = payload.full_text;
  resetTokenView(generatedTokens);
  renderTokenChips(generatedTokens, payload.detection.tokens || []);
  showResult(...);  // 既存
}
```

判定ハンドラを更新:

```js
const detectTokens = document.getElementById('detect-tokens');
const detectTextView = document.getElementById('detect-text-view');
...
body: JSON.stringify({ text, include_tokens: true })
...
const detection = await response.json();
resetTokenView(detectTokens);
renderTokenChips(detectTokens, detection.tokens || []);
detectTextView.textContent = (detection.tokens || []).map((t) => t.text).join('');
detectTokens.hidden = false;
```

- [ ] **Step 3: ブラウザで確認**

`uvicorn` を再起動（Task 5 Step 5 と同じ手順）し、http://127.0.0.1:8000/ を開いて:
- 生成中にチップが 1 トークンずつ緑/赤で増える
- ホバーで `ID / Green / T / g / z` が出る
- 「テキスト表示」に切り替えると従来表示になる
- 判定タブに生成文を貼って判定するとチップ表示される
- 日本語で `�` が表示されない

- [ ] **Step 4: コミット**

```bash
git add app/static/index.html
git commit -m "Show per-token Green/Red chips in the web UI"
```

---

### Task 7: UI — z 推移グラフと数式パネル

**Files:**
- Modify: `app/static/index.html`

**Interfaces:**
- Consumes: `tokens[]`（`t`, `z`）、`/api/health` の `gamma`, `z_threshold`
- Produces (JS): `renderZChart(svg, tokens, threshold)`、`renderFormula(el, detection, gamma)`

- [ ] **Step 1: マークアップと CSS**

`<style>` に:

```css
.viz { margin-top: 1.25rem; display: grid; gap: 1rem; }
.chart { width: 100%; height: 220px; border: 1px solid var(--line); border-radius: .55rem; background: #fff; }
.chart text { font-size: 11px; fill: var(--muted); }
.chart .axis { stroke: #aeb9c9; stroke-width: 1; }
.chart .z-line { fill: none; stroke: var(--brand); stroke-width: 2; }
.chart .threshold { stroke: var(--danger); stroke-dasharray: 4 3; }
.chart .inconclusive { fill: #f2f4f8; }
.formula { padding: .85rem 1rem; border: 1px solid var(--line); border-radius: .55rem; background: #fff; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .95rem; }
.formula .muted { color: var(--muted); }
```

生成パネルの `<output id="generation-result">` の直前、判定パネルの `<output id="detection-result">` の直前にそれぞれ:

```html
<div class="viz">
  <svg id="generate-chart" class="chart" viewBox="0 0 640 220" preserveAspectRatio="none" role="img" aria-label="z 値の推移"></svg>
  <div id="generate-formula" class="formula"></div>
</div>
```

（判定側は `id="detect-chart"`, `id="detect-formula"`）

- [ ] **Step 2: JS**

```js
let settings = { gamma: 0.25, z_threshold: 4.0 };
fetch('/api/health').then((r) => r.json()).then((h) => { settings = { gamma: h.gamma, z_threshold: h.z_threshold }; }).catch(() => {});

const MIN_T = 25;

function renderZChart(svg, tokens, threshold) {
  const W = 640, H = 220, L = 42, R = 12, T = 12, B = 28;
  const pts = tokens.filter((tok) => tok.green !== null).map((tok) => ({ t: tok.t, z: tok.z }));
  const maxT = Math.max(MIN_T + 5, pts.length ? pts[pts.length - 1].t : 0);
  const zs = pts.map((p) => p.z).concat([threshold, 0]);
  const minZ = Math.min(-1, ...zs), maxZ = Math.max(threshold + 1, ...zs);
  const x = (t) => L + (t / maxT) * (W - L - R);
  const y = (z) => T + (1 - (z - minZ) / (maxZ - minZ)) * (H - T - B);
  const path = pts.map((p, i) => (i ? 'L' : 'M') + x(p.t).toFixed(1) + ',' + y(p.z).toFixed(1)).join(' ');
  svg.innerHTML =
    '<rect class="inconclusive" x="' + L + '" y="' + T + '" width="' + (x(MIN_T) - L).toFixed(1) + '" height="' + (H - T - B) + '"/>' +
    '<text x="' + (L + 4) + '" y="' + (T + 12) + '">T&lt;' + MIN_T + ' 判定不能</text>' +
    '<line class="axis" x1="' + L + '" y1="' + y(0).toFixed(1) + '" x2="' + (W - R) + '" y2="' + y(0).toFixed(1) + '"/>' +
    '<line class="axis" x1="' + L + '" y1="' + T + '" x2="' + L + '" y2="' + (H - B) + '"/>' +
    '<line class="threshold" x1="' + L + '" y1="' + y(threshold).toFixed(1) + '" x2="' + (W - R) + '" y2="' + y(threshold).toFixed(1) + '"/>' +
    '<text x="' + (W - R - 70) + '" y="' + (y(threshold) - 4).toFixed(1) + '">z = ' + threshold + '</text>' +
    '<text x="4" y="' + (y(0) + 4).toFixed(1) + '">0</text>' +
    '<text x="4" y="' + (T + 10) + '">' + maxZ.toFixed(1) + '</text>' +
    '<text x="' + (W - R - 60) + '" y="' + (H - 8) + '">T = ' + maxT + '</text>' +
    (path ? '<path class="z-line" d="' + path + '"/>' : '');
}

function renderFormula(el, detection, gamma) {
  const T = detection.num_tokens, g = detection.green_count;
  const expected = gamma * T;
  const denom = Math.sqrt(gamma * (1 - gamma) * T);
  el.innerHTML =
    '<div>z = (g − γ·T) / √(γ(1−γ)·T)</div>' +
    '<div class="muted">g = Green トークン数, T = 採点トークン数（先頭を除く）, γ = Green 比率</div>' +
    '<div>z = (' + g + ' − ' + gamma + '×' + T + ') / √(' + gamma + '×' + (1 - gamma).toFixed(2) + '×' + T + ')' +
    ' = (' + g + ' − ' + expected.toFixed(2) + ') / ' + (denom || 0).toFixed(3) + ' = <strong>' + Number(detection.z_score).toFixed(3) + '</strong></div>' +
    '<div>p = ½·erfc(z/√2) = ' + Number(detection.p_value).toExponential(3) + '　　判定: z &gt; ' + detection.threshold + ' かつ T ≥ ' + MIN_T + ' で「透かしあり」</div>';
}
```

生成ハンドラ: `token` イベントごとに `liveTokens.push(...payload.tokens)` して `renderZChart(generateChart, liveTokens, settings.z_threshold)`（毎イベント再描画で十分）。`done` で `renderZChart(generateChart, payload.detection.tokens, ...)` と `renderFormula(generateFormula, payload.detection, settings.gamma)`。生成開始時に `liveTokens = []`, `renderZChart(generateChart, [], settings.z_threshold)`, `generateFormula.innerHTML = ''`。
判定ハンドラ: 結果受信後に `renderZChart(detectChart, detection.tokens || [], settings.z_threshold)` と `renderFormula(detectFormula, detection, settings.gamma)`。

- [ ] **Step 3: ブラウザで確認**

- 生成中に折れ線が右へ伸び、閾値の破線・T<25 の帯が見える
- 数式パネルの z が結果パネルの z と一致
- 判定タブでも同じ表示になる

- [ ] **Step 4: コミット**

```bash
git add app/static/index.html
git commit -m "Add z-score trajectory chart and formula panel to the web UI"
```

---

### Task 8: README 更新と最終確認

**Files:**
- Modify: `README.md`（「Use the API」「Use the browser UI」）

- [ ] **Step 1: README を更新**

「Use the API」に追記:

```
- `POST /api/detect` also accepts optional `include_tokens` (default `false`). When true, the response includes `tokens`, a per-token list of `{index, id, text, green, t, green_count, z}` where `green` is `null` for the unscored first token and `z` is the cumulative z-score after that token.
- Generation `token` events now contain `text` and `tokens` (same shape as above, for the tokens consumed since the previous event); the `done` event's `detection.tokens` lists every generated token.
```

「Use the browser UI」に追記:

```
Both tabs can show the result as Green/Red token chips (hover for token id, T and cumulative z), a z-score trajectory chart with the threshold line and the `T < 25` inconclusive band, and the z formula filled with the actual values.
```

- [ ] **Step 2: 全テストと実機確認**

Run: `.venv/bin/python -m pytest tests -q`
Expected: 全件 PASS

サーバを再起動し、生成・判定の両タブで Task 6/7 の確認項目を再チェック。

- [ ] **Step 3: コミット**

```bash
git add README.md
git commit -m "Document per-token detection details and learning UI"
```
