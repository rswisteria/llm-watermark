# LLM生成テキスト電子透かしデモ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 論文の Soft Watermark を Qwen3 の CPU 生成へ組み込み、モデル重みなしの検出専用モードを備えた日本語デモを完成させる。

**Architecture:** `watermark.py` に設定、green-list 再構築、`LogitsProcessor`、検出器を集約し、生成と検出が同じ `GreenListGenerator` を使う。`demo.py` は Qwen3 の chat template と `model.generate()` を呼び出し、検出専用モードでは tokenizer だけをロードする。

**Tech Stack:** Python 3.10、CPU 版 PyTorch、`transformers>=4.51.0`、pytest、標準ライブラリ `dataclasses` / `math` / `argparse`。

**Spec:** `docs/superpowers/specs/2026-08-14-llm-watermark-demo-design.md`

## Global Constraints

- `gamma=0.25`、`delta=2.0`、z 閾値 `4.0` を既定値にする。
- シード方式は論文の h=1、直前 1 token の ID に hash key を乗算して `torch.Generator` を seed する方式にする。
- green list は `torch.randperm(vocab_size)` の先頭 `floor(gamma * vocab_size)` 個とする。
- 透かしは `transformers.LogitsProcessor` として `model.generate()` に渡し、独自生成ループを実装しない。
- Qwen3 は `torch.bfloat16`、CPU、`enable_thinking=False` でロードする。fp32 へフォールバックしない。
- 通常デモの既定生成長は 200 token、sampling は temperature `0.7`、top-p `0.8`、top-k `20` とする。
- 検出統計量は `z=(G-gamma*T)/sqrt(T*gamma*(1-gamma))`、p 値は片側標準正規 survival function、判定は strict `z > 4.0` とする。
- 検出専用モードはモデル重みをロードせず、同じ tokenizer と watermark パラメータだけを使う。
- テストはモデルをダウンロードせず、固定 vocabulary と token ID 列で実行できるようにする。
- ソースとドキュメントは少数ファイルに保ち、GUI・学習・攻撃耐性評価を追加しない。

---

### Task 1: プロジェクト依存関係と failing tests の土台

**Files:**
- Create: `requirements.txt`
- Create: `tests/test_watermark.py`

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-08-14-llm-watermark-demo-design.md`
- Produces: pytest cases that define the public `watermark.py` API before implementation.

- [ ] **Step 1: Write the failing tests**

Create tests that import the not-yet-existing public API and specify the core behavior:

```python
import math

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


def test_detector_uses_strict_z_threshold():
    config = WatermarkConfig(gamma=0.5, z_threshold=4.0)
    detector = WatermarkDetector(vocab_size=8, config=config)
    token_ids = [0, 1]

    result = detector.detect_token_ids(token_ids)

    assert result.is_watermarked is (result.z_score > 4.0)


def test_invalid_configuration_and_short_sequence_are_rejected():
    with pytest.raises(ValueError, match="gamma"):
        WatermarkConfig(gamma=0.0)

    with pytest.raises(ValueError, match="at least two"):
        WatermarkDetector(vocab_size=8, config=WatermarkConfig()).detect_token_ids([1])


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
```

- [ ] **Step 2: Run the focused tests and verify the expected RED state**

Run:

```bash
pytest -q tests/test_watermark.py
```

Expected: collection fails because `watermark.py` does not exist yet. If pytest itself cannot import torch, install the CPU dependencies from `requirements.txt` before continuing; do not weaken the tests with unconditional skips.

- [ ] **Step 3: Add the minimal runtime dependency declaration**

Create `requirements.txt`:

```text
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.6.0
transformers>=4.51.0
pytest>=8.0
```

- [ ] **Step 4: Re-run collection to confirm the only remaining failure is the missing implementation**

Run:

```bash
pytest -q tests/test_watermark.py
```

Expected: the test module collects after dependencies are installed, then fails with `ModuleNotFoundError: watermark`.

### Task 2: Watermark configuration and shared green-list generator

**Files:**
- Create: `watermark.py`
- Test: `tests/test_watermark.py`

**Interfaces:**
- Consumes: Task 1 tests.
- Produces: `WatermarkConfig`, `GreenListGenerator`, and the shared deterministic `green_list()` method.

- [ ] **Step 1: Implement the smallest code for configuration and green-list tests**

Add the imports, dataclass, validation, and generator before adding the logits processor:

```python
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class WatermarkConfig:
    gamma: float = 0.25
    delta: float = 2.0
    hash_key: int = 15485863
    z_threshold: float = 4.0
    ignore_repeated_bigrams: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if self.delta < 0.0:
            raise ValueError("delta must be non-negative")
        if self.hash_key <= 0:
            raise ValueError("hash_key must be positive")
        if self.z_threshold < 0.0:
            raise ValueError("z_threshold must be non-negative")


class GreenListGenerator:
    def __init__(self, vocab_size: int, config: WatermarkConfig) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.vocab_size = vocab_size
        self.config = config

    def green_list(self, previous_token_id: int, device: Optional[torch.device] = None) -> torch.Tensor:
        target_device = torch.device(device or "cpu")
        generator = torch.Generator(device=target_device)
        generator.manual_seed(self.config.hash_key * int(previous_token_id))
        permutation = torch.randperm(self.vocab_size, generator=generator, device=target_device)
        green_size = math.floor(self.vocab_size * self.config.gamma)
        return permutation[:green_size]
```

Because Task 1 imports every public symbol at module collection time, also add
minimal importable placeholders for `WatermarkLogitsProcessor`,
`DetectionResult`, and `WatermarkDetector` in this task. Their behavior is
implemented from the already-failing tests in Task 3; the placeholders must
not make the focused green-list test depend on later functionality.

Import `math` at the module top. The generator must create a fresh or reseeded generator per call so the result depends only on the previous token, vocabulary size, device, and watermark parameters.

- [ ] **Step 2: Run only the generator/configuration tests**

Run:

```bash
pytest -q tests/test_watermark.py::test_green_list_is_reproducible_for_same_previous_token
```

Expected: the reproducibility assertion passes and the module imports through the placeholders. Use this run to catch syntax and dataclass validation errors before proceeding; detector and processor behavior remains Task 3 work.

- [ ] **Step 3: Refactor only after green**

Keep the public method name `green_list` and its return type `torch.Tensor` unchanged. Do not add alternate PRF schemes, context widths, or model-specific behavior.

### Task 3: Logits processor and detector/statistics

**Files:**
- Modify: `watermark.py`
- Test: `tests/test_watermark.py`

**Interfaces:**
- Consumes: `WatermarkConfig` and `GreenListGenerator` from Task 2.
- Produces: `WatermarkLogitsProcessor`, `DetectionResult`, and `WatermarkDetector.detect()` / `.detect_token_ids()`.

- [ ] **Step 1: Implement `WatermarkLogitsProcessor` to satisfy its failing test**

Add:

```python
from transformers import LogitsProcessor


class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size: int, config: WatermarkConfig) -> None:
        self.config = config
        self.generator = GreenListGenerator(vocab_size, config)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        result = scores.clone()
        for batch_index in range(input_ids.shape[0]):
            previous_token_id = int(input_ids[batch_index, -1].item())
            green_ids = self.generator.green_list(previous_token_id, device=scores.device)
            result[batch_index, green_ids] += self.config.delta
        return result
```

- [ ] **Step 2: Run the processor test and verify GREEN**

Run:

```bash
pytest -q tests/test_watermark.py::test_logits_processor_adds_delta_only_to_green_tokens
```

Expected: PASS. Confirm the input `scores` remains unchanged because the processor clones it before applying the bias.

- [ ] **Step 3: Implement detection and normal-distribution p value**

Add the result dataclass and detector:

```python
from dataclasses import dataclass
from math import erfc, sqrt
from typing import Sequence


@dataclass(frozen=True)
class DetectionResult:
    token_count: int
    green_count: int
    green_fraction: float
    z_score: float
    p_value: float
    is_watermarked: bool


class WatermarkDetector:
    def __init__(self, vocab_size: int, config: WatermarkConfig, tokenizer=None) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.generator = GreenListGenerator(vocab_size, config)

    def detect_token_ids(self, token_ids: Sequence[int]) -> DetectionResult:
        if len(token_ids) < 2:
            raise ValueError("text must contain at least two tokens")
        green_count = 0
        scored = 0
        seen_bigrams = set()
        for index in range(1, len(token_ids)):
            bigram = (int(token_ids[index - 1]), int(token_ids[index]))
            if self.config.ignore_repeated_bigrams and bigram in seen_bigrams:
                continue
            seen_bigrams.add(bigram)
            scored += 1
            green_ids = self.generator.green_list(bigram[0])
            green_count += int(bigram[1] in green_ids.tolist())
        if scored == 0:
            raise ValueError("text must contain at least one scorable token")
        expected = self.config.gamma * scored
        z_score = (green_count - expected) / sqrt(scored * self.config.gamma * (1 - self.config.gamma))
        p_value = 0.5 * erfc(z_score / sqrt(2.0))
        return DetectionResult(
            token_count=scored,
            green_count=green_count,
            green_fraction=green_count / scored,
            z_score=z_score,
            p_value=p_value,
            is_watermarked=z_score > self.config.z_threshold,
        )
```

Implement `detect(text)` after this method. It must require a tokenizer, call `tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]`, remove a leading `bos_token_id` if present, and delegate to `detect_token_ids`.

- [ ] **Step 4: Run all core tests and fix implementation, not expectations**

Run:

```bash
pytest -q tests/test_watermark.py
```

Expected: all core tests PASS. If the all-green sequence does not exceed z=4, inspect token counting and gamma arithmetic; do not lower the required threshold.

### Task 4: Qwen3 generation and detection-only CLI

**Files:**
- Create: `demo.py`
- Modify: `watermark.py`
- Test: `tests/test_watermark.py` (add CLI-independent helper tests only if needed)

**Interfaces:**
- Consumes: `WatermarkConfig`, `WatermarkLogitsProcessor`, `WatermarkDetector` from Task 3.
- Produces: `python demo.py`, `python demo.py --detect TEXT`, and `python demo.py --detect-file PATH`.

- [ ] **Step 1: Add a failing test for CLI formatting helper**

Add a small pure-Python helper test to `tests/test_watermark.py`:

```python
from demo import format_detection


def test_format_detection_includes_required_metrics():
    result = DetectionResult(10, 8, 0.8, 3.0, 0.001, False)
    output = format_detection(result)
    assert "T=10" in output
    assert "green=8" in output
    assert "z=3.000" in output
    assert "p=" in output
    assert "透かしなし" in output
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest -q tests/test_watermark.py::test_format_detection_includes_required_metrics
```

Expected: FAIL with `ModuleNotFoundError: demo` or missing `format_detection`.

- [ ] **Step 3: Implement CLI helpers and lazy model loading**

Implement these functions in `demo.py`:

```python
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_PROMPT = "日本の四季と自然の魅力について、日本語で読みやすい解説文を書いてください。具体例を交え、200トークン程度で説明してください。"
HUMAN_SAMPLE = """日本の春は、冬の寒さがゆるみ、草木が芽吹き始める季節です。桜や新緑を楽しみに多くの人が公園や山を訪れます。夏には海や高原の風景が鮮やかになり、秋には紅葉と実りの季節を迎えます。冬の雪景色は静かで美しく、地域ごとに異なる自然の表情を見せてくれます。四季の変化は、暮らしや文化にも深く結びついています。"""


def format_detection(result: DetectionResult) -> str:
    label = "透かしあり" if result.is_watermarked else "透かしなし"
    return (
        f"T={result.token_count}, green={result.green_count}, "
        f"z={result.z_score:.3f}, p={result.p_value:.3e}, 判定={label}"
    )
```

`load_tokenizer(model_name)` imports `AutoTokenizer` inside the function. `load_model(model_name)` imports `AutoModelForCausalLM` inside the function, calls:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
)
model.to("cpu")
model.eval()
```

The tokenizer input must be built with:

```python
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=False,
)
```

`generate_text` calls `model.generate()` twice, once without a logits processor and once with `LogitsProcessorList([WatermarkLogitsProcessor(...)])`. Use the same `torch.manual_seed(seed)` immediately before each call, pass `max_new_tokens`, `do_sample=True`, `temperature=0.7`, `top_p=0.8`, `top_k=20`, and decode only tokens after the input length.

For detection-only mode, load tokenizer, build a `WatermarkDetector(vocab_size=len(tokenizer), config=config, tokenizer=tokenizer)`, and print `format_detection(detector.detect(text))`. Do not call `load_model` on either `--detect` or `--detect-file` code paths.

- [ ] **Step 4: Run the helper test and verify GREEN**

Run:

```bash
pytest -q tests/test_watermark.py::test_format_detection_includes_required_metrics
```

Expected: PASS.

- [ ] **Step 5: Add argparse and the three-case default demo**

`main()` must parse all parameters from the spec, construct one `WatermarkConfig`, and dispatch in this order:

1. If `--detect` is supplied, detect that string and exit.
2. If `--detect-file` is supplied, read UTF-8 text, detect it, and exit.
3. Otherwise load model/tokenizer, generate unwatermarked and watermarked output from `DEFAULT_PROMPT`, detect each generated string, detect `HUMAN_SAMPLE`, and print each text plus the metrics.

- [ ] **Step 6: Run syntax and unit verification**

Run:

```bash
python -m py_compile watermark.py demo.py
pytest -q
```

Expected: syntax compilation succeeds and all tests pass. No model download is required for this step.

### Task 5: Documentation and model-free smoke verification

**Files:**
- Create: `README.md`
- Modify: `requirements.txt` if dependency installation instructions need clarification.

**Interfaces:**
- Consumes: final CLI from Task 4.
- Produces: reproducible setup and usage documentation.

- [ ] **Step 1: Document venv and CPU installation**

Include exact commands:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Document that `Qwen/Qwen3-1.7B` is downloaded on first normal demo run, `transformers>=4.51.0` is required for Qwen3, and `--model Qwen/Qwen3-4B` is available when RAM permits.

- [ ] **Step 2: Document model-free detection**

Include:

```bash
python demo.py --detect "これは検出専用モードの日本語サンプルです。十分な長さの文章を入力してください。"
python demo.py --detect-file sample.txt
```

Explain that these commands load the tokenizer but do not load model weights, and that detection must use the tokenizer and hash key from generation.

- [ ] **Step 3: Document statistical interpretation and limitations**

Explain `T`, `green`, `z`, p value, the strict `z > 4` rule, tokenization sensitivity, the first generated token exclusion, low-entropy limitations, and that the output is a statistical signal rather than a semantic claim.

- [ ] **Step 4: Run the model-free smoke command**

Run:

```bash
python demo.py --detect "日本の自然は四季の変化によって豊かな表情を見せます。春の花、夏の海、秋の紅葉、冬の雪景色は、それぞれ地域の文化や暮らしと結びついています。"
```

Expected: the command prints `T`, `green`, `z`, `p`, and `判定`, and it does not download or load model weights.

### Task 6: Full Qwen3 acceptance run

**Files:**
- No new files; inspect `demo.py` output and repository diff.

**Interfaces:**
- Consumes: final implementation, installed dependencies, and the Hugging Face model cache.
- Produces: fresh evidence for the acceptance criteria.

- [ ] **Step 1: Run the default demo with a bounded timeout**

Run:

```bash
timeout 600 python demo.py --max-new-tokens 200
```

Expected: the command completes within 10 minutes on the target CPU environment and prints all three cases.

- [ ] **Step 2: Check the detection results against requirements**

Verify from the fresh output:

- watermarked generation has `z > 4.0` and `p < 3e-5`;
- unwatermarked generation and `HUMAN_SAMPLE` have `z <= 4.0`;
- watermarked text is Japanese and readable on visual inspection;
- each result includes `T`, green count, z, p, and Japanese decision text.

If the watermarked run is below threshold because the model ended early or generated low-entropy text, rerun with the same default parameters and inspect the generated length/prompt before changing behavior. The acceptance threshold must not be lowered. The documented fallback is `--model Qwen/Qwen3-4B`.

- [ ] **Step 3: Verify detector-only behavior against the generated output**

Run the detector-only command with the captured watermarked output:

```bash
python demo.py --detect-file watermarked.txt
```

Expected: the detection result matches the in-process result within tokenizer round-trip limitations, and no model weights are loaded.

- [ ] **Step 4: Run final verification before claiming completion**

Run:

```bash
pytest -q
python -m py_compile watermark.py demo.py
git diff --check
```

Expected: all tests pass, compilation succeeds, and `git diff --check` reports no whitespace errors. If the workspace is not a Git repository, report that `git diff --check` could not run and use `git diff --no-index /dev/null <file>` only for manually added files if needed.
