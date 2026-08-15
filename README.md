# Qwen3 watermark demonstration

This demo generates Japanese text with and without a token-level statistical watermark, then reports detector statistics for both outputs and for a human-written sample. It also supports detector-only operation for existing text.

## Setup

The intended setup is a CPU virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`transformers>=4.51.0` is required for Qwen3 support. The normal demo downloads `Qwen/Qwen3-1.7B` on its first run. The model is loaded on CPU with bfloat16, and Qwen3 thinking mode is disabled. If RAM permits, use the larger `Qwen/Qwen3-4B` model.

## Usage

Run the default generation demo:

```bash
python demo.py
```

Use the larger model and limit generation length:

```bash
python demo.py --model Qwen/Qwen3-4B --max-new-tokens 200
```

Detect text directly, without generating:

```bash
python demo.py --detect "これは検出専用モードの日本語サンプルです。十分な長さの文章を入力してください。"
python demo.py --detect-file sample.txt
```

Detector-only commands load the selected model's tokenizer but do not load model weights. Detection must use the same tokenizer, hash key, and gamma that were used for generation. The CLI also exposes `--delta`, `--hash-key`, `--z-threshold`, and `--ignore-repeated-bigrams`; the last option counts each repeated token bigram only once.

The tokenizer ID range is the watermark vocabulary universe. If a model returns padded logits beyond that range, the demo masks those extra IDs so generation cannot sample tokens the detector does not know; if model logits are smaller than the tokenizer vocabulary, generation is rejected with an error.

Other available options include `--seed` and `--max-new-tokens`. The default watermark settings are gamma 0.25, delta 2.0, hash key 15485863, and a strict z threshold of 4.0.

## Interpreting the detector

The output fields mean:

- `T`: number of scored token transitions (token bigrams). The first token is excluded because scoring needs a previous token.
- `green`: number of scored next tokens that fall in the pseudorandom green list selected from their preceding token.
- `z`: standardized excess of green tokens over the expected count `gamma * T`.
- `p`: one-sided normal-tail p value corresponding to `z`.
- `判定`: `透かしあり` only when `z > 4` by default; equality is not sufficient.

These are statistical signals, not semantic claims about the text or its authorship. Results are sensitive to tokenization, so the matching tokenizer and configuration are essential. Short text produces unstable statistics, and low-entropy text or repetitive/ constrained writing can make the green-token count uninformative. The optional repeated-bigram setting can prevent repeated transitions from dominating a score, but changes the effective `T` and must match the configuration used for analysis.

## Limitations

The demo is CPU-oriented and model downloads require network access and sufficient disk/RAM. It does not provide semantic watermark guarantees or robustness against rewriting, tokenization changes, or adversarial manipulation. Detector results should be treated as evidence to evaluate alongside context, not as proof.
