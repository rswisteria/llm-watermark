# LLM生成テキスト電子透かしデモ 設計書

## 目的

論文「A Watermark for Large Language Models」(arXiv:2301.10226v4) の Soft Watermark（Algorithm 2）を、Qwen3 の Hugging Face `generate()` API に `LogitsProcessor` として組み込み、生成テキストの統計的な検出までを CPU 上で確認できる小さなサンプルを実装する。

このデモで確認することは次の二点である。

1. モデルを再学習せず、生成時のロジット加算だけで透かしを埋め込める。
2. 検出時はモデル重みをロードせず、生成に使った tokenizer と watermark パラメータだけで検出できる。

## スコープ

含めるもの:

- Qwen/Qwen3-1.7B を既定値とする CPU 生成
- `Qwen/Qwen3-4B` などへのモデル名切り替え
- `gamma=0.25`、`delta=2.0`、hash key、z 閾値の CLI 設定
- h=1（直前 1 token）の green-list 方式
- `transformers.LogitsProcessor` による透かし付き生成
- z スコア、片側 p 値、判定結果の表示
- 生成・検出で共有する green-list 再構築ロジック
- 重複 bigram を初出だけ数える検出オプション

含めないもの:

- 学習・ファインチューニング
- パラフレーズ等への攻撃耐性評価
- GUI/Web UI
- ビームサーチ、private watermark API

## 方式

`WatermarkConfig` に以下を保持する。

| パラメータ | 既定値 | 意味 |
| --- | ---: | --- |
| `gamma` | `0.25` | vocabulary に占める green list の割合 |
| `delta` | `2.0` | green token に加える logit bias |
| `hash_key` | `15485863` | 前 token ID から PRNG seed を作る秘密値 |
| `z_threshold` | `4.0` | `z > threshold` で透かしありとする閾値 |
| `ignore_repeated_bigrams` | `False` | 検出時の重複 bigram スキップ |

各位置の green list は、生成・検出の両方で次の同一手順を用いる。

1. 直前 token ID を `previous_token_id` とする。
2. `hash_key * previous_token_id` を `torch.Generator` の seed にする。
3. `torch.randperm(vocab_size)` を生成する。
4. permutation の先頭 `floor(gamma * vocab_size)` 個を green list とする。

生成時は green list の token の logits に `delta` を加え、通常の `model.generate()` の sampling に戻す。検出時は token 列の 2 番目以降について、直前 token から作った green list に含まれるかを数える。

検出対象を生成された出力文字列だけにするため、最初の出力 token はスコア対象から除外する。生成時の最初の token はプロンプト末尾 token を seed にするが、出力文字列単独の検出ではプロンプトを再現できないためである。この扱いは論文公式実装の `simple_1` と同じである。

`T` をスコア対象 token 数、`G` を green token 数とすると、検出統計量は次のとおり。

```text
z = (G - gamma * T) / sqrt(T * gamma * (1 - gamma))
p = 0.5 * erfc(z / sqrt(2))
watermarked = z > z_threshold
```

scipy は追加せず、標準ライブラリ `math.erfc` を使う。

## コンポーネントとインターフェース

### `watermark.py`

- `WatermarkConfig`: パラメータの dataclass。範囲を検証する。
- `GreenListGenerator(vocab_size, config)`: `green_list(previous_token_id, device)` を提供する。
- `WatermarkLogitsProcessor`: `transformers.LogitsProcessor` 実装。バッチ中の各系列の末尾 token から green list を作り、該当 logits に bias を加える。
- `DetectionResult`: `token_count`, `green_count`, `green_fraction`, `z_score`, `p_value`, `is_watermarked` を保持する。
- `WatermarkDetector(vocab_size, config, tokenizer)`: `detect(text)` と `detect_token_ids(token_ids)` を提供する。検出処理はモデルを参照しない。

### `demo.py`

- `--model`: モデル名。既定 `Qwen/Qwen3-1.7B`
- `--max-new-tokens`: 既定 `200`
- `--seed`: 生成用乱数 seed
- `--gamma`, `--delta`, `--hash-key`, `--z-threshold`
- `--ignore-repeated-bigrams`
- `--detect TEXT`: 検出専用モード
- `--detect-file PATH`: UTF-8 テキストファイルの検出専用モード

引数なしでは、日本語の自由度が高いプロンプトから未透かし/透かし付きの二つを生成し、固定の日本語人間文とともに検出結果を表示する。モデルロードは通常モードだけで行い、検出専用モードでは `AutoTokenizer` のみを使う。

生成は Qwen3 の chat template に `enable_thinking=False` を渡し、`torch_dtype=torch.bfloat16`、`device=cpu`、`do_sample=True`、`temperature=0.7`、`top_p=0.8`、`top_k=20` を使う。未透かしと透かし付きの呼び出しの前に同じ generation seed を設定して再現性を確保する。

## エラー処理

- `gamma` は `(0, 1)`、`delta >= 0`、`z_threshold >= 0`、`vocab_size > 0` とする。
- 検出 token 数が 2 未満の場合は、統計量を計算せず分かりやすい `ValueError` を送出する。
- `T=0` になる重複 bigram 設定も同様に拒否する。
- Qwen3 の chat template で thinking を無効化できる transformers バージョンを `4.51.0` 以上にする。
- bfloat16 以外の fp32 ロードへのフォールバックは実装しない。環境が要件を満たさない場合はエラーを明示する。

## テスト計画

モデルをダウンロードしない単体テストで次を検証する。

1. 同じ token ID、vocabulary、設定から green list が再現する。
2. `LogitsProcessor` が green token だけに delta を加える。
3. 共有 green-list ロジックで全 green token を並べた列の z が 4 を超える。
4. z スコア、p 値、strict な判定境界が正しい。
5. gamma 不正値と短い入力を拒否する。
6. 重複 bigram スキップ時に `T` と `G` が unique bigram 数になる。

単体テスト後、次を実行して受け入れ基準を確認する。

- `python demo.py --detect "..."` がモデル重みなしで完了する。
- `python demo.py` が未透かし/透かし付き/人間文の三つを表示する。
- 透かし付き出力が `z > 4`、未透かしと人間文が `z <= 4` になることを確認する。
- 透かし付き日本語出力を目視確認する。

## 依存関係と実行環境

- Python 3.10 系
- CPU 版 PyTorch
- `transformers>=4.51.0`
- Qwen3-1.7B の初回ダウンロードにインターネット接続

モデル重みはリポジトリへ追加せず、Hugging Face のキャッシュを使う。
