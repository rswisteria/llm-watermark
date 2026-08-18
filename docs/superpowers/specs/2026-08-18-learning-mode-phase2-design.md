# 学習モード Phase 2 設計: 鍵比較・改ざん実験

## 目的

Phase 1（トークン着色・z 推移グラフ・数式パネル）の上に、次の 2 つの体験を追加する。

- **B: 鍵比較** — 同じ prompt / seed で「鍵A / 鍵B / 透かしなし」を順に生成し、各文章を各鍵で相互検出して
  「別鍵で見ると Green 比率が γ に戻り、検出できない」ことを体感する。
- **E: 改ざん実験** — 生成文を編集すると z が即時に再計算され、「どこまで書き換えると透かしが消えるか」を体感する。

学習 API は常に有効（環境変数によるオプトインなし）。本サービスはローカル専用であり、
任意鍵での検出は鍵推定のオラクルになるため、公開してはならない（README に明記）。

対象外（Phase 3 以降）: logit インスペクタ、トークナイザー単体の可視化、低エントロピー prompt プリセット。

## 機能要件

### API
1. `GenerateRequest` と `DetectRequest` に省略可能フィールド `watermark: WatermarkOverride | None = None` を追加。
   `WatermarkOverride = {hash_key: int | None, gamma: float | None, delta: float | None}`（すべて省略可）。
   - 省略された値はサーバー設定（`ServiceSettings`）の値を使う。`z_threshold` は上書きできない。
   - 検証は既存 `WatermarkConfig.__post_init__` に委ねる（`1 ≤ hash_key ≤ MAX_HASH_KEY`、`0 < gamma < 1`、`delta ≥ 0`）。
     不正値は既存の RequestValidationError ハンドラ経由で `400 {"detail": "invalid request"}`。
   - `delta == 0` のときは生成で `WatermarkLogitsProcessor` を付けない（透かしなし）。検出側は `delta` を使わない。
2. 生成のライブ判定・`done.detection`・`/api/detect` は、同一リクエストでは同じ解決済み config を使う（Phase 1 の一致性を維持）。
3. 鍵をログ・レスポンス・ヘルスに出さない（既存方針）。
4. `include_tokens` は従来どおり `watermark` と併用できる。

### UI「鍵比較」タブ
5. 入力: prompt（2000 文字まで）、seed（既定はランダム、3 レーン共通）、max_new_tokens（既定 150、1〜設定上限）。
6. レーンカード 3 枚（固定）: A（鍵: サーバー既定 = 上書きなし）、B（鍵: ランダム整数をプリセット、編集可）、
   なし（δ = 0 固定）。A/B は γ・δ を編集可（既定はサーバー値）。
7. 実行で A → B → なし の順に `/api/generate`（同じ prompt/seed/max_new_tokens、各レーンの `watermark`）を SSE で
   呼び、3 列にチップ・z グラフ・自己判定をライブ表示する。1 レーン失敗しても残りは続行し、失敗レーンにはエラー表示。
8. 全レーン完了後、相互検出マトリクス（行 = レーンの文章、列 = レーンの鍵設定）を `/api/detect`（`include_tokens: true`、
   列レーンの `watermark`）で埋める（最大 9 回）。セルは z と判定を表示。セルクリックでその文章をその鍵で
   着色したチップとグラフに切り替える（既定は対角セル = 自分の鍵）。
9. 各レーンに「この文章で改ざん実験」ボタンを置き、文章と鍵設定を改ざん実験タブへ持ち込む。生成タブにも同じボタンを置く。

### UI「改ざん実験」タブ
10. 編集可能な textarea（10,000 文字まで）と、持ち込まれた鍵設定の表示（鍵は「既定」または末尾 4 桁のみ表示）。
11. 入力のたびに 300 ms デバウンスで `/api/detect`（`include_tokens: true`、持ち込んだ `watermark`）を呼び、
    チップ・z グラフ・数式パネルを更新する。進行中のリクエストは新しい入力が来たら結果を捨てる（最後の入力が勝つ）。
12. 上部に「元の z → 現在の z」「元の判定 → 現在の判定」「編集文字数（レーベンシュタイン距離。元文または現在文が
    2000 文字を超える場合は文字数差のみ）」を表示。「元に戻す」で持ち込み時の文章に戻す。

## 実装方針

### app/schemas.py
```python
class WatermarkOverride(BaseModel):
    hash_key: int | None = None
    gamma: float | None = None
    delta: float | None = None
```
`GenerateRequest.watermark`, `DetectRequest.watermark`（いずれも `WatermarkOverride | None = None`）。

### app/config.py
`ServiceSettings.watermark_config(override: WatermarkOverride | None = None) -> WatermarkConfig`
に拡張し、None でない値だけ差し替えて `WatermarkConfig(...)` を返す（`__post_init__` の検証で ValueError）。
`main.py` は ValueError を 400 に変換する（`RequestValidationError` ハンドラと同じ本文）。

### app/services.py
- `GenerationService.begin(prompt, max_new_tokens, seed, config: WatermarkConfig | None = None)`;
  `_generate` は `config = config or self.config` を使い、`WatermarkLogitsProcessor` は `config.delta > 0` のときのみ
  `LogitsProcessorList` に入れる（`delta == 0` では `logits_processor` を渡さない）。ライブ scorer と
  `_classify_token_ids` にも同じ `config` を渡す。
- `DetectionService.classify(text, include_tokens=False, config: WatermarkConfig | None = None)`。

### app/main.py
- `/api/generate`, `/api/detect` で `payload.watermark` から `resolved_settings.watermark_config(payload.watermark)` を作り
  サービスに渡す。ValueError → 400。

### app/static/index.html
- タブ「鍵比較」「改ざん実験」を追加。Phase 1 の `renderTokenChips` / `renderZChart` / `renderFormula` / `readSse` を再利用。
- 鍵比較: レーン状態 `lanes = [{label, watermark, text, tokens, detection, cross: {}}]`。順次 `for` ループで SSE を await。
  マトリクスは `Promise.all` で 9 回の detect。
- 改ざん実験: `tamper = {original, watermark, originalDetection}`; `input` イベントで debounce → fetch、
  リクエスト連番で古い応答を無視。レーベンシュタインは JS 実装（2000 文字上限）。

## エラー処理
- 不正な `watermark` は 400。生成中のレーン失敗は SSE `error` イベントとして UI に表示し、次レーンへ進む。
- 改ざん実験の detect 失敗は上部にエラー表示し、前回の表示は残す（連続入力中の一時失敗で画面が消えないように）。

## テスト
- `WatermarkOverride` の部分指定・不正値（`hash_key=0`, `gamma=1.0`, `delta=-1`）→ 400。
- `watermark_config(override)`: 未指定はサーバー値、指定分だけ差し替え。
- 生成: 上書き config が `LogitsProcessorList[0].config` に反映される／`delta=0` で `logits_processor` が渡されない／
  `done.detection` とライブ z が上書き config で一致。
- 判定: 同じテキストを既定鍵と別鍵で判定して `green_count` が変わる（Green 列で作ったテキストが別鍵では非検出）。
- main: `watermark` が両エンドポイントでサービスに渡る。
- 既存 78 件を壊さない。

## README
- `watermark` 上書きの説明、鍵比較・改ざん実験タブの説明、任意鍵検出のリスク（ローカル専用）を追記。
