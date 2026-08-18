# 学習モード Phase 1 設計: トークン着色・z 推移グラフ・数式パネル

## 目的

本サービス（Qwen3 + Green/Red list 方式の統計的透かし）を使って、開発者・研究者が
「透かし = Green トークンが統計的に多いだけ」という仕組みを体感できるようにする。
Phase 1 はその土台となる **トークン単位の判定データの公開** と **可視化 UI** を扱う。

対象外（Phase 2 以降）: 鍵・γ・δ の任意指定と並列比較、改ざん実験、logit インスペクタ、
トークナイザー単体の可視化、低エントロピー prompt プリセット。

## 機能要件

1. 生成タブ・判定タブとも、結果を **トークン単位のチップ** で表示できる。
   - Green = 緑、Red = 赤、未採点（先頭トークン）= 灰。
   - ホバーで `前トークンID → このトークンID / Green? / T / 累積 z` を表示。
   - 従来のプレーンテキスト表示も残し、切替できる。
2. 生成中は **トークンが到着するたびに** チップの着色と z 推移グラフが更新される（ライブ）。
3. **z 推移グラフ**（インライン SVG、外部ライブラリなし）: 横軸 T（採点トークン数）、縦軸 z。
   閾値 `z_threshold`（既定 4.0）の水平線と、`T < 25` の判定不能領域を描く。
4. **数式パネル**: `z = (g − γT) / √(γ(1−γ)T)` と、実値（g, γ, T, γT, z, p）を代入した式を表示。
5. ライブで表示する累積 z の最終値と、`done` イベント／`/api/detect` の最終 z は **一致** する。

## API 変更

- `DetectionResponse` に省略可能フィールド `tokens: list[TokenDetail] | None = None` を追加。
  `TokenDetail = {index:int, id:int, text:str, green:bool|null, t:int, green_count:int, z:float}`
  - `index` はテキスト内の 0 始まりトークン位置。先頭トークン（index 0）は `green: null, t: 0, z: 0.0`。
  - `t` はその時点までの採点トークン数、`green_count`・`z` は累積値。
- `DetectRequest` に `include_tokens: bool = False` を追加。`true` のときのみ `tokens` を返す。
- SSE `token` イベントは `{"text": str, "tokens": [TokenDetail...]}` になる（`text` は従来どおり）。
  1 イベント内の `tokens` はそのイベントで確定したテキストに対応するトークン列。
- SSE `done` イベントの `detection` は `tokens` を含む（全トークン）。
- `/api/health`・既定レスポンス形状は不変。既存テスト 59 件は互換維持。

## 実装方針

### watermark.py
- `WatermarkDetector.detect_token_ids` の内部ループを共通化し、ステップごとの記録
  `DetectionStep(index, token_id, previous_id, is_green, scored, green_count, z_score)` を返す
  `detect_token_ids_detailed(token_ids) -> (DetectionResult, list[DetectionStep])` を追加する。
  z の式は既存と同一。`ignore_repeated_bigrams` で採点しないステップは `is_green=None`。

### app/services.py
- `token_pieces(tokenizer, token_ids) -> list[str]`: 各 ID を個別に decode し、`�` を含む
  断片は連続グループとしてまとめて decode し、その文字列をグループ末尾のトークンに割り当て、
  先頭側は空文字にする（O(n)）。
- `IdRecordingStreamer(TextIteratorStreamer)`: `put()` で受け取った ID を記録する
  （`skip_prompt` と同様に最初の put はプロンプトとして読み飛ばす）。`on_finalized_text()` を
  上書きし、`{"text": text, "token_ids": pending_ids}` をキューに入れる。`end()` 時も保留分を流す。
- `GenerationService._generate`: 上記ストリーマーからイベントを受け取り、特殊トークンを除外した
  うえで **逐次スコアラー**（前トークンとの bigram で Green 判定・累積 z）を回し、
  `token` イベントに `tokens` を付与する。`done` では従来どおり全 ID を再判定して `tokens` を付ける。
  逐次スコアラーと最終判定は同じ `GreenListGenerator` と式を使うので値は一致する。
- `DetectionService.classify(text, include_tokens=False)`: `include_tokens` のとき
  `detect_token_ids_detailed` と `token_pieces` から `tokens` を組み立てる。
  判定タブのチップは decode した piece を表示する（tokenize→decode は厳密には往復しないため）。

### app/schemas.py / app/main.py
- `TokenDetail` モデル追加、`DetectionResponse.tokens`、`DetectRequest.include_tokens`。
- `/api/detect` は `include_tokens` を `DetectionService.classify` に渡すだけ。

### app/static/index.html
- チップ表示 CSS（`.tok.green/.red/.unscored`）とツールチップ（`title` 属性）。
- 表示切替（チップ / テキスト）。
- SVG 折れ線グラフ描画関数 `renderZChart(svg, steps, threshold)`：閾値線・T<25 の帯・軸ラベル。
- 数式パネル描画関数 `renderFormula(el, detection, gamma)`。γ は `/api/health` から取得。
- 生成タブ: `token` イベントごとにチップ追加・グラフ更新、`done` で最終値に差し替え。
- 判定タブ: `include_tokens: true` で送信し、結果をチップ・グラフ・数式で表示。

## エラー処理
- ストリーマーの ID 記録に失敗しても生成本体は継続し、`tokens` を空配列にする（既存の
  `error` イベント経路は変更しない）。
- `include_tokens` 指定時に短すぎるテキストは従来どおり `inconclusive` を返し、`tokens` は
  得られた分だけ返す。

## テスト
- `detect_token_ids_detailed`: 累積 z の最終値が `detect_token_ids` の z と一致、先頭は未採点、
  ステップ数 = len(ids) - 1。
- `token_pieces`: ASCII・日本語・マルチバイト分割トークンのマージ。
- `IdRecordingStreamer`: プロンプト分の ID を含まない、`end()` で保留分が流れる。
- SSE `token` イベントに `tokens` が含まれ、最後のライブ z == `done.detection.z_score`。
- `/api/detect`: `include_tokens` の有無で `tokens` の有無が変わる。
- 既存 59 件がそのまま通ること。
