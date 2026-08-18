# 学習モード Phase 3 設計: logit インスペクタ・トークナイザー可視化・prompt プリセット

## 目的

Phase 1（トークン着色・z グラフ・数式）、Phase 2（鍵比較・改ざん実験）に続き、
アルゴリズムの「内側」と前提条件を見せる 3 機能を追加する。

- **G: logit インスペクタ** — 各生成ステップで候補トークンの元 logit / Green 補正後 logit / 確率を見せ、
  「Green に δ が足された結果どの候補が選ばれたか」を 1 ステップ単位で体感する。
- **D: トークナイザー可視化** — 任意文のトークン境界と ID を見せ、編集前後で分割がどう変わるかを比較する。
- **H: 低エントロピー prompt プリセット** — 定型出力では Green を選ぶ余地がなく z が伸びにくいことを実演する。

学習 API は常に有効。ローカル専用（Phase 2 と同じ前提）。

## 機能要件

### G: logit インスペクタ
1. `GenerateRequest.inspect: bool = False`。`true` のとき各生成ステップの候補記録を行う。
2. 候補は「元 logit 上位 10 件 ∪ 補正後 logit 上位 10 件」（最大 20 件）。各候補は
   `{id:int, text:str, raw:float, adjusted:float, green:bool, prob:float}`。`prob` は補正後 logit を
   temperature（0.7）で割った softmax（語彙全体、top-p / top-k 打ち切り前）。`text` は Phase 1 の
   `token_pieces` と同じ個別 decode（断片は `�` のままでよい）。
3. `done` イベントの payload に `steps: list[StepInspection] | null` を追加。`StepInspection = {index:int, chosen_id:int, candidates:[...]}`。
   `steps[i]` は `detection.tokens[i]` に対応する（特殊トークンを除外して整列）。`inspect=false` なら `steps` は `null`。
4. 記録は先頭 **400 ステップ**まで。それ以降のトークンには `steps` の要素がない（`steps` の長さ ≤ min(400, len(tokens))）。
5. `inspect=true` のときは `delta == 0` でもプロセッサを付けて記録する（加算 0）。補正後 logit は
   既存 `WatermarkLogitsProcessor` と完全に一致すること。
6. UI（生成タブ）: 生成完了後、トークンチップをクリックすると「このステップの候補」パネルを表示。
   候補を補正後 logit の降順に並べ、元 logit のバー、Green なら +δ 分の伸び、補正後、確率 %、選ばれたトークンを強調。
   サンプリング設定（temperature 0.7 / top-p 0.8 / top-k 20 / 表示確率は打ち切り前）を注記。
   400 ステップ以降のチップをクリックしたら「このステップは記録されていません（先頭 400 ステップのみ）」と表示。
   鍵比較タブ・改ざん実験タブでは使わない（`inspect` は生成タブのみ true）。

### D: トークナイザー可視化
7. `POST /api/tokenize {text}` → `{count:int, tokens:[{index:int, id:int, text:str}]}`。空文字 / 10,000 文字超は 400。
   tokenizer 未ロードは 503（`ModelNotReadyError`）。モデル重みは不要（tokenizer のみ）。`text` は `token_pieces` の結果。
8. UI「トークナイザー」タブ: テキスト A / テキスト B（B は任意）。入力ごとに 300 ms デバウンスで `/api/tokenize`。
   交互配色チップ（境界が分かる）＋各チップ下に ID、トークン数表示。B があれば A/B の ID 列を LCS で比較し、
   A のみ / B のみのトークンをそれぞれ着色、共通トークン数・変化トークン数を表示。
   古い応答は捨てる（seq）。エラーは表示し前回結果は残す。

### H: prompt プリセット
9. 生成タブと鍵比較タブの prompt 欄の上に「プリセット」`<select>` を置く。低エントロピー 3 件・高エントロピー 3 件（下記）。
   選択で prompt に反映し、選択肢ごとの一言ヒントを表示。サーバー変更なし。
   - 低: 「1 から 100 まで数字を改行区切りで書いてください。」「Python で FizzBuzz を書いてください。」「五十音（あいうえお…）を順に、間に空白を入れて書いてください。」
   - 高: 「架空の街を舞台にした短い物語を 400 字程度で書いてください。」「日本の四季それぞれの魅力を段落ごとに説明してください。」「旅行の計画の立て方について、初心者向けにアドバイスしてください。」
   - ヒント（低）:「定型的な出力では次のトークンがほぼ決まっていて Green を選ぶ余地がなく、z が伸びにくい」
   - ヒント（高）:「自由度が高い文章では Green を選びやすく、z が伸びやすい」

## 実装方針

### app/inspection.py（新規）
```python
@dataclass
class Candidate: id:int; raw:float; adjusted:float; green:bool; prob:float
@dataclass
class StepRecord: index:int; candidates:list[Candidate]; chosen_id:int|None = None
class InspectingProcessor(LogitsProcessor):
    def __init__(self, inner: WatermarkLogitsProcessor, temperature: float, top_n: int = 10, max_steps: int = 400)
    records: list[StepRecord]
    def __call__(self, input_ids, scores):
        adjusted = self.inner(input_ids, scores)      # 既存の補正をそのまま使う
        if len(self.records) < self.max_steps:
            raw = scores[0]; adj = adjusted[0]
            ids = union(topk(raw, top_n), topk(adj, top_n))
            probs = softmax(adj / temperature)
            green = adj[ids] > raw[ids]  (delta>0) / delta==0 のときは inner.generator.green_list で判定
            records.append(...)
        return adjusted
```
`delta == 0` の Green 判定は `inner.generator.green_list(prev_id)` を使い、`adjusted == raw` でも Green を出す。

### app/services.py
- `_generate(..., inspect: bool)`: `inspect` なら `InspectingProcessor(WatermarkLogitsProcessor(...), temperature=0.7)` を
  `LogitsProcessorList` に入れる（delta==0 でも）。`done` 構築時に `records` と生成 ID 列（特殊含む）を突き合わせ、
  非特殊トークンの分だけ `StepInspection` を作り `steps` に入れる（`chosen_id` は output ids から）。
- `TokenizeService`（または `DetectionService.tokenize(text)`）: `_token_ids` + `token_pieces`。

### app/schemas.py
`GenerateRequest.inspect`, `TokenizeRequest{text}`, `TokenizeResponse{count, tokens:[TokenizedToken{index,id,text}]}`,
`CandidateDetail`, `StepInspection`。

### app/main.py
`/api/tokenize` を追加。`generate` は `payload.inspect` を `begin(..., inspect=...)` に渡す。

### app/static/index.html
- 生成タブ: `inspect: true` で送信、`done.steps` を保持、チップクリック（`#generated-tokens` にイベント委譲、チップに `data-index`）で
  `renderInspector(step)`。`tokenChipMarkup` に `data-index` 属性を追加（Phase 1 関数の拡張、既存表示に影響なし）。
- 「トークナイザー」タブ: 2 つの textarea、`tokenizeWith(text)`、`renderTokenList(container, tokens, diffSet)`、`lcsDiff(idsA, idsB)`。
- プリセット: `<select>` と `PRESETS` 配列、`change` で prompt へ反映＋ヒント表示（生成・鍵比較の両方）。

## エラー処理
- `inspect` の記録が失敗しても生成は継続し `steps` は `null`（ログに出す）。
- `/api/tokenize` の tokenizer 未ロードは 503、入力不正は 400。

## テスト
- `InspectingProcessor`: 補正後 logit が `WatermarkLogitsProcessor` と一致 / 候補に chosen 相当が含まれる / prob が softmax と一致 /
  `max_steps` で打ち切り / `delta==0` でも green が出る。
- 生成: `inspect=True` で `done["steps"]` が `tokens` と整列（`steps[i].chosen_id == tokens[i].id`）、`False` で `None`。
  `FakeModel` は `logits_processor` を呼ばないので、テスト用に処理を呼ぶ `InspectableFakeModel` を追加する（vocab 16 のダミー logits を作って processor を呼び出す）。
- `/api/tokenize`: 正常 / 空 400 / 上限超 400 / 未ロード 503。
- 既存 96 件を壊さない。

## README
- `inspect` と `steps`、`/api/tokenize`、新タブ・プリセットの説明。
