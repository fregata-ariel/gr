# AR baseline — 学習パイプライン (design note)

作成日: 2026-09-03
対応: `docs/handoff_questions.md` A3-3(階層的自己回帰 baseline)、
セッション 1 決定(データ生成ローカル / 学習 Colab)

## 役割分担

| 場所 | 実行物 | 依存 |
|---|---|---|
| ローカル | `cfg_reducer.dataset` → `training/prepare_tokens.py` | cfg_reducer(torch 不要) |
| Colab | `training/train_ar.py` | torch のみ(cfg_reducer 不要) |
| ローカル | `training/eval_samples.py` | cfg_reducer(torch 不要) |

torch は pyproject に追加しない(Colab 側で `colab install`)。境界は
ファイル(JSONL / JSON)で受け渡す。

## データ準備(ローカル)

```bash
uv run python -m cfg_reducer.dataset --out data/ds1 \
  --split train=0:2000 --split val=2000:2200
uv run python -m training.prepare_tokens --dataset data/ds1 --out data/tokens1
```

`prepare_tokens.py` は manifest の全 split を読み、**全 split 横断の
`max_offset`** で語彙を 1 つ構築して次を出力する:

```text
<out>/vocab.json    — token -> id(PAD=0)
<out>/meta.json     — max_offset, max_len, split ごとの件数
<out>/<split>.jsonl — {"sample_id", "seed", "tokens": [...]} per line
```

## モデル(Colab)

decoder-only Transformer(`train_ar.py`、単一ファイル・torch 標準のみ):

| 項目 | 既定値 |
|---|---|
| d_model / heads / layers / FFN | 128 / 4 / 4 / 512(CLI で変更可) |
| 位置表現 | 学習埋め込み(max_len は meta.json 起点) |
| 目的関数 | next-token cross entropy(PAD は ignore_index) |
| 最適化 | AdamW + grad clip、epoch ごとに val loss / token accuracy |
| 生成 | temperature + top-k、EOS 停止、`samples.json` に N 本出力 |

語彙 ~10+max_offset 種・系列長 ~数十トークンの小規模問題なので、
まず T4 で十分。

## 評価(ローカル)

```bash
uv run python -m training.eval_samples --samples run1_samples.json \
  --vocab data/tokens1/vocab.json --train-tokens data/tokens1/train.jsonl
```

- **well-formed 率** — `model_input.detokenize` の strict パーサが通る割合
  (文法検査を学習パイプラインの一次メトリクスに使う)
- **unique 率** — 生成 Sketch の重複排除後の割合
- **novelty 率** — 学習 split の Sketch 集合に含まれない割合
  (丸暗記の検出)
- 付随統計: 平均系列長

## Colab 実行手順

```bash
colab new -s trainer --gpu T4
colab install -s trainer torch
colab upload data/tokens1/train.jsonl /content/train.jsonl
colab upload data/tokens1/val.jsonl /content/val.jsonl
colab upload data/tokens1/vocab.json /content/vocab.json
colab upload data/tokens1/meta.json /content/meta.json
colab upload training/train_ar.py /content/train_ar.py
colab exec -s trainer -f /content/train_ar.py -- \
  --train /content/train.jsonl --val /content/val.jsonl \
  --vocab /content/vocab.json --meta /content/meta.json \
  --out /content/run1 --epochs 30 --num-samples 200
colab download -s trainer /content/run1/samples.json ./run1_samples.json
colab download -s trainer /content/run1/model.pt ./run1_model.pt
colab download -s trainer /content/run1/history.json ./run1_history.json
colab stop -s trainer
```

(`colab exec` の引数渡しは CLI の版により異なる可能性がある。渡せない場合は
`train_ar.py` の既定値が上記パスに一致するよう設定してあるため、引数なしで
実行できる。)

## 見送り(将来)

- 文法制約付きデコード(REF 範囲・括弧対応のマスク)— 精度頭打ち時の拡張。
  現状は「制約なしでどこまで文法を学習するか」自体を baseline の観測対象とする
- 離散拡散・階層展開との比較 — baseline の結果取得後(A3-3)
- ノードトークン割当・重み付与(後段フェーズ)
