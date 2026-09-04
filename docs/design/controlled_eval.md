# 対照実験(B): pointer 表現の帰属と評価の厳密化

作成日: 2026-09-04
状態: **実行中**(18 run、Colab T4)
前提: `docs/design/external_review_2026-09-04.md` §3-B、決定 A6-2 / A6-5 / A6-6
ブランチ: `feat/controlled-eval`(`feat/metagraph` から分岐)

## 目的

案 2″(合法 pointer + 文脈距離ロジット)の優位が **pointer 機構によるもの**か、
**参照合法マスク(出力空間の制約)によるもの**かを分離し、単一 seed・同一 val の
反復利用で確定できなかった差を、独立 test と複数 seed で判定する。あわせて
「難度は参照距離 k の関数」という結論の頻度交絡を分離する。

## 実験設計

| 項目 | 内容 |
|---|---|
| データ | `data/ds_b_n{24,32}`: train 2000 / val(dev)200 / **test 200**(seed 範囲は sweep と同じ + test を追加)。語彙窓は **train のみ**から決定(`prepare_tokens --window-from train`、窓超過の除外は 0 件) |
| 構成 | `base`: 語彙 baseline(raw)/ `mask`: baseline + 案 2″ と同一の参照合法マスク(`--ref-legal-mask`)/ `ptr`: 案 2″ |
| 学習 seed | 0, 1, 2(構成間で同一番号を対応付け) |
| サンプリング | 診断 400 本(seed 1000+s で構成間対応)+ 制約デコード 400 本(分布忠実度用) |
| 学習 | 128d/4L、`--epochs 300 --patience 20`(dev で早期停止、best checkpoint) |
| test の扱い | 学習・早期停止・設計判断に使わない。採点のみ(`test_scores.jsonl`) |

## 指標(`training/controlled_eval.py`)

- **paired ΔNLL**: 同一 test サンプル上で構成 − baseline(seed ごと、percentile
  bootstrap 95% CI、改善サンプル比率)。3 seed で符号が一致するか。
- **WF と Wilson 区間**: `base` は raw WF、`mask` / `ptr` は reference-constrained WF
  (`eval_axes.md` の 3 名称)。同条件比較は `mask` vs `ptr`。
- **REF NLL-by-k**: 件数、bootstrap CI、頻度 baseline 2 種(train で推定):
  unigram −log p(k)、条件付き頻度 P(k − klast | 合法候補数)。
  k 均等の macro 平均と頻度加重の micro 平均を併記。
- **edge accuracy**: REF 位置で argmax の参照先が正解か(語彙 baseline は REF_k 上の
  argmax、pointer は候補上の argmax)。type-level accuracy の代わりの共通指標。
- **matched 層別**: 合法候補数 n_legal を固定した層内で k ごとの NLL(一様候補コスト
  log n_legal が層内で一定になるため、距離効果を候補数から分離)。

## 判定

案 2″ の採用を確定する条件(レビュー C-4 の ID セル分):

1. untouched test で `ptr` が `mask` に対して REF NLL または edge accuracy で改善
2. KIND / LOOP / EOS canary(`eval_axes` は dev で継続)を悪化させない
3. 短距離 k を犠牲にしない
4. 3 seed で paired ΔNLL の符号が一致

`ptr` ≈ `mask` なら改善の主因は制約であり、pointer 機構の寄与は帰属できない。

## 記録

(結果は末尾に追記)
