# 対照実験(B): pointer 表現の帰属と評価の厳密化

作成日: 2026-09-04
状態: **n24 完了・n32 実行中**(18 run、Colab T4)
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

### n24 結果(2026-09-04、test 200、seed 0–2)

| 構成 | test NLL/token(平均 ± sd) | paired ΔNLL vs base(3 seed 符号一致) | edge acc | WF(種別) | 制約サンプル KS 平均 |
|---|---|---|---|---|---|
| base | 0.7371 ± 0.0012 | – | 0.761 | 66.7%(raw) | 0.082 |
| **mask**(base + 参照合法マスク) | **0.7293 ± 0.0044** | **−0.0079(一致)** | 0.762 | **93.3%**(ref-constrained) | 0.086 |
| ptr(案 2″) | 0.7364 ± 0.0045 | −0.0008(不一致) | 0.759 | 90.1%(ref-constrained) | 0.100 |

- **ptr vs mask(直接 paired)**: **+0.0071、3 seed とも ptr が悪い**。WF も mask が高い
  (Wilson 区間: mask 0.91–0.96 / ptr 0.87–0.93、seed ごとにほぼ非重複)。
- mask vs base の paired Δ は seed 0 / 2 で CI が 0 を含まず(−0.011)、seed 1 は
  ほぼ 0(−0.0006)。改善サンプル比率 0.55–0.64。
- **edge accuracy は 3 構成で同一**(0.76)。REF NLL-by-k も 3 構成で一致(下表)。
- dev canary(`eval_axes compare` base → mask / ptr、seed 対応): フラグは mask s2 の
  1 件のみ(canary_token_class.nll_eos_mean: 0.122531 -> 0.2505365  NLL +0.13 — 要検証)。KIND / LOOP / EOS の NLL: base 0.77 / 0.78 / 0.17、
  mask 0.76 / 0.77 / 0.21、ptr 0.77 / 0.79 / 0.21。
- 違反プロファイル(診断 400 本 × 3 seed): base は `ref_out_of_range` 288 が支配的。
  mask / ptr は参照系の違反が構成的にゼロで、残りは括弧系(unclosed / unbalanced)。
  ptr の括弧違反(99)は mask(68)より多い。

**判定(n24)**: 案 2″ の改善(参照違反ゼロ、WF 上昇、NLL 同等)は**すべて参照合法
マスクで説明でき、pointer 機構の寄与は帰属できない**。同一マスク下では pointer は
NLL・WF・分布忠実度のいずれでもわずかに劣る(content 項が短距離で雑音を加える:
k1 / k3 の NLL が mask より高い)。

### 頻度交絡の分離(n24、test)

REF の型コスト(−log P(REF)、平均 0.07–0.09)を除いた **offset-only NLL** と、train で
推定した頻度 baseline 2 種の比較(seed 平均):

| k | 件数 | base / mask / ptr(offset-only) | unigram −log p(k) | 条件付き P(k−klast \| 合法候補数) |
|---|---|---|---|---|
| 1 | 3338 | 0.26 / 0.25 / 0.25 | 0.65 | 0.29 |
| 2 | 1752 | 0.63 / 0.65 / 0.66 | 1.33 | 0.97 |
| 3 | 771 | 1.00 / 0.97 / 0.99 | 2.11 | 1.38 |
| 4 | 317 | 1.89 / 1.80 / 1.73 | 3.06 | 2.06 |
| 5 | 119 | 2.25 / 2.26 / 2.36 | 3.92 | 2.64 |
| 6 | 59 | 2.78 / 2.68 / 2.40 | 4.72 | 3.19 |
| 7 | 24 | 3.38 / 3.22 / 3.10 | 5.34 | 3.86 |
| 8 | 21 | 3.46 / 2.99 / 3.26 | 5.89 | 4.31 |

- 条件付き頻度 baseline だけで **NLL-by-k の形の大半が再現**される。k とともに
  増える難度は、主に「長距離参照は稀で、合法候補が多い」という分布構造。
- モデルの上乗せ(条件付き頻度との差)は k=1 で 0.04、k=2–3 で 0.3–0.4、k ≥ 4 で
  0.3–1.0 nats と、むしろ長距離で大きい。「暗黙のカウントで長距離が指数的に難化する」
  という以前の解釈は支持されない。
- 型コストを引かずに比べると k ≥ 4 で条件付き baseline と同等に見える(型コストが
  長距離 REF 位置で大きいため)。記録 → 検証の順で確認し、上表は分離後の値。
- 3 構成の offset-only 曲線は一致 → 参照先の選択そのものに表現差はない。
- k 均等 macro 平均: base 2.92 / mask 2.72 / ptr 2.72、頻度加重 micro: 0.70 / 0.69 / 0.70。
- 合法候補数を固定した層内でも k とともに NLL は上がる(例: n_legal=4 で k1→k4:
  0.35 → 2.57)が、層内の P(rel) も k とともに減るため、これも頻度で説明される範囲。

(n32 は実行中。結果を追記する)
