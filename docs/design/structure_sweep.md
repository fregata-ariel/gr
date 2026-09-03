# ノード数細分 + 構造特徴スイープ(計画)

作成日: 2026-09-03
状態: **段階 1 実行中**(スイープ 8 点を飽和学習中)
前提: `docs/design/scale_experiments.md`(ノード数 12→24→48 で非制約 WF が
89→46.5→8% に急落、`ref_out_of_range` が全スケールで支配的)

## 目的

非制約 well-formed 率の劣化が「ノード数そのもの」によるのか、ノード数と
相関する「グラフ構造特徴」(階層幅、ネスト深さ、ループ数など)によるのかを
分離し、表現改善(REF の相対位置バイアス、カウント補助等)の要否と対象を
特定する。

## 進め方(3 段、2026-09-03 合意)

1. **スイープ + 飽和学習**(本段): ノード数ごとに seed 範囲を分離し、
   各点を early stopping で飽和まで学習する。学習時に val の per-sample
   NLL(トークン単位の trace 含む)を出力しておく。
2. **特徴量探索**: 難度(NLL)・違反を分類できる構造特徴を幅広く試す
   (`training/structure_features.py` + `training/analyze_features.py`)。
3. **評価軸化**: 分離力のある特徴を評価軸として固定し、以後のモデル比較・
   データ設計の指標にする。

## 実験軸

- **ノード数細分**: 12, 16, 20, 24, 28, 32, 40, 48 の 8 点
  (`edge_prob 0.18` 固定、train 2000 / val 200)
- **seed 分離**: ノード数 index i ごとに seed 基点 `i * 100000`
  (train=[base, base+2000)、val=[base+2000, base+2200))。生成器の乱数列を
  ノード数間で共有しない。
- **飽和学習**: 128d/4L 固定、`--epochs 300 --patience 20`(val loss が
  20 epoch 改善しなければ停止、best checkpoint を採用)。n24 スタディの
  結論(容量増は床を下げない、best epoch ~57)に基づき epochs 上限ではなく
  early stopping で各点を飽和させる。
- **診断サンプリング**: 非制約 400 本(±2.5pp 精度)。

## サンプル単位の構造特徴

canonical sample(または MetaGraph)から機械的に計算する:

| 特徴 | 意味 |
|---|---|
| total_motifs / loop 数 / 最大ネスト深さ | 全体規模と階層性 |
| 最大・平均の階層幅(level 内 motif 数) | REF カウント負荷(= REF 窓要求) |
| merge 比率・平均 in-degree(in_offsets 長) | 合流の複雑さ |
| `max_offset_needed`(サンプル単位) | 実際に必要な後方参照距離 |
| トークン系列長 | 系列学習負荷 |

## 分析手法

1. **Teacher-forced per-sample difficulty**
   固定チェックポイントで val 全サンプルの token NLL / accuracy を
   サンプル単位に再スコアリング(CPU セッションで可)。構造特徴との相関を
   (a) ノード数プール全体、(b) 同一ノード数内の層別 — の両方で取り、
   ノード数と特徴の交絡を制御する。
   per-sample スコアは `train_ar.py` が学習終了時に `val_scores.jsonl` として
   出力する(トークン単位 NLL trace 付き)。特徴結合は
   `training/structure_features.py`、分離力の評価は `training/analyze_features.py`。
2. **生成側の違反位置分析**
   非制約生成の違反を「違反発生時点の階層幅・ネスト深さ・レベル内位置・
   REF 超過量」で集計(`eval_samples` の違反シミュレータを拡張)。
   仮説: `ref_out_of_range` は階層幅に対して超線形に増える。
3. **判断基準**
   - 同一ノード数内でも階層幅と NLL/違反が強く相関する
     → カウント表現の問題 → 相対位置バイアス等の表現改善を優先
   - ノード数間の差が系列長でほぼ説明される
     → 学習投資(データ・容量・エポック)の継続で対応

## 成果物

- `runs/sweep_n{12..48}/` + per-sample スコア JSONL
- 特徴 × (NLL, 違反率) の相関表と所見(scale_experiments.md へ追記)
- 表現改善の要否・対象の判断
