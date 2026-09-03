# 評価軸(evaluation axes)

作成日: 2026-09-04
実装: `training/eval_axes.py`(`report` / `compare`)、1 run = 1 `axes.json`
前提: `docs/design/structure_sweep.md`(難度 = 参照距離 k の関数、という結論)

## 運用ルール: 変化 → 記録 → 検証

canary 軸の値は**参考値**であり、逸脱を即「壊れた」と判断しない。値が変化
したら (1) どの run で何がどう変わったかを記録し、(2) 標的を絞った検証を
行い、(3) その後に判断する。`compare` が出すフラグの語は常に「要検証」。
pass/fail として扱うのは末尾の不変条件だけ。

## 主軸(表現改善で動かしたいもの)

| 軸 | 内容 | 期待 |
|---|---|---|
| REF NLL-by-k | 参照距離 k ごとの REF トークン平均 NLL(teacher-forced, val) | 曲線が平坦化・低下 |
| NLL by mean_offset bin | val を平均参照距離の 3 分位に分け、各 bin の NLL/token | 高 offset bin の低下 |
| primary ρ | `mean_offset` / `max_offset` / `max_width` と NLL の Spearman | 低下(距離依存の解消) |
| 非制約 WF・違反プロファイル | 400 本の文法通過率と違反カテゴリ内訳 | WF 上昇、`ref_out_of_range` 減 |

## canary(動いてはいけないもの)

| 群 | 軸 | 現状(sweep 8 点) | フラグ条件 |
|---|---|---|---|
| A: 平坦 | `n_tokens`, `max_out_degree`, `max_in_degree`, `max/mean_scc_size`, `top_width` の ρ | ≈0〜±0.28 | \|Δρ\| > 0.2 |
| B: 符号 | `n_loops`, `max_depth`, `n_back_edges` の ρ | 負(−0.2〜−0.37) | 符号反転(\|ρ\| ≥ 0.15) |
| C: 基礎文法 | KIND / LOOP_* / EOS トークンの平均 NLL | 0.6〜0.8 / 0.6〜0.8 / ~0.1 | +0.1 以上の上昇 |

想定される「要検証」の例:
- 案 2(pointer ヘッド)で `max_in_degree` が立ち上がる — 複数参照の競合。
- 案 1(位置埋め込み)で B 群が正へ反転 — LOOP_START/END をまたぐ階層内位置の計算ミス。
- 相対位置化で `n_tokens` が効き始める — 長さ依存の注意劣化。
- KIND NLL 上昇 — 表現変更が基礎文法を壊した。

閾値は 200 サンプルの揺らぎ(ρ で ±0.14 程度)を踏まえた緩いもの。

## 不変条件(pass/fail)

- 制約デコードの WF = 100%、生成長・unique・novelty 分布が変わらない
- すべての val サンプルで `n_entry == 1`

## 使い方

```bash
uv run python -m training.eval_axes report --run runs/<run> \
  --dataset data/ds_<name> --tokens data/tokens_<name>      # -> runs/<run>/axes.json
uv run python -m training.eval_axes compare runs/<before>/axes.json runs/<after>/axes.json
```

baseline: `runs/sweep_n{12..48}/axes.json`(2026-09-04、128d/4L 飽和学習)。
