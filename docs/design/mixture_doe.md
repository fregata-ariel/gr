# 混合重みの実験計画(混合物計画 + Scheffé 多項式)

作成: 2026-09-16。状態: 提案(判断事項は `docs/handoff_questions.md` Q9)。
前提: `generator_v2.md` §13〜§14(混合族データ = 既定候補、優位はノード数に頑健)、
`compute_backend.md`(実行はローカル GPU、ルーター経由)。

## 1. 目的

混合族データの重み w = (w_lay, w_str, w_spa) を最適化する。現在は等重み(重心)で、layered
test の損失 +0.075〜+0.10 が代償。2:1:1 のような点を場当たりに試すのではなく、和が 1 の比率
に対する古典的な**混合物計画**(simplex 上の設計点 + Scheffé 多項式)を入れ、応答曲面として
最適点と各族の相互作用(相乗・拮抗)を推定する。要因が増えても同じ枠組みで分析できる。

## 2. 因子・応答・固定条件

| 区分 | 内容 |
| --- | --- |
| 主因子 | 組成 x = (x_lay, x_str, x_spa)、Σx = 1(simplex 上) |
| 副因子 | seed(学習 seed 0–2、ブロック)。n は 24 を主、48 は確認(§5) |
| 固定 | 学習データの試行数 2150(受理 ≈ 2144、n24)、val 220、REF 窓 `max_offset`(設計点で共通)、構成 mask、epochs 300 / patience 20、sample-seed 1000+seed |
| 応答 | 固定 test 3 種(`data/s24_{layered,structured,spaghetti}` の test、全設計点で同一行)の NLL/token: y_lay, y_str, y_spa。副応答: WF(reference-constrained)、edge accuracy |
| 合成応答 | y_bal = (y_lay + y_str + y_spa) / 3(均等重視)、y_max = max(y_lay, y_str, y_spa)(最悪ケース)。用途重みを変えた合成も同じ当てはめから計算できる |

固定条件の根拠: REF 窓は現状 `--window-from train` で学習 split から決まり、設計点ごとに
19 / 20(n24)のように変わる。test 行の窓外除外は 0 なので test 自体は同一だが、モデル形状
(距離ヘッド)が変わるため、`prepare_tokens` に `--max-offset` を追加して全設計点で固定する
(n24 は 20、n48 は 37 = 各族 pool の最大値)。データ数は試行数固定で ≈ 一定(重複除去の差 ±5)。

## 3. 設計点(simplex-centroid + 軸点、10 点)

| # | 種別 | (x_lay, x_str, x_spa) | 備考 |
| --- | --- | --- | --- |
| 1–3 | 頂点 | (1,0,0), (0,1,0), (0,0,1) | 純族。混合族は weight > 0 のみなので純族 spec で生成 |
| 4–6 | 辺中点 | (½,½,0), (½,0,½), (0,½,½) | 2 族混合 |
| 7 | 重心 | (⅓,⅓,⅓) | 現行の既定候補(§13) |
| 8–10 | 軸点 | (⅔,⅙,⅙), (⅙,⅔,⅙), (⅙,⅙,⅔) | 内点。二次モデルの lack-of-fit 検出用 |

各点 × 3 seed = 30 run(n24、ローカルで 1 run 約 3 分 → 学習 1.5 h + 再スコア)。
ユーザー案の 2:1:1 = (½,¼,¼) は設計点に含めず、当てはめた曲面からの予測を確認 run で検証する
(§4-4)。含める場合は 11 点(判断 Q9-1)。

生成: 各点の学習 / val データは `dataset_v2` の混合族 spec(weight = x)で生成し、3 つの test
dataset を `--exclude-dataset` する。seed 範囲は全点で共通。混合族の成分選択は seed ごとの
乱択なので、実現組成は x から二項ゆらぎ(N = 2150 で ±1%)だけずれる。**回帰には実現組成を
使う**(各サンプルの provenance の spec と seed から `families.mixture.component_for` で数える)。

不採用案: 族ごとの大きな pool から先頭 round(x_i · N) 件を切り出して合成する方式。共通乱数で
点間の分散は減るが、dataset dir を合成する新機構が要り、実運用の生成経路(混合族)と乖離する。

## 4. 分析(`training/mixture_doe.py`、torch 非依存)

1. **Scheffé 二次モデル**(応答ごと): y = Σ_i β_i x_i + Σ_{i<j} β_ij x_i x_j + γ_seed。
   切片なし(Σx = 1 のため)。β_i は純族の応答、β_ij > 0 は拮抗(混ぜると悪化)、< 0 は相乗。
   30 観測、パラメタ 6 + ブロック 2、残差 df 22。最小二乗(numpy、matplotlib 経由で導入済み)。
   seed 反復から純誤差、点平均から lack-of-fit を F 検定。特殊三次(+ β_123 x_1 x_2 x_3)も当てはめ
   て比較する。
2. **報告**: 係数と SE、R²、lack-of-fit p、各設計点の観測平均 vs 予測、simplex 格子(0.05 刻み、
   231 点)上の予測表と等高線図(matplotlib、`docs/figures/mixture_doe_*.png`)。
3. **重心対比**: 各点の y − y(重心) を seed でペアにした ΔNLL と 95% CI(既存の paired 手法)。
4. **最適化と確認**: 合成応答 y_bal と y_max それぞれで格子上の最小点 x* と予測 CI を出し、
   x* と 2:1:1 で確認 run(各 3 seed)。予測と観測の差が予測 CI 内なら曲面を採用。
5. 実現組成のばらつきが小さいことの確認(設計行列の条件数)も報告に含める。

テスト: 既知の二次曲面 + 乱数から生成した合成データで係数が復元されること、格子最小点が
真の最小点に一致すること、実現組成カウンタが小さな手作り dataset で正しいこと。

## 5. n48 での確認

頂点 3 + 重心 + x*(y_bal)の 5 点 × 3 seed = 15 run(1 run 約 5 分 → 約 1.5 h)。n24 の曲面が
n48 でも順位を保つかを見る(§14 の「優位は n に頑健」の重み版)。

## 6. 実装と実行の分担

| # | 内容 | 担当 |
| --- | --- | --- |
| T1 | `prepare_tokens --max-offset N`(窓の固定)+ テスト | OpenCode |
| T2 | `training/mixture_doe.py`: 設計点 → spec JSON / runner Plan の生成、実現組成、Scheffé 当てはめ、格子最適化、Markdown + 図 | OpenCode(設計は本書) |
| T3 | データ生成(10 dataset + bundle 40)、Plan 実行(ローカル、ルーター)、分析、`generator_v2.md` §15 に記録 | Claude |

Run 名: `d_s24_p<点番号>_mask_n24_s<seed>`、再スコア `d_s24_p<点>2{lay,str,spa}_mask_n24_s<seed>`。
所要: データ生成 15 分、学習 + 再スコア 2 h、確認 run 30 分、n48 1.5 h。
