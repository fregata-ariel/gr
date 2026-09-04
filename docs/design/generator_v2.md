# C の設計見直し: 生成器 v2 と構造 OOD マトリクス

作成日: 2026-09-04
状態: **設計レビュー中**(Q7 の判断待ち。実装未着手)
前提: B の判定(`controlled_eval.md`)— 既定構成は baseline + 参照合法マスク
(`--ref-legal-mask`)、pointer は不採用。決定 A6-6 により窓なし pointer は C へ。
ブランチ: `feat/generator-v2`(`feat/controlled-eval` から分岐)

## 1. B を受けた C の目的の変更

外部レビュー時点の C は「案 2″ の優位が生成器に依存しないか」を検証するものだった。
B で pointer が不採用になったため、C の目的を次に改める。

1. **既定モデル(mask)の構造分布シフト耐性の測定**: 学習分布と異なる CFG 族・
   深さ・合流次数で何が壊れるかを、同一 manifest 上で比較する。実 CFG(pyClangAST)
   接続前に「どの方向のシフトに弱いか」を知る。
2. **実 CFG に近い分布を作れる生成器**: v1 は単一族で、realized 分布が実 CFG から
   遠い(§2)。v2 は構造因子を独立に制御し、実測値でバケット採用する。
3. **評価軸の OOD 拡張**: B の評価基盤(`controlled_eval.py`)に bucket 別の NLL / WF を
   加え、以後の全モデル比較(pointer 再検討を含む)を同じ枠で行う。

窓なし pointer(A6-6)は「offset OOD」セルの前提であり、REF 語彙モデルでは構成上
実行できない(train 窓外の k は tokenize できない)。**offset OOD セルと窓なし
pointer は C から外し、後続(C′)に送る**ことを提案する(Q7-6)。

## 2. v1 の realized 分布(ds_b_n24 train 2000、`features_for` + 多重入口判定)

| 特徴 | 分布 | 含意 |
|---|---|---|
| **reducibility** | **1793 / 2000 が irreducible**(多重入口 SCC を含む)。多重入口 SCC は計 2857 個 | 実コンパイラの CFG はほぼ reducible。最大のギャップ |
| n_loops | 1 / 2 / 3 = 79 / 610 / 1311(`int(0.15 n)` = 3 本の後方エッジで固定) | ループ数が n に固定、制御不能 |
| max_depth | 1 / 2 / 3 = 412 / 978 / 610 | ランダム後方エッジの入れ子で決まる |
| max_in_degree | 2 / 3 / 4 / 5+ = 204 / 1412 / 325 / 59 | 層幅 1–3 に固定されているため 3 に集中 |
| mean_offset | 1.88(四分位 1.68 / 1.84 / 2.03、最大 3.3) | 参照距離の分散が小さい |
| max_offset | 6.3(四分位 4 / 6 / 8、最大 18) | 長距離参照は稀(頻度交絡の源) |
| max_width | 13.8(7–23) | — |

v1 = layered DAG(層幅 1–3)+ 後方エッジ 0.15n + 前方ジャンプ 0.1n の単一族。
「structured / reducible」「層幅」「合流次数」「span 分布」を独立に振れない。

## 3. GeneratorSpec(v2)

```python
@dataclass(frozen=True)
class GeneratorSpec:
    family: Literal["layered", "structured", "spaghetti"]
    num_nodes: int
    max_layer_width: int = 3        # layered: 層幅の上限 / structured: 分岐幅の上限
    branch_degree: int = 2          # if/switch の分岐数の上限
    merge_degree: int = 3           # 合流ノードの入次数の上限(超える辺は張らない)
    loop_count: int = 2             # ループ構文の数(structured)/ 後方エッジ数(layered)
    goto_count: int = 1             # 前方ジャンプ(break / return / goto)の数
    span_mode: Literal["short", "uniform", "long"] = "uniform"   # 前方ジャンプ・合流の距離分布
    target_depth: int | None = None # ループ入れ子の目標深さ(structured)
    spaghetti_rate: float = 0.0     # structured の上に無作為辺を足す割合(irreducible 化)
```

- **layered**: v1 と同じ構成法をパラメータ化(層幅上限、後方エッジ数、前方ジャンプ数、
  span)。v1 は `layered(max_layer_width=3, loop_count=int(0.15n), goto_count=int(0.1n),
  span_mode="uniform")` として再現できる(回帰テストで v1 と同一出力を確認)。
- **structured**: 制御構文テンプレートの再帰展開 — `seq`, `if`, `if/else`, `switch(k)`,
  `while`, `do-while`、`break` / `continue` / `return` を対応する出口・ヘッダ・EXIT への
  辺として生成。**構成的に reducible**。`target_depth` で入れ子深さ、`branch_degree` /
  `max_layer_width` で幅、`loop_count` でループ数、`goto_count` で break / return 数を制御。
  ノード予算 `num_nodes` に合わせて展開を打ち切る(realized ノード数は ±10% 程度の
  ばらつきを許容し manifest に記録)。
- **spaghetti**: structured の上に `spaghetti_rate` の割合で無作為な前方 / 後方辺を追加。
  reducibility は生成後に判定して realized に記録(irreducible 化の程度を連続的に制御)。

## 4. realized feature と bucket 採用

指定値と縮約後の MetaGraph 特徴は一対一でない(レビュー指摘)。生成後に測定して
bucket を埋める。

- **測定**: `structure_features.features_for(mg)`(既存 23 特徴)+ CFG 側の
  `reducible: bool`(多重入口 SCC の再帰判定、§2 で使ったもの)+ realized `num_nodes`。
- **bucket 次元**(既定): `mean_offset` low / mid / high(v1 の三分位 1.68 / 2.03 を境界に
  固定)、`max_depth` 0–1 / 2 / 3+、`max_in_degree` ≤2 / 3 / 4+、`reducible` yes / no。
- **採用**: `BucketPlan(dims, target_per_bucket)`。seed を昇順に試行し、対応 bucket が
  未充足なら採用、充足済みなら棄却(棄却 seed と bucket を manifest に記録)。試行数・
  採用率・bucket ごとの充足数を manifest に出す。既存の重複排除はそのまま。
- **provenance**: `generator.name = "generate_cfg_v2"`、`config = spec`(dataclass →
  dict)。requested(spec)と realized(特徴)を manifest の sample エントリに併記。
  `sample_id` の導出は変えない(name / version / seed / config)。

## 5. 実験マトリクス(改訂)

| セル | train | test | 確認するもの |
|---|---|---|---|
| ID | v1 相当(layered)| 同族別 seed | B の再現(既に済み。参照点) |
| family OOD | layered(v1) | structured(reducible) | **実 CFG 方向へのシフト**。最優先 |
| family OOD′ | structured | layered / spaghetti | 逆方向(構造化データで学習したモデルの頑健性) |
| depth OOD | structured depth ≤ 1 | depth 2–3 | 階層汎化 |
| degree OOD | merge_degree ≤ 2 | 3–4 | 複数参照の競合 |
| balanced-k | bucket 採用で mean_offset を均等化 | 同分布 | 頻度交絡を除いた距離効果(B の検証の生成側からの確認) |
| offset OOD | — | — | **C′ へ送る**(窓なし pointer が前提) |

- 構成: `base`(raw)と `mask`(既定)。`ptr` は family OOD のみ参考(Q7-3)。
- seed 3 本、ノード数は **n24 を主**(1 run ≈ 4 分)、family OOD のみ n32 で確認。
- 概算: 5 セル × 2 構成 × 3 seed = 30 run ≈ 2–3 時間(T4)。
- 評価: `controlled_eval`(test、paired ΔNLL、edge acc、頻度 baseline)+ `eval_axes`
  (dev canary)+ `sketch_stats`(制約サンプルの分布忠実度)。OOD セルでは
  **bucket 別 NLL / WF** を追加する(`controlled_eval --by-bucket`)。
- 判定の枠組み: OOD 劣化 = test NLL の ID からの上昇量と、WF(reference-constrained)
  の低下量。「どの因子で壊れるか」を族 × 因子の表にまとめる。合否基準は設けず、
  変化 → 記録 → 検証で運用する。

## 6. 実装計画(承認後)

1. `cfg_reducer/generate_v2.py`: `GeneratorSpec`、`generate_cfg_v2(engine, *, seed, **spec)`、
   族ごとの構成関数。v1 再現テスト、reducible 保証テスト(structured)、ノード予算テスト。
2. `cfg_reducer/reducibility.py`(または `dataset.py` 内): 多重入口 SCC 判定。
3. `cfg_reducer/dataset.py`: `BucketPlan` と `build_dataset(..., accept=...)`、manifest 拡張
   (attempts / accepted / per_bucket / requested / realized)。CLI に `--family` 等。
4. `training/controlled_eval.py`: bucket 別集計、OOD セル用の train / test dataset 対応
   (train と test で dataset dir が異なる → `prepare_tokens` を「train dataset で窓を決め、
   別 dataset の test を tokenize」できるよう拡張)。
5. 実験実行(Colab)と記録。

## 7. 判断が必要な点(Q7)

| # | 論点 | 推奨 |
|---|---|---|
| Q7-1 | 族の定義: structured(reducible 保証)/ layered(v1)/ spaghetti(structured + 無作為辺)の 3 族でよいか。実 CFG(pyClangAST)の統計に合わせる情報があれば族の設計に反映したい | 3 族で開始 |
| Q7-2 | bucket 次元と境界: mean_offset(v1 三分位)/ max_depth / max_in_degree / reducible の 4 次元、bucket あたり同数採用 | 既定どおり |
| Q7-3 | `ptr` を C に含めるか | family OOD のみ参考として含める(1 セル × 3 seed) |
| Q7-4 | v2 は新モジュール(`generate_v2.py`、名前 `generate_cfg_v2`)とし v1 は残す | 残す(既存 provenance の互換) |
| Q7-5 | ノード数: n24 主、family OOD のみ n32 | 既定どおり |
| Q7-6 | offset OOD と窓なし pointer を C′ に送る | 送る |
| Q7-7 | structured 族のノード予算: realized `num_nodes` の ±10% を許容し記録 | 許容 |
