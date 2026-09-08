# C の設計見直し: 生成器 v2 と構造 OOD マトリクス

作成日: 2026-09-04
状態: **Q7 承認済み(A7)・詳細設計中**(設計: Codex GPT-6 Astra high / 実装: Codex GPT-6 Astra low、進行と判断: Claude)
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

## 8. 決定(A7、2026-09-09)

| # | 決定 | 設計への反映 |
|---|---|---|
| A7-1 | 3 族で開始。**族は後で複数回差し替え・追加する**ので抽象化に注意 | 族を plugin として登録する共通インタフェース(spec → CFG 構成、realized 特徴の共通測定、provenance 名の族別付与)。族の追加が `dataset.py` / `controlled_eval.py` の変更を要しない構造にする |
| A7-2〜4, 6, 7 | 既定どおり承認 | — |
| A7-5 | n24 主体で進めるが、**節目では B 以前のスイープと同様に多くのノード数(12〜48)で実証する** | 実験計画に「節目のノード数スイープ」を入れる(family OOD の確定時など) |

作業分担(2026-09-09): 全体の進行と設計判断は Claude、詳細設計は Codex(GPT-6 Astra、
effort high)、実装は Codex(GPT-6 Astra、effort low)。設計・実装の成果物はこの
リポジトリの文書とコードに残し、判断は本ファイルと `handoff_questions.md` に記録する。

## 9. 実装中の実測(2026-09-09、タスク 7 受け入れ時)

structured 族の n24・200 seed(棄却 0、全件 reducible、realized ノード数 23–25):

| params | mean_offset | max_offset | max_depth | n_loops | max_in_degree | n_merge |
|---|---|---|---|---|---|---|
| 既定(L2 G1) | 1.70(1.36–1.97) | 7.6 | 1–2 | 2 | 2 | 3.8 |
| L3 G2 | 1.42(1.29–1.52) | 5.6 | 1–3 | 3 | 2–3 | 3.0 |
| L3 G2 depth3 branch3 | 1.53 | 6.4 | **3 に固定** | 3 | 2–3 | 2.9 |

- v1(§2: irreducible 90%、max_in_degree 3–8、n_merge 8.3、mean_offset 1.88)とは明確に
  異なる族になっている。`target_depth` は MetaGraph の max_depth を正確に制御できる。
- family OOD(layered → structured)は「reducible 化・合流の減少・参照距離の短縮」の
  複合シフトになる。depth / degree OOD は structured 内で params を振って作る。

## 10. Pilot(2026-09-09、n24、測定モード、seed train 0:2000 / val 2000:2200 / test 2200:2400、version `pilot-c`)

| 族(params) | 採用 / 試行 | reducible | mean_offset(low/mid/high) | max_depth | max_in_degree | n_merge | 占有 bucket |
|---|---|---|---|---|---|---|---|
| layered(v1 相当: width3, p.18, L3, G2) | 2000/2000 | 8.8% | 1.88(448 / 1030 / 522) | 1:410, 2:974, 3:615 | 2:198, 3:1386, 4+:416 | 8.4 | 47/54 |
| structured(L3, G2, depth 乱択 1–3) | 1970/2000(重複 30) | **100%** | 1.43(**1766** / 179 / 25) | 1:633, 2:676, 3:661 | 2:1484, 3:486 | 3.0 | 14/54 |
| spaghetti(同 + rate 0.1) | 2000/2000 | 43% | 1.53(1591 / 328 / 81) | 1:196, 2:677, 3:755, 4:316, 5:56 | 2:1168, 3:766, 4+:65 | 4.4 | 47/54 |

所見:
- mean_offset の境界 (1.68, 2.03) は v1 の四分位(q25 / q75)であり、layered では 22 / 52 / 26% に分かれる(設計文書の注記どおり三分位ではない。A7-2 の数値は固定のまま使う)。
- structured は参照距離が構造的に短く、mean_offset 次元ではほぼ 1 bin。depth / degree は params で直接制御できる(§9)。
- structured の同型重複(1.5%)は同型排除で自然に落ちる。seed 範囲は必要数の 1.05 倍程度を見込む。

### D2 の具体化: セルとデータ(n24、production は pilot と別 seed 範囲・別 version)

| セル | train / val(source) | test(target) | 採用 |
|---|---|---|---|
| ID | layered(v1 相当) | layered | 自然分布 |
| family OOD | 同上(**学習は ID と共有**) | structured / spaghetti(別 test) | 自然分布、source を exclude |
| family OOD′ | structured(L3 G2) | layered / spaghetti | 自然分布、source を exclude |
| depth OOD | structured target_depth=1 | structured target_depth=3(2 は参考) | ranges: realized max_depth を train ≤1 / test =3 で確認 |
| degree OOD | structured merge_degree=2 | structured merge_degree=4 | ranges: realized max_in_degree を train ≤2 / test ∈[3,4]。test 側の採用率が低いので seed 範囲を 4 倍 |
| balanced-k | layered、plan: mean_offset 3 bin を等数(他次元は無制限) | 同じ plan | bucket 採用。low bin が 22% なので seed 範囲は 3 倍 |

- 学習 run(n24、seed 0–2): layered(base / mask / ptr)9、structured(base / mask)6、
  structured depth1(base / mask)6、structured merge2(base / mask)6、balanced-k layered
  (base / mask)6 = **33 run**(1 run ≈ 4 分)。
- 同じ source train から作った bundle は vocab が同一なので、1 つの学習済みモデルを複数の
  test(ID / family OOD の各 target)に**再採点**して使う(Colab の再採点スクリプト)。
- 節目(family OOD の確定時)に n = 12, 16, 24, 32, 48 のスイープ(A7-5)。
