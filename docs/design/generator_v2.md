# C の設計見直し: 生成器 v2 と構造 OOD マトリクス

作成日: 2026-09-04
状態: **実験完了・結果記録済み(§12)**(設計: Codex GPT-6 Astra high / 実装: Codex GPT-6 Astra low、進行と判断: Claude)
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

## 11. Production データと実験の起動(2026-09-09)

`python -m cfg_reducer.dataset_v2`(version = git commit)で n24 の 8 dataset、
`prepare_tokens --window-from train [--test-dataset]` で 11 bundle を生成した(所要 1 分弱)。

| dataset | spec | 採用 / 試行 | 備考 |
|---|---|---|---|
| c_layered | layered v1 相当 | 2000 / 200 / 200 | ID・family OOD の source |
| c_structured | structured L3 G2 | 2066 / 203 / 206(試行 2100 / 210 / 210) | c_layered を exclude |
| c_spaghetti | spaghetti rate .1 | test 210 | c_layered, c_structured を exclude |
| c_structured_depth1 / depth3 | target_depth 1 / 3 + ranges | 2183 / 216 / 217、test 216 | depth OOD |
| c_structured_merge2 / merge4 | merge_degree 2 / 4 + ranges | 2163 / 214 / 208、test **230 / 1000**(採用率 23%) | degree OOD |
| c_layered_balanced | layered + mean_offset 3 bin 等数 | 2001 / 201 / 201(試行 6000 / 600 / 600) | balanced-k |

bundle の窓は source train で決まる(layered 20、structured 14、depth1 13、merge2 13、
balanced 18)。OOD target の窓超過除外は 2〜3 件(str2lay 2、depth 2、merge 3)。

実験: Colab T4 セッション `c`、33 run(layered: base / mask / ptr、他: base / mask、各 seed 0–2、
`--epochs 300 --patience 20`、診断 400 + 制約 400 サンプル)。各 source のモデルは学習後に
OOD target の test を同 VM 上で再採点する(`rescore_c.py`)。集計は `controlled_eval`
(--prefix c_<src>_ / c_<src>2<tgt>_、--by-bucket)。

## 12. 結果(2026-09-15、n24、seed 0–2、test = 各 target の test split、`training/summarize_c.py`)

実行は Colab T4 の無料枠で 3 セッションに分けた(約 1 時間で切断されるため。§11 のランナーは
完了 run をスキップして再開できる)。集計は `runs/c_summary.md` / `runs/c_summary.json`。

### ID セル(各 source の自分の test)

| source | base NLL | mask NLL(paired Δ、3 seed 符号) | ptr | WF base / mask / ptr | edge acc |
|---|---|---|---|---|---|
| layered(v1 相当) | 0.7227 ± 0.0035 | **0.7183 ± 0.0017**(−0.0044、一致) | 0.7240(+0.0013) | 71.2 / **93.3** / 88.5% | 0.761 / 0.755 / 0.754 |
| structured | 0.2844 ± 0.0014 | 0.2838 ± 0.0013(−0.0006、不一致) | – | 86.8 / **95.5**% | 0.915 / 0.914 |
| structured depth1 | 0.2785 ± 0.0026 | 0.2800 ± 0.0012(+0.0014、不一致) | – | 95.0 / **98.7**% | 0.915 / 0.916 |
| structured merge2 | 0.2978 ± 0.0011 | 0.2975 ± 0.0024(−0.0002、不一致) | – | 88.8 / **96.1**% | 0.908 / 0.907 |
| layered balanced-k | 0.7306 ± 0.0010 | 0.7293 ± 0.0027(−0.0011、不一致) | – | 67.7 / **92.4**% | 0.749 / 0.749 |

- structured 系の NLL は layered の 4 割以下(KIND / LOOP / EOS の dev NLL: 0.30 / 0.31 / 0.14
  vs 0.77 / 0.80 / 0.21)。構成的に reducible な CFG は自己回帰的にはるかに予測しやすい。
- mask は全 ID セルで base と同等以上、WF は +8〜+25 pt。canary(dev、base → mask)は
  15 比較中フラグ 1 件(balanced s1 の EOS +0.18、要検証)。ptr は layered の 2 seed で
  EOS フラグ(+0.22 / +0.18、B と同じ)。
- 分布忠実度(制約サンプル vs test、KS 平均): layered で mask 0.07–0.08 < base 0.08–0.10、
  structured / depth1 / balanced で同等、merge2 のみ mask 0.09 > base 0.07(要検証)。

### OOD シフト(source モデルを target test で再採点、NLL/token)

| source → target | base: ID → OOD(Δ) | mask: ID → OOD(Δ) | mask − base の劣化差 |
|---|---|---|---|
| layered → structured | 0.723 → 0.878(**+0.155**) | 0.718 → 0.836(**+0.118**) | −0.037 |
| layered → spaghetti | 0.723 → 0.993(+0.271) | 0.718 → 0.934(+0.216) | −0.054 |
| structured → layered | 0.284 → **4.058**(+3.77) | 0.284 → **3.800**(+3.52) | −0.258 |
| structured → spaghetti | 0.284 → 1.557(+1.27) | 0.284 → 1.477(+1.19) | −0.079 |
| depth1 → depth3 | 0.279 → 1.351(+1.07) | 0.280 → 1.180(+0.90) | −0.172 |
| merge2 → merge4 | 0.298 → 0.636(+0.34) | 0.298 → 0.631(+0.33) | −0.006 |

ptr(layered のみ): → structured 0.832(+0.108)、→ spaghetti 0.937(+0.213)。mask と同等だが
seed 分散が 2〜3 倍(±0.021 / ±0.026)。

### realized 特徴で層別した OOD NLL(mask、base も同傾向)

- layered → structured: max_depth 1 / 2 / 3 = 0.80 / 0.80 / 0.90、max_in_degree 2 / 3 = 0.83 / 0.86。
- layered → spaghetti: irreducible 0.95 vs reducible 0.91、depth 1→4 で 0.84 → 1.00。
- structured → layered: irreducible 3.81 vs reducible 3.52、in_degree 2 / 3 / 4+ = 3.41 / 3.75 / 4.18、
  mean_offset low / mid / high = 3.42 / 3.84 / 4.10。
- structured → spaghetti: **reducible 1.13 vs irreducible 1.70**。
- balanced-k(mean_offset 3 bin 等数で学習)の bin 別 NLL: low / mid / high = 0.636 / 0.721 / 0.831。
  自然分布で学習した layered モデルの同 bin は 0.645 / 0.705 / 0.833 → **等数採用は bin 別の
  難度を変えない**(参照距離の難度はサンプル頻度ではなく構造に内在する。B の token 単位の
  頻度分析と整合)。

### 判定(族 × 因子)

1. **族シフトが最大の因子で、非対称**。layered(irreducible 90%)で学習したモデルは structured へ
   +0.12〜0.16、spaghetti へ +0.22〜0.27 の劣化にとどまるが、structured で学習したモデルは
   layered で **+3.5〜3.8** と崩壊する(未見の irreducible ループと高い合流次数)。実 CFG
   (ほぼ reducible)へ向かう方向のシフトは「layered → structured」であり、v1 データでの学習は
   その方向には比較的頑健。逆は成り立たないので、**学習データは irreducible / 高次数を含む
   族を混ぜる**必要がある。
2. **深さシフト(1 → 3)は +0.9〜1.1** と大きい。未見の入れ子深さは高コストで、depth を
   train に含めることが必須。
3. **合流次数シフト(2 → 3)は +0.33**、族・深さより小さい。
4. **mask は既定として妥当**: ID で base と同等以上、すべての OOD セルで劣化幅が base より
   小さく(−0.006〜−0.26)、WF は 92〜99%、canary はほぼ無反応。B の判定を分布シフト下でも
   支持する。
5. **ptr を再検討する理由は出なかった**: OOD でも mask と同等で、分散と EOS フラグが残る。C′
   (窓なし pointer)まで保留。
6. **balanced-k は不要**: サンプル単位の参照距離バランスは bin 別難度を変えない。

### 次の候補

- A7-5 の節目スイープ: family OOD(layered → structured)を n = 12, 16, 24, 32, 48 で mask / base
  × 3 seed(30 run、無料枠なら 2〜3 セッション)。劣化幅のノード数依存を見る。
- **混合族データ**(layered + structured [+ spaghetti])での学習を既定データ候補とし、各 target への
  劣化が単族学習より小さいかを確認する(実 CFG 転移の準備)。
- pyClangAST 実 CFG の接続(family OOD の「実データ」セル)。

2026-09-15: R2 修正により structured 族の出力が変わる。
以後の dataset は新 version(git commit)で生成する。
§10〜§12 と混合族実験のデータは旧 version(1439828 / 4fe5569)で生成されたもの。
