# Synthetic dataset generation (design note)

作成日: 2026-09-03
対応: `docs/handoff_questions.md` A2 手順 4(seed ベースの synthetic CFG バッチ生成)

## モジュール構成

| モジュール | 責務 |
|---|---|
| `cfg_reducer/generate.py` | 純粋な synthetic CFG 生成(`generate_cfg`)。`main.py` から移設し、matplotlib 非依存。同一 `(num_nodes, edge_prob, seed)` から常に同一グラフ |
| `cfg_reducer/dataset.py` | バッチ生成 `build_dataset()` と CLI。split 分離、構造重複排除、manifest 出力 |

`main.py` は `generate_cfg` を import して使う可視化専用のスクリプトとなり、
データパイプラインからは import しない。

## 出力レイアウト

```text
<out>/
  manifest.json                 — 生成条件と split ごとの結果
  <split>/<sample_id>.json      — canonical sample(metagraph_schema.md 準拠)
```

manifest:

```json
{
  "schema_version": 1,
  "generator": {"name": "generate_cfg", "version": "<git commit>", "config": {...}},
  "code": {"commit": "<git commit>", "dirty": false},   ← CLI 実行時のみ
  "splits": {
    "train": {
      "seed_range": [0, 800],
      "kept": 793, "dropped_duplicates": 7,
      "samples": [{"seed": 0, "sample_id": "..."}],
      "dropped": [{"seed": 5, "duplicate_of": "<sample_id>"}]
    }
  }
}
```

タイムスタンプは含めない。同一引数からは manifest・全ファイルともバイト同一
(決定性)。seed と config が provenance に入るため、任意サンプルの CFG は
manifest から再生成できる。

## split 分離と重複排除(A3-4 の実装)

1. split の seed range は**互いに素**であることを検証し、重複していれば拒否する。
2. 各 CFG について WL hash(`weisfeiler_lehman_graph_hash`, directed 対応の
   nx >= 3.5)で候補 bucket を作る。hash 一致は同型の証明ではない。
3. 同一 bucket 内の候補と directed graph isomorphism(`nx.is_isomorphic`)で
   照合し、同型なら drop して `duplicate_of` に生存サンプルを記録する。
4. 照合は **split 横断**(dataset 全体)で行う。最初の出現(split の挿入順、
   通常 train が先)が生存する。

現在は全ノードが単一 node_type のため構造のみが同一性である。node_type を
多様化する際は、hash の `node_attr` と matcher の `node_match` の両方に
属性を渡す(コード内コメント参照)。

## CLI

```bash
uv run python -m cfg_reducer.dataset \
  --out dataset/ \
  --split train=0:800 --split val=800:1000 \
  --num-nodes 12 --edge-prob 0.18
```

`--version` 省略時は `git rev-parse --short HEAD` を記録する。

## 将来: 実 CFG(`pyClangAST`)の接続

`build_dataset()` は generator を注入可能(テストでも使用)だが、実 CFG は
seed 列挙ではなく provenance group(repository/file/function)単位の分離が
必要なため、別の loader 関数として実装し、重複排除
(`fingerprint` / `is_structural_duplicate`)と `store.save_sample` を共有する。

## 追記(2026-09-04、外部レビュー 1-5 / 1-6)

- **generator の同一性**: `build_dataset(generator=...)` は `GeneratorDescriptor(name, fn)`
  を受け取り、provenance と manifest の `generator.name` に **実際に走った生成器の名前**
  を記録する(以前は固定の `GENERATOR_NAME`)。素の callable を渡した場合は関数名で
  包む。生成器 family を増やしても異なる生成器が同名で記録されることはない。
  `sample_id` は provenance 由来なので、生成器名が変われば同 seed でも別 id になる。
- **コード状態**: CLI は manifest に `code: {commit, dirty}` を記録する(provenance
  には入れない。sample_id を生成器の同一性だけの関数に保つため)。dirty な作業木から
  作った dataset を後から見分けられる。
- **同一性のノード集合**: 重複判定の `DiGraph` を edge だけでなく node 集合からも構築
  し、孤立ノードを同一性に含める(`cfg_nodes()`)。現生成器は線形骨格を先に張るため
  孤立ノードは出ないが、family 追加前の前提整備。
