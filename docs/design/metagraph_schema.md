# MetaGraph canonical JSON schema (draft v1)

作成日: 2026-08-28
状態: **承認済み**(2026-09-03 レビュー確定。実装は `cfg_reducer/store.py` の
`encode_sample` / `decode_sample` / `save_sample` / `load_sample`)

`docs/handoff_questions.md` の A2/A3/A5 で確定した方針を、実装可能な schema に
落とした設計文書。サンプル fixture は
[`tests/fixtures/metagraph_nested_loop.json`](../../tests/fixtures/metagraph_nested_loop.json)。

## スコープ

- **対象**: 1 サンプル(1 CFG 由来の MetaGraph)の canonical 保存形式。
  architecture-neutral かつ lossless(A3-3)。
- **対象外**: モデル入力向け flat 表現・トークン列。canonical JSON から導出する
  別変換とし、保存上の正本にしない(A3-2)。

## Envelope

```json
{
  "schema_version": 1,
  "sample_id": "fixture-nested-loop-001",
  "provenance": { "source": "..." },
  "metagraph": { }
}
```

| フィールド | 型 | 内容 |
|---|---|---|
| `schema_version` | int | この文書の版。互換性が壊れる変更で増やす。envelope のみに置く |
| `sample_id` | string | データセット内で一意なサンプル ID(下記の導出規則) |
| `provenance` | object | 生成元の追跡情報。`source` は必須(下記) |
| `metagraph` | object | トップレベルの MetaGraph(下記) |

### sample_id の導出

役割を分離する: `sample_id` は**同一性**(参照・重複排除のキー)、`provenance` は
**再現性**(どう再生成するか)を担い、消費側は `sample_id` を不透明な UUID として
扱う(パースしない)。

- `synthetic` / `clang`: 正準化した provenance(キー再帰 sort の compact JSON)を
  名前として、リポジトリ固定 namespace から導出した **UUIDv5** を必須とする。
  導出は「provenance → sample_id」の一方向のみで、同じ provenance からは常に
  同じ ID になる(決定性・冪等性・重複検出が成立)。
- `fixture`: 手書きの一意な文字列でよい(UUIDv5 は要求しない)。

decode 時に ID と provenance の整合は検証しない(整合検査はデータセット生成側の
責務とする)。

`provenance.source` の値と追加フィールド:

- `"synthetic"` — `generator: {name, version, seed, config}`。
  seed range の split 分離(A3-4)に使う。`config` は「`(name, version, seed)` と
  合わせれば CFG を一意に再生成できる最小の生成パラメータ一式」= 生成器関数の
  入力引数そのもの。`version` は本リポジトリの git commit(またはパッケージ版)を
  記録し、変換規則の変遷は git 履歴で追跡する。
- `"clang"` — `repository`, `file`, `function`。
  provenance group をまたいだ split 分離(A3-4)に使う。
- `"fixture"` — 手書きレビュー用。`description` と、再検証用の
  `cfg_edges`(元 CFG の edge list)を持つ。

## MetaGraph object(再帰)

```json
{
  "motifs": [ ],
  "edges": [[0, 5], [5, 6]],
  "subgraphs": [
    {"loop_step": 5, "graph": { }}
  ]
}
```

- `motifs` — この階層の Motif 配列。**`step` 昇順**。
- `edges` — `[src_step, dst_step]` の配列。**辞書順 sort**(現行
  `metagraph.build()` の出力と同一)。
- `subgraphs` — `{loop_step, graph}` の配列。**`loop_step` 昇順**。
  Python の `dict[int, MetaGraph]` を JSON object にすると integer key が
  string 化されるため、配列形式を採用する(A3-2)。

## Motif object

```json
{"kind": "loop", "node": null, "preds": ["A"], "succs": ["E"],
 "meta": {"header": "B", "scc": ["B", "C", "D"], "back_edges": [["D", "B"]]},
 "step": 5}
```

| フィールド | 型 | 内容 |
|---|---|---|
| `kind` | string | `entry` / `linear` / `merge` / `loop` の 4 種のみ(A5-1) |
| `node` | string \| null | 復元ノード ID。`loop` のみ null |
| `preds` / `succs` | string[] | 復元時の隣接ノード ID(sort 済み)。loop は外部インターフェース |
| `meta` | object | 非 loop は `{}`。loop は `header`(string)、`scc`(sort 済み string[])、`back_edges`(sort 済み `[src, dst][]`) |
| `step` | int | サンプル内で一意な永続 ID(A3-1)。再採番しない |

### ルール

1. `step` は `motif.extract()` が振った値をそのまま保存する。`edges` と
   `subgraphs[].loop_step` はこの ID を参照する。階層ごとの連番が必要な場合は
   読み込み側の derived field とし、canonical ID との対応表を保持する(A3-1)。
2. `Motif.children` は**シリアライズしない**。Loop の children と
   `subgraphs` 内の `graph.motifs` は同一物であり(不変条件)、children は
   subgraph 再帰から復元できる。重複保存はしない。
3. 未知の `kind` を含む入力は、encoder が **validation error** にするか、
   **明示的な skip reason** 付きで除外する。暗黙に学習データへ混ぜない(A5-1)。
4. Loop の子 Motif の `preds`/`succs` は階層外ノード(例: header の外部 pred)を
   含み得る。lossless 性のためそのまま保存する。エッジ構築時に無視される点は
   現行 `metagraph.build()` と同じ。
5. すべての配列は上記の sort 規則に従い、同一入力から常にバイト同一の出力を
   得られるようにする(決定性)。JSON は UTF-8、キー順は上記表の記載順を推奨。

## 導出される flat 表現(参考・対象外)

モデル入力用の flat table は canonical JSON から生成し、各行に
`parent_loop_step`(トップレベルは null)と `depth` を付与する(A3-2)。
仕様の確定は自己回帰 baseline の architecture spike 時に行う(A3-3)。

## レビュー結果(2026-09-03 確定)

1. `sample_id` — 消費側には不透明な UUID。synthetic / clang は正準化 provenance
   からの UUIDv5 で導出(上記「sample_id の導出」)。fixture は任意の一意文字列可。
2. `provenance.generator.config` — 生成器関数の入力引数一式(最小)。`version` は
   git commit で固定(上記 provenance 節)。
3. 元 CFG の edge list(`cfg_edges`)— `fixture` のみ必須。synthetic は seed
   再生成、clang は provenance 参照で代替する。
4. `schema_version` — envelope のみ。生成源が一意に定まれば変換規則は本リポジトリの
   git で管理できるため、再帰 `graph` には埋めない。
