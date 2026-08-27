# MetaGraph canonical JSON schema (draft v1)

作成日: 2026-08-28
状態: **レビュー待ち**(承認後に encode/decode と round-trip test を実装する)

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
| `schema_version` | int | この文書の版。互換性が壊れる変更で増やす |
| `sample_id` | string | データセット内で一意なサンプル ID |
| `provenance` | object | 生成元の追跡情報。`source` は必須(下記) |
| `metagraph` | object | トップレベルの MetaGraph(下記) |

`provenance.source` の値と追加フィールド:

- `"synthetic"` — `generator: {name, version, seed, config}`。
  seed range の split 分離(A3-4)に使う。
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

## レビュー観点(未確定点)

1. `sample_id` の命名規則(generator 名 + seed の合成か、UUID か)。
2. synthetic の `provenance.generator.config` に含める最小フィールド。
3. 元 CFG の edge list を全 source で保存するか(`fixture` のみ必須とするか)。
   保存すれば round-trip 検証が自己完結するが、サイズが増える。
4. `schema_version` を envelope だけに置くか、再帰 `graph` にも置くか
   (現案: envelope のみ)。
