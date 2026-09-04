# Model input — flatten / tokenize (design note)

作成日: 2026-09-03
対応: `docs/handoff_questions.md` A3-2(flat 化)/ A3-3(階層的自己回帰 baseline)

## スコープと境界

- canonical JSON(`metagraph_schema.md`)は lossless の正本のまま変更しない。
  本変換は **canonical から導出する片方向の view** であり、保存上の正本にしない。
- セッション 1 の決定どおり、ノードトークン割当は後段フェーズ。トークン列は
  **トポロジーのみ**を運ぶ(node ID・step はトークンに含めない)。
- 実装は `cfg_reducer/model_input.py`。Colab 側バッチ整形(padding 等)は
  学習パイプライン確定時に別途決める。

## 前提となる構造的性質

MetaGraph の各階層で、エッジ `(src, dst)` は常に `src.step < dst.step` を満たす。

- 非 Loop Motif の `succs` は常に空(remove_node は出次数 0 でのみ発火)。
  したがってエッジは「dst の preds 参照」か「Loop の外部 succs」由来。
- preds 参照先は必ず先に復元済み(step が小さい)。Loop の succs 先は
  必ず Loop より後に復元される(step が大きい)。

よって階層内を step 昇順に出力すると、全エッジは**後方参照**
(現在位置から見て過去の Motif への参照)として表現できる。

## Flatten(A3-2)

`flatten(mg) -> list[dict]`。DFS pre-order(Loop 行の直後にその children)で
1 行 = 1 Motif:

| フィールド | 内容 |
|---|---|
| `step` | canonical の永続 ID(対応表の役割) |
| `kind` | Motif kind |
| `parent_loop_step` | 親 Loop の step(トップレベルは null) |
| `depth` | 階層深さ(トップレベル 0) |
| `position` | 階層内の位置(step 昇順で 0 始まり) |
| `in_offsets` | 入エッジの後方オフセット列(`position` 差、昇順) |

## トークン列(階層的自己回帰 baseline)

文法(DFS、階層内は step 昇順):

```text
sample := BOS level EOS
level  := motif*
motif  := KIND_* REF_k* [LOOP_START level LOOP_END]   # LOOP_* は kind=loop のみ
```

- `KIND_ENTRY / KIND_LINEAR / KIND_MERGE / KIND_LOOP` — 4 種語彙(A5-1)。
  未知 kind は tokenize が `ValueError` で拒否する。
- `REF_k` — 同一階層内の k 個前の Motif からの入エッジ(k >= 1、昇順に列挙)。
  「Motif kind + 親参照のシーケンス」(A3-3)の具体化。相対参照にするのは
  絶対位置より系列長への汎化が期待できるため。
- 語彙は `build_vocab(max_offset)` で決定的に構築(`PAD=0`, `BOS`, `EOS`,
  `LOOP_START`, `LOOP_END`, KIND 4 種, `REF_1..REF_max`)。window を超える
  オフセットは `ValueError`(データから `max_offset_needed()` で見積もる)。

例(nested loop fixture: `A -> Loop{B, Loop{C,D}} -> E`):

```text
BOS  KIND_ENTRY
     KIND_LOOP REF_1  LOOP_START
         KIND_LINEAR
         KIND_LOOP REF_1  LOOP_START
             KIND_LINEAR
             KIND_LINEAR REF_1
         LOOP_END
     LOOP_END
     KIND_LINEAR REF_1
EOS
```

kind と参照は独立に扱う(例: Loop 内の header 子 Motif は CFG 上の pred が
階層外にあるため、kind=linear でも階層内 `REF` を持たない)。

## 往復保証と検証

トークン列は step と node ID を落とすため canonical へは戻らないが、
**位置ベースのトポロジー**(kind、階層構造、階層内エッジ)は完全に保存する。

- `sketch_of(mg)` — MetaGraph から位置ベースの入れ子構造を直接導出
- `detokenize(tokens, vocab)` — トークン列を厳密にパースして同じ構造を復元
  (文法違反は `ValueError`)
- テストで `detokenize(tokenize(mg)) == sketch_of(mg)` を固定グラフと
  synthetic 生成グラフの双方で保証する

`detokenize` の strict パーサは、将来モデルが生成したトークン列の
妥当性検査(well-formedness check)にそのまま使える。

## 見送り(将来)

- ノードトークン割当・重み・node_type の付与(後段フェーズ)
- エッジ種別トークン(精度頭打ち時の拡張オプション、セッション 2 決定)
- 離散拡散・階層展開向けの別 view(baseline 比較後)
