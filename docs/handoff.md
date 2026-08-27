# 開発引き継ぎガイド

最終確認日: 2026-08-28

## 目的

この文書は、`gr` の設計経緯、MetaGraph 実装、現在の Git 状態を、
次の担当者が短時間で再構築するための入口である。

## 現在の状態

| 項目 | 状態 |
|---|---|
| ブランチ | `feat/metagraph` |
| HEAD | `438ef97 Add MetaGraph DAG construction from hierarchical Motifs` |
| リモート | `origin/feat/metagraph` と同期済み |
| MetaGraph | 実装・公開 API・テストまで完了 |
| テスト | `tests/test_metagraph.py` の 3 件が通過 |
| 型チェック | `uv run ty check` が通過 |
| 既存の未コミット変更 | `pyproject.toml`, `uv.lock` |

MetaGraph の作業は完了している。引き継ぎガイド作成前から存在する未コミット
変更は、dev dependency への `ipykernel` と `ipython` の追加、および対応する
lockfile 更新である。MetaGraph とは別件なので、意図を確認せず破棄・混載しない。

## 最初に読む資料

次の順序で読むと、設計から実装へ無理なく追跡できる。

| 順序 | 資料 | 確認する内容 |
|---|---|---|
| 1 | [`CLAUDE.md`](../CLAUDE.md) | プロジェクトの目的、構成、共通規約 |
| 2 | [`docs/discussion_log.md`](discussion_log.md) | Motif 階層化と MetaGraph の設計判断、未決事項 |
| 3 | [`docs/plans/metagraph_impl.md`](plans/metagraph_impl.md) | MetaGraph の要求仕様、対象ファイル、テスト構造 |
| 4 | [`cfg_reducer/types.py`](../cfg_reducer/types.py) | `Op`, `Motif`, `MetaGraph` のデータ契約 |
| 5 | [`cfg_reducer/algorithm.py`](../cfg_reducer/algorithm.py) | CFG を縮約して `Op` 履歴を生成する処理 |
| 6 | [`cfg_reducer/motif.py`](../cfg_reducer/motif.py) | `Op` 履歴から階層 Motif を抽出する処理 |
| 7 | [`cfg_reducer/metagraph.py`](../cfg_reducer/metagraph.py) | Motif 階層から依存 DAG を構築する処理 |
| 8 | [`tests/test_metagraph.py`](../tests/test_metagraph.py) | diamond、単純ループ、ネストループの期待動作 |

`docs/plans/metagraph_prompt.txt` は実装依頼に使った履歴資料であり、現行仕様の
確認には実装計画とテストを優先する。`README.md` は現在空なので、導入資料としては
`CLAUDE.md` と本書を使用する。

## 処理パイプライン

```text
GraphEngine
  -> ReductionAlgorithm
  -> Op history
  -> motif.extract()
  -> hierarchical Motif tree
  -> metagraph.build()
  -> hierarchical MetaGraph DAG
  -> [未実装] データセット化 / モデル入力
```

- `GraphEngine` がグラフ変更と undo/redo 履歴を管理する。
- `ReductionAlgorithm` は reverse-Kahn と SCC cycle breaking で `Op` を生成する。
- `motif.extract()` は `Op` を復元順に解釈し、Loop を子 Motif のコンテナにする。
- `metagraph.build()` は各階層を独立した DAG に変換し、Loop 内部を再帰処理する。
- `store.py` が保存できるのは現在 `Op` 履歴だけで、Motif/MetaGraph の永続化形式は
  まだ定義されていない。

## MetaGraph の不変条件

1. `MetaGraph.motifs` と `MetaGraph.edges` は決定的な tuple として返す。
2. エッジの識別子には Motif の `step` を使う。
3. 非 Loop Motif は `node` を、Loop Motif は `meta["scc"]` の全ノードを代表する。
4. `preds` の参照は `参照先 Motif -> 現在の Motif` として扱う。
5. `succs` の参照は `現在の Motif -> 参照先 Motif` として扱う。
6. 同一階層にないノード参照は無視し、階層をまたぐエッジを作らない。
7. Loop の `children` は `subgraphs[loop.step]` に再帰的に格納する。
8. 自己エッジを除外し、重複を除去してからエッジを sort する。

### 実装計画との重要な差異

`docs/plans/metagraph_impl.md` の Step 2 は、`preds` と `succs` の双方を
`参照先 Motif -> 現在の Motif` にする疑似コードになっている。このままでは単純
ループの出口が `Linear(D) -> Loop` となり、期待される `Loop -> Linear(D)` と逆になる。

現行実装では `preds` と `succs` を分けて上記 4、5 の向きで処理している。
これは 3 つのテストと設計例に一致する。エッジ仕様を見直す場合は、計画書だけを
根拠に戻さず、Loop の外部インターフェースと全テストを同時に再評価すること。

## 検証手順

```bash
uv run python -m pytest tests/test_metagraph.py -v
uv run ty check
git diff --check
```

2026-08-28 時点の期待結果:

- pytest: `3 passed`
- ty: `All checks passed!`
- `git diff --check`: 出力なし

現在の `tests/` には `test_metagraph.py` だけがある。したがって、この結果は
MetaGraph の回帰確認にはなるが、engine、algorithm、motif 全体を独立して網羅する
テストスイートではない。

## Git 履歴

| Commit | 日付 | 内容 |
|---|---|---|
| `9b79c5f` | 2026-06-24 | `Op` 履歴からの Motif 抽出を追加 |
| `9db3257` | 2026-06-24 | Loop Motif の階層化と設計ログを追加 |
| `c1d89be` | 2026-06-26 | MetaGraph 実装計画と実装依頼文を追加 |
| `e1801ba` | 2026-06-26 | MetaGraph の設計判断を議論ログへ反映 |
| `438ef97` | 2026-06-27 | MetaGraph、公開 API、3 テストを実装 |

`438ef97` は `e1801ba` をベースに作成され、次を含む。

- `MetaGraph` frozen dataclass
- `metagraph.build()`
- package-level exports
- diamond、simple loop、nested loop のテスト
- pytest 設定と dev dependency

## Codex セッション

MetaGraph 実装時のローカルセッション ID:

```text
019f0976-b94b-7220-beca-8bd83f65206d
```

同じローカル環境で会話を再表示する場合:

```bash
codex resume 019f0976-b94b-7220-beca-8bd83f65206d
```

このセッションは 2026-06-27 に `e1801ba` から開始され、実装、計画書のエッジ方向の
不整合調査、テスト通過までを記録している。Codex 内では `/resume`、現在のリポジトリの
最新セッションには `codex resume --last` も利用できる。

## 未決事項と次の作業候補

設計ログに残っている主要な未決事項:

- モデルアーキテクチャの選択: 自己回帰、離散拡散、階層展開
- バッチデータ生成と出力フォーマット
- Motif 語彙を 4 種のままにするか、サブタイプを追加するか
- 必要になった場合の MetaGraph エッジ種別追加
- Colab 上の学習パイプライン確定

次の具体的な実装候補は、`Op -> Motif -> MetaGraph` の複数グラフ向けデータ生成と、
MetaGraph の安定したシリアライズ形式の定義である。その前に、以下を仕様として決める。

1. `step` をサンプル内の永続 ID として保存するか、階層ごとに再採番するか。
2. Loop の `subgraphs` を再帰 JSON にするか、親 ID 付きの flat table にするか。
3. ノード ID、Motif kind、SCC membership、edge をモデル入力へどう符号化するか。
4. 学習・検証データ間で CFG 生成 seed とグラフ構造をどう分離するか。

## 引き継ぎ時の注意

- `pyproject.toml` と `uv.lock` の既存変更を MetaGraph 修正と同じコミットに混ぜない。
- `Motif` と `MetaGraph` は frozen dataclass だが、`meta` と `subgraphs` の dict は
  深い意味では immutable ではない。共有後に内容を書き換えない。
- Loop の階層判定は `remove_edges` Op の `meta["scc"]` と Op の並びに依存する。
  algorithm の SCC metadata や Op 順序を変更する場合は、motif と metagraph の
  ネストテストまで実行する。
- 実装の正しさは、ファイル単体ではなく `GraphEngine -> ReductionAlgorithm ->
  motif.extract -> metagraph.build` の一連のパイプラインで確認する。
