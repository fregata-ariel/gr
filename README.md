# gr — CFG Reducer

Control Flow Graph を段階的に縮約し、その縮約履歴(Op history)を構造的な
building block(**Motif**)として再解釈するエンジン。抽出した Motif 階層は
依存 DAG(**MetaGraph**)に変換され、Graph Transformer による CFG 生成の
学習データとなることを目指している。

## 処理パイプライン

```text
GraphEngine                 グラフ変更と undo/redo 履歴の管理
  -> ReductionAlgorithm     reverse-Kahn + SCC cycle breaking で Op を生成
  -> Op history             全変更の forward/inverse 記録(単一の真実)
  -> motif.extract()        Op 履歴から階層 Motif ツリーを抽出
  -> metagraph.build()      Motif 階層を依存 DAG(MetaGraph)へ変換
  -> [開発中] データセット化 / モデル入力
```

Motif は entry / linear / merge / loop の 4 種。Loop は SCC 全体を代表する
コンテナで、内部構造を子 Motif として保持する。

## 環境構築

Python 3.13+ と [uv](https://docs.astral.sh/uv/) を使用する。

```bash
uv sync            # 依存導入(dev 含む)
uv run python main.py   # インタラクティブ可視化(matplotlib)
```

## テスト・型チェック

```bash
uv run python -m pytest   # テスト
uv run ty check           # 型チェック
```

## 資料の入口

| 資料 | 対象・内容 |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | エージェント向けのプロジェクト規約と構成 |
| [`docs/handoff.md`](docs/handoff.md) | 開発引き継ぎガイド(ブランチ状態、不変条件、検証手順) |
| [`docs/discussion_log.md`](docs/discussion_log.md) | 設計判断の経緯と未決事項 |
| [`docs/plans/`](docs/plans/) | 実装計画(履歴資料) |

詳細設計は各資料を参照し、本 README には重複させない。
