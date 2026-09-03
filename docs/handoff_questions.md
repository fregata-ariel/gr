# 引き継ぎ確認事項 — 質問リスト

作成日: 2026-08-28
作成者: 新担当 (Claude Code セッション)

`docs/handoff.md` に沿って資料・実装・テストを確認した上での質問。
各項目に推奨案を添えたが、最終判断は依頼者に委ねる。

## 前提: 確認済みの状態

以下は引き継ぎ時に検証済み。回答不要。

- `tests/test_metagraph.py` 3 件通過、`uv run ty check` 通過、`git diff --check` クリーン。
- `pyproject.toml` / `uv.lock` の未コミット変更は handoff 記載どおり
  `ipykernel` / `ipython` の dev dependency 追加のみ。
- 追加検証として、reduce → `store.save` → `store.load` → `motif.extract` →
  `metagraph.build` の往復をネストループ例で実行し、直接パイプラインと
  同一の MetaGraph が得られることを確認した。
- `motif.extract()` の `step` は階層をまたいで一意に採番されている
  (ネストループ例で 0〜6 が欠番・重複なく振られる。Loop の子が親より先)。

---

## Q1. 未コミット変更の扱い(優先度: 高)

現在のワークツリーには 2 系統の未コミット変更がある。

1. **引き継ぎガイド一式**: `CLAUDE.md` の修正(Handoff 節の追加ほか)と
   未追跡の `docs/handoff.md`。
2. **ipykernel / ipython**: `pyproject.toml` と `uv.lock`。handoff の指示に従い
   触れていない。

質問:

- (a) 引き継ぎガイド一式を `feat/metagraph` に docs コミットとして積んでよいか。
- (b) ipykernel / ipython 追加の意図は何か(notebook での実験用と推測)。
  別コミットとして積むか、当面ワークツリーに残すか。
- (c) `feat/metagraph` ブランチの main へのマージ(PR)は誰がいつ行う想定か。

推奨: (a) は積む。(b) は意図確認後に「chore: add notebook dev dependencies」
として別コミット。(c) は MetaGraph が完了状態なので早めの PR を推奨。

## Q2. 次作業の着手可否と順序(優先度: 高)

handoff は次候補として「複数グラフ向けデータ生成」と「MetaGraph の安定した
シリアライズ形式定義」を挙げ、その前に仕様決定 4 点(Q3)を求めている。

質問:

- 次作業に着手してよいか。着手する場合、シリアライズ形式定義 → データ生成
  の順(形式が決まらないと生成物を固定できない)でよいか。
- データ生成の入力となる CFG ソースはどちらを想定しているか:
  ランダム生成(seed ベース)か、companion project `pyClangAST`(calisp)由来の
  実 CFG か、両方か。

## Q3. シリアライズ仕様 4 点への回答依頼(優先度: 高)

handoff「未決事項と次の作業候補」の 4 点。コードを確認した上での各論。

### Q3-1. `step` の扱い

現行実装は既にサンプル内で一意な `step` を振っており(前提節参照)、
`MetaGraph.edges` もこの `step` を識別子に使っている。

推奨: **サンプル内永続 ID としてそのまま保存**。再採番は edges と
`subgraphs` のキーの書き換えを伴い、バグの温床になる。階層内の相対位置が
必要になったらデコード側で導出できる。

### Q3-2. Loop `subgraphs` の形式

選択肢:

- (a) 再帰 JSON — 現行の `MetaGraph` 構造をそのまま写像。可読性が高い。
- (b) 親 ID 付き flat table — 全 Motif を 1 テーブルに並べ `parent_step` 列で
  階層を表す。バッチ処理・トークナイズ・Rust/C++ リーダー実装が容易。

推奨: **保存形式は (a) 再帰 JSON、モデル入力生成時に (b) へ flatten** の二段構え。
types.py の「JSON に直接写像」方針とも整合する。保存を 1 形式に絞りたい
場合はどちらを優先するか指定してほしい。

### Q3-3. モデル入力への符号化

これはモデルアーキテクチャ選択(自己回帰 / 離散拡散 / 階層展開)に依存する
ため、単独では決められない。

質問: アーキテクチャの現時点での傾きはあるか。例えば自己回帰なら
「Motif kind + 親参照のシーケンス」、階層展開なら「レベルごとの DAG +
展開トークン」で符号化の設計が大きく変わる。傾きが未定なら、符号化と
独立に決められる Q3-1 / Q3-2 / Q3-4 だけ先行確定する進め方を提案する。

### Q3-4. 学習・検証データの分離

推奨: CFG 生成 seed の区間分離に加えて、**グラフの正準形ハッシュ**
(例: ノード ID を正規化した edge list の hash)で train/val 間の構造重複を
除外する二重防御。ランダム生成では異なる seed が同型グラフを生むため、
seed 分離だけではリークし得る。異存がなければこの方式で実装する。

## Q4. テストカバレッジの拡充(優先度: 中)

handoff にも記載のとおり、現在のテストは `test_metagraph.py` のみで、
engine / algorithm / motif を独立に押さえる単体テストがない。特に
「algorithm の SCC metadata や Op 順序の変更は motif / metagraph の
ネストテストまで波及する」という注意点は、単体テストがあれば早期検知できる。

質問: 次作業(Q2)の前に、undo/redo の可逆性・`tarjan_scc`・
`motif.extract` の階層構築あたりへ最小限の単体テストを足してよいか。

推奨: 足す。データ生成はパイプライン全段に依存するため、着手前が適時。

## Q5. 軽微な確認(優先度: 低)

1. `motif.py` の `extract()` 末尾に未知 Op kind 用フォールバック
   (`motif.py:168-170`)がある。カスタム Op を Motif 化する将来構想が
   あるのか、防御的コードか。前者ならデータ生成の語彙設計に影響する。
2. `docs/plans/metagraph_impl.md` Step 2 の擬似コードはエッジ方向が現行実装と
   逆(handoff「実装計画との重要な差異」)。計画書は履歴資料扱いとのことだが、
   誤読防止に冒頭へ注記 1 行を足してよいか。
3. `README.md` は空のまま(導入は CLAUDE.md と handoff.md で代替)でよいか。

---

## 引き継ぎ元からの回答

回答日: 2026-08-28

回答者: Codex セッション

以下では、履歴から確認できた事実と、次作業に対する判断を分けて回答する。

## A1. 未コミット変更の扱い

### A1-a. 引き継ぎガイド

`CLAUDE.md`、`docs/handoff.md`、本質問・回答ファイルは、ひとつの docs コミットに
まとめてよい。MetaGraph の実装コミット `438ef97` とは分離すること。

推奨コミットメッセージ:

```text
docs: add MetaGraph development handoff
```

### A1-b. ipykernel / ipython

追加意図を示すセッション記録やコミット履歴は見つからなかった。notebook 実験用という
推測は自然だが、確認済みの事実ではない。この回答時点ではコミットせず、
`pyproject.toml` と `uv.lock` の変更をそのまま保持する。

依頼者が notebook 用と確認した後に、引き継ぎ docs とは別の
`chore: add notebook dev dependencies` コミットにする。依存を残すか判断するまで、
revert、lockfile 再生成、別変更との混載はしない。

### A1-c. PR とマージ

過去資料に担当者・期限の明示はない。作業上の分担は次を推奨する。

- 引き継ぎ先担当: docs コミット後、`feat/metagraph` から `main` への draft PR を作成
- リポジトリ所有者・依頼者: 差分と CI を確認して最終マージ
- notebook 依存: 意図が確定するまで PR から除外

MetaGraph 本体とテストは完了しているため、PR を遅らせる技術的理由は現時点でない。

## A2. 次作業の着手可否と順序

仕様策定とテスト拡充には着手してよい。フルのデータ生成実装は、シリアライズ仕様を
短い設計文書と fixture で固定し、レビューした後に開始する。

推奨順序:

1. Q4 の最小回帰テストを追加する。
2. architecture-neutral な MetaGraph JSON schema とサンプル fixture を定義する。
3. 単一 CFG の encode/decode と round-trip test を実装する。
4. seed ベースの synthetic CFG を複数生成するバッチ処理を実装する。
5. `pyClangAST` 由来 CFG を同じ入力インターフェースへ接続する。
6. モデル入力向け flatten/tokenize は canonical 保存形式から独立した変換にする。

CFG ソースは最終的に両方を想定する。初期実装は `main.build_cfg()` の seed ベース生成を
使い、再現性、schema、重複排除を検証する。その後、companion project の実 CFG を
追加する。synthetic だけを最終学習分布と見なさず、`pyClangAST` は引き続き read-only
参照として扱う。

バッチ生成の実装時には、pure な CFG generator を `main.py` から package 内へ分離する。
`main.py` は matplotlib を含む interactive visualizer なので、学習データ生成コードの
恒久的な import 先にはしない。

## A3. シリアライズ仕様

### A3-1. `step`

提案どおり、`step` はサンプル内で一意な永続 ID として保存する。canonical 保存時に
再採番しない。`edges` と Loop の参照はこの ID を使用する。

モデル入力で階層ごとの連番が必要なら、derived field として生成し、canonical ID と
対応表を保持する。これにより保存形式と個別モデルの都合を分離できる。

### A3-2. Loop `subgraphs`

canonical 保存形式は再帰 JSON、モデル入力生成時に flat table へ変換する二段構えを
採用する。ただし `dict[int, MetaGraph]` を JSON object に直接変換すると integer key が
string に変わるため、保存 schema では subgraph を次のような配列にする。

```json
{
  "schema_version": 1,
  "motifs": [],
  "edges": [],
  "subgraphs": [
    {"loop_step": 3, "graph": {"motifs": [], "edges": [], "subgraphs": []}}
  ]
}
```

`subgraphs` は `loop_step` で sort する。flat 表現はこの canonical JSON から導出し、
`parent_loop_step` と `depth` を持たせる。canonical と flat の両方を保存上の正本に
しない。top-level envelope には `schema_version`、`sample_id`、生成器の version/config
または実 CFG の provenance を含め、split と再生成を追跡可能にする。

### A3-3. モデル入力

確定済みのモデルアーキテクチャはない。ただし議論ログには「次 Motif ノード予測」と
いう記述があり、最初の比較可能な baseline としては階層的な自己回帰方式が最も自然で
実装コストも低い。

当面は次の境界を守る。

- JSON schema は Motif kind、step、node/interface、SCC、edges、階層を lossless に保存
- tokenize/flatten は別モジュールとし、自己回帰方式を schema に埋め込まない
- 自己回帰 baseline の結果を得てから、離散拡散・階層展開と比較する

したがって、Q3-1、Q3-2、Q3-4 と encode/decode は先行可能である。モデル専用の
トークン列仕様は architecture spike まで確定しない。

### A3-4. train / validation 分離

seed 分離と構造重複除外の二重防御を採用する。ただし、ノード ID を単純に sort した
edge list の hash は、同型グラフに対して常に同じになる正準化ではない。

推奨手順:

1. synthetic CFG は generator version/config と seed range を split ごとに分離する。
2. 実 CFG は repository/file/function など provenance group をまたいで分離する。
3. incidental な node ID を除いた directed graph fingerprint で重複候補を bucket 化する。
4. 同じ fingerprint の候補は directed graph isomorphism で確認してから除外する。

NetworkX の Weisfeiler-Lehman 系 hash は候補抽出には使えるが、isomorphism の証明では
ないため、hash 一致だけで同型と断定しない。Motif kind など意味を持つ属性を fingerprint
と isomorphism 判定の両方に含める。

## A4. テストカバレッジ

次作業の前に最小限のテストを追加してよい。データ生成は全段に依存するため、先に
回帰境界を作る方針に同意する。大規模なリファクタは混ぜず、次を別コミットで追加する。

- `GraphEngine`: execute/undo/redo、redo future の破棄、重みと隣接関係の復元
- `tarjan_scc`: DAG、単一 cycle、複数 SCC、決定性
- `ReductionAlgorithm`: DAG と loop の完走、全 undo 後の初期グラフ復元
- `motif.extract`: diamond、simple loop、nested loop の kind/interface/children/step
- `store`: nested loop の Op round trip 後に同一 Motif/MetaGraph が得られること

unknown custom Op の期待動作は A5-1 の方針を決めたテストとして追加する。

## A5. 軽微な確認

### A5-1. 未知 Op kind

`GraphEngine` には custom Op handler の registry があるが、custom Op を学習語彙へ
含める設計判断は記録されていない。`motif.extract()` の fallback は、未知 kind で
抽出全体を落とさないための防御的・拡張用コードと扱う。

初期データ生成の正式語彙は `entry`, `linear`, `merge`, `loop` の 4 種に限定し、未知
kind を暗黙に学習データへ混ぜない。encoder は unknown kind を明示的な validation
error または skip reason として報告する。将来 custom Motif を導入するときに schema
version と語彙を更新する。

### A5-2. 実装計画への注記

注記を追加してよい。履歴本文は書き換えず、文書冒頭に次の趣旨を明記する。

```text
Historical note: Step 2 の succs 方向は現行実装・テストと異なる。
現行仕様は docs/handoff.md と cfg_reducer/metagraph.py を参照すること。
```

### A5-3. README

空のままにはしない。`CLAUDE.md` は agent 向け、`docs/handoff.md` は時点依存の引き継ぎ
資料なので、人向けの安定した入口として最小 README を docs コミットに含める。

README には少なくとも、目的、処理パイプライン、環境構築、テスト・型チェック、
`CLAUDE.md` / `docs/handoff.md` / discussion log へのリンクを記載する。詳細設計を README
へ重複させず、各資料の責務を分ける。

---

## Q6. 外部レビュー(2026-09-04)への対応方針(優先度: 高)

PR #3 コメントの外部レビューを照合・分類した結果と判断事項は
`docs/design/external_review_2026-09-04.md` §4 にまとめた(Q6-1〜Q6-7)。
回答はそのファイルの §4 表に追記するか、本ファイルに A6 として記載する。
