# CFG生成モデル設計 — 議論ログ

## 2026-06-24 セッション1

### 背景・目的

- `gr` のリダクションOp履歴から抽出した **Motif**（Entry / Linear / Merge / Loop）を単位としてCFGを生成するモデルを構築したい
- 最終目標: Graph Transformerのグラフ結合を表現するMASKの生成機能の獲得
- まずMotifの配置構造（トポロジー）の生成に集中し、ノードトークン割当は後段フェーズとする

### 決定事項

1. **Motifの表現形式: メタグラフ（DAG）**
   - Motif列を線形シーケンスではなく、Motif間の依存関係を明示したDAGとして扱う
   - 各Motifの `preds`/`succs` が先行Motifで復元されたノードを参照 → Motif間にDAG的な依存構造が存在
   - この構造を明示的にモデル入力に持たせることで、次Motifノード予測の学習に繋がると期待

2. **実験環境: Google Colab CLI**
   - 学習実験はColab上で実行する方針
   - データ生成パイプライン（`gr` 側）はローカル、モデル学習はColab — の分離を基本とする
   - Colab CLIの具体的な使い方は学習対象の仕様確定後に決定

### Google Colab CLI メモ

ターミナルからColab VMをプロビジョニングしてリモート実行できるCLIツール。

```bash
# インストール
uv tool install google-colab-cli

# 基本ワークフロー
colab new -s trainer --gpu A100          # VM起動（セッション名: trainer）
colab install -s trainer torch transformers  # パッケージ導入
colab upload local_data.json /content/data.json  # ローカルファイル転送
colab exec -s trainer -f train.py        # スクリプト実行
colab download -s trainer checkpoints/model.bin ./model.bin  # 結果回収
colab stop -s trainer                    # VM停止

# ワンショット（VM起動→実行→自動停止）
colab run --gpu T4 train.py
```

- GPU選択肢: T4, L4, G4, H100, A100, TPU v5e1/v6e1
- `colab drivemount` でGoogle Driveマウント可
- 認証: Google ADC（デフォルト）またはOAuth2
- Colab Pro/Pro+のコンピュートユニットが必要（GPU利用時）

### 未決事項・議論中

- [x] Motifメタグラフの具体的なエッジ定義 → セッション2で確定
- [ ] モデルアーキテクチャの選択（自己回帰 vs 離散拡散 vs 階層展開）
- [ ] データ生成パイプラインの具体仕様（バッチ生成、出力フォーマット）
- [ ] Motif語彙の粒度（4種で十分か、サブタイプを設けるか）

### 議論: Motifメタグラフのエッジ定義

**提案A: ノード共有による依存（基本方針）**

Motif `M_i` が復元したノードが、後続 Motif `M_j` の `preds` or `succs` に含まれる場合 `M_i → M_j`。

例（ダイヤモンド A→B, A→C, B→D, C→D）:
```
step 0: Entry(A)   — preds=(), succs=()
step 1: Linear(B)  — preds=(A,) → Entry(A) に依存
step 2: Linear(C)  — preds=(A,) → Entry(A) に依存
step 3: Merge(D)   — preds=(B,C) → Linear(B), Linear(C) に依存

メタグラフ: Entry(A) → Linear(B) → Merge(D)
            Entry(A) → Linear(C) → Merge(D)
```

**提案B: Loop Motifの特殊扱い**

Loop Motifは `node=None`（ノード復元なし）のため、提案Aだけでは孤立する。
`meta["scc"]` メンバーを復元したMotifとの間にエッジを張る拡張が必要。

**論点: エッジ種別を持たせるか**

- 方向のみ（`M_i → M_j`）で十分か？
- エッジ種別（pred依存 / succ依存 / loop-member）も持たせるべきか？

### 決定: Motif入れ子構造（Loopコンテナ化）

**動機**: 現在の `motif.extract()` はOp列をフラットに逆順展開するだけで、「ループ内部のMotifがどのLoopに属するか」という親子関係が消失していた。リダクションアルゴリズムの `Scope` enter/leave がこの階層構造を既に持っているので、Motif側にも反映する。

**設計**:

1. `Motif` に `children: tuple[Motif, ...] = ()` を追加。非Loop Motifは空、Loopのみコンテナ
2. `extract()` でOp列を正順走査し `scope_snapshot`/`scope_after` からOp間の親子関係を構築、逆順でMotifツリーを組み立て
3. Loop Motif の `preds`/`succs` は**SCCの外部インターフェース**（SCC外→header入辺の始点、SCCメンバー→SCC外出辺の終点）
4. 外部インターフェース情報はOp.inverseに事前記録せず、抽出時にSCCメンバー接続から**その場で計算**。パフォーマンス問題が出ればキャッシュを検討するが優先度低。

**入れ子Motifの例** (`A→B→C→B, C→D`):
```
step 0: Entry(A)               preds=(), succs=()
step 1: Loop(header=B, scc={B,C})  preds=(A,), succs=(D,)
  ├── Entry(B)                 preds=(), succs=(C,)  [ループ内]
  └── Linear(C)                preds=(B,), succs=()  [ループ内]
step 2: Linear(D)              preds=(C,), succs=()
```

**将来の拡張余地**: if-else分岐のBranch Motif導入によりMergeの子にできる可能性があるが、アルゴリズム側のスコープ切り出し拡張を伴うため現時点では見送り。

**実装完了 (2026-06-24)**: `Motif.children` フィールド追加、`motif.extract()` をスコープ追跡対応に書き換え。検証済みケース:
- Diamond (A→B,C→D): フラット4 Motif
- Simple loop (A→B→C→B, C→D): Loop(preds=A, succs=D) + 子2件
- Nested loop (A→B→C→D→C, D→B, B→E): 外Loop内に内Loopが入れ子
- build_cfg 8ノード: Loop内Merge含む複合構造

---

## 2026-06-25 セッション2

### 開発フロー決定

- **設計・議論**: Claude Codeセッションで行う
- **実装委譲**: Codex CLI（GPT-5.4）に仕様を渡して実装させる
  - アウトカム指向でプロンプトを書く（手順ではなく「何を実現するか」）
  - ファイルパス・既存パターン・制約を明示
  - コードレビューにも活用可（評価軸を具体的に指定）
- **小規模修正**: セッション内で直接実施

### 決定: メタグラフ エッジ定義

**Loopコンテナ化により統一ルールが成立**:

Motif `M_i` が復元したノードが、Motif `M_j` の `preds` or `succs` に含まれるとき `M_i → M_j`。

- 非Loop: `M_i.node` が復元ノード
- Loop: `M_i.meta["scc"]` の全メンバーが復元ノード群（メタグラフ上はLoopが代表）

Loopが外部インターフェース（`preds`/`succs`）を持つため、提案B（Loop特殊扱い）は不要になった。

**エッジ種別は当面不要**: 方向のみで十分。pred依存/succ依存/loop-memberなどの種別追加は精度頭打ち時の拡張オプション。

**例**:
```
Diamond:  Entry(A) → Linear(B) → Merge(D)
          Entry(A) → Linear(C) → Merge(D)

Loop:     Entry(A) → Loop({B,C}) → Linear(D)
```

### 決定: 階層的メタグラフ

**トップレベル**: Loop Motifは中身を隠蔽した1ノード。CFGの大局的な流れだけが見えるDAG。

**Loop内部**: childrenだけを対象に同じエッジ構築アルゴリズムを再帰適用 → サブメタグラフ。ネストしたLoopがあればさらに再帰。

**将来の拡張余地**: Linear/分岐（3つ以上の分岐含む）もサブグラフを内包する可能性がある（例: アルゴリズムの列挙がトップレベル、各Linearが内部に分岐やループを持つ）。しかし現時点ではLoopだけ。

### 決定: MetaGraph データ構造

Motifは`frozen`なのでエッジ情報を埋め込まず、外部構造として分離:

```python
@dataclass(frozen=True)
class MetaGraph:
    motifs: tuple[Motif, ...]
    edges: tuple[tuple[int, int], ...]   # (src_step, dst_step)
    subgraphs: dict[int, 'MetaGraph']    # loop_step → child MetaGraph
```

**利点**: 同じMotifリストに対して異なるエッジ定義を試すことが容易。

### 決定: metagraph.py 分離

パイプラインの処理フロー:

```
algorithm.py → engine.py → Op列
  → motif.py (Op → Motifツリー)
  → metagraph.py (Motifツリー → MetaGraph DAG)
  → [将来] データパイプライン / モデル入力
```

`motif.py` の責務は「Op→Motif抽出」、`metagraph.py` は「Motifツリー→DAG構造」。変換の種類が異なるため分離。

### 実装計画

`docs/plans/metagraph_impl.md` に詳細仕様を記載。Codex CLI（GPT-5.4）で実装予定。

対象ファイル:
- `cfg_reducer/types.py` — MetaGraph追加
- `cfg_reducer/metagraph.py` — 新規作成（`build()`関数）
- `cfg_reducer/__init__.py` — エクスポート追加
- `tests/test_metagraph.py` — テスト3件（diamond, simple loop, nested loop）

---
