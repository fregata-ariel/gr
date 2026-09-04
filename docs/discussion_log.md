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

## 2026-06-29 セッション3

### メタグラフ実装完了・マージ

PR #1 (feat/metagraph) を main にマージ。最終状態:
- `cfg_reducer/types.py` — MetaGraph frozen dataclass追加
- `cfg_reducer/metagraph.py` — `build()` 関数（再帰的サブグラフ構築）
- `tests/test_metagraph.py` — 7テスト（diamond, simple loop, nested loop, linear chain, multi-exit loop, DAG invariant, build_cfg integration）
- `main.py` — `build_cfg()` を `engine` 引数オプショナル化、`GraphEngine` を返すように変更

### 議論: モデルアーキテクチャの選択

**候補**: 階層展開型 Graph Transformer（隣接ノード全体を1ステップで生成）

#### 決定: 順列不変性の要件分離

ノード順列の正規化について議論した結果、2つの異なる不変性要件を混同していたことが判明。明確に分離する。

**A. 生成過程の順序（正規化不要・バイアス歓迎）**

グラフ構築ステップの順序はバイアスがあって良い。構築過程を意味空間上のダイナミクスと捉えると、構築順序は構造の意味論を反映する有用な情報。同一グラフでも異なる構築パスは異なる意味を持ちうる。リダクションのstep順序がこのダイナミクスの記録にあたる。

**B. 完成グラフのトークン表現（順列不変性必須）**

完成グラフを下流モデル（LLM等）にトークン列として入力する際、ノードの並び順が出力を変えてはならない。`{A, B, C}` と `{C, A, B}` は等価に扱われる必要がある。これは生成アーキテクチャの問題ではなく、エンコーダ設計の問題。

**Bの解決策: 構造的位置エンコーディング**

Graph Transformerのself-attentionは集合演算であり、本質的に順列等変。ただしpositional encodingの選択が鍵:
- 絶対位置エンコーディング（sin/cos等）→ ノード順に依存 → **使用禁止**
- 構造的位置エンコーディング（Laplacian固有ベクトル、Random Walk SE）→ グラフ構造から計算、順序非依存 → **採用**
- 隣接行列をattention biasとして入力（Graphormer方式）も順序非依存

#### 決定: 生成のコアオペレーション

```
expand(node_v) → Set(neighbors)
```

特定ノードの隣接ノードをセットとして同時生成する。これが基本操作。

- 展開順序: 意味を持つため保持（ダイナミクスの記録）
- セット内部: 順序なし（set prediction, Hungarian matching等）
- 部分修正: 問題ノードまで遡って再展開（既存のOp history/undo機構と対応）

#### 決定: 全体アーキテクチャ構成

```
[生成]  expand(v) → Set(neighbors)   ← 階層展開 + set prediction
          ↓ 構築過程 = ダイナミクス
[完成グラフ]
          ↓ Graph Transformer (構造的PE, no absolute PE)
[エンコーダ出力]  node embeddings (permutation equivariant)
          ↓ cross-attention
[LLM]
```

生成と理解（エンコーディング）で異なる不変性要件を持ち、それぞれに適切な機構で対処する。

**将来展望**: CFGのMotifグラフだけでなく、抽象DAG（アルゴリズム選択、概念の連鎖）にも適用予定。Graph TransformerエンコーダをLLMに統合し、構造化された推論をサポートする。

### 外部レビュー: GPT-5.5 (xhigh effort)

アーキテクチャ設計に対しCodex CLI経由でGPT-5.5のレビューを実施。以下は主要な指摘。

#### 1. ターゲット分布: p(G) vs p(G,π)

生成過程の順序バイアスを「歓迎」とする判断は概念的に正しいが、step順序がCFGの意味論ではなくアルゴリズムの決定的tie-breaking（`(weight, node_id)`による選択、ソート済みpredecessor/successorリスト）を反映している可能性がある。Chen et al. (ICML 2021) "Order Matters" によれば、単一の構築トレースで訓練すると p(G,π) を学習し、より多くの有効な構築順序を持つグラフに不適切な確率質量が割り当てられるリスクがある。

**対策**: 完成グラフエンコーダにはstepを絶対に露出させないこと。ターゲット分布が軌道上か完成グラフ上かを明示的に決定すること。

#### 2. expand(v) → Set(neighbors) の不完全性

現在の定式化は基本操作としては有用だが、完全な生成文法ではない。以下の拡張が必要:

- **Diamond問題**: B,Cが同一Merge(D)に接続する必要があるが、独立したneighbor setではD-likeノードが2つ生成されうる → `CREATE` vs `ATTACH_EXISTING` の区別が必須
- **Loop Motif**: `node=None`, SCCコンテナ, 再帰的children → `OPEN_SUBGRAPH` / `CLOSE` アクション
- **Motif種別の遅延決定**: mergeは最終predecessor数で定義されるが、最初の親がノードを作る時点では種別不明
- **完全な生成規則**: `expand(v, partial_graph) → {new nodes, attachments to existing nodes, typed edges, control actions}`

**対策**: アクション型 `{CREATE, ATTACH_EXISTING, OPEN_SUBGRAPH, CLOSE, STOP}` + topological/frontier policyの導入。

#### 3. 位置符号化のDAG固有の問題

- **RWSEの致命的欠陥**: self-loopなしDAGでは遷移行列Pが厳密上三角 → `diag(P^k) = 0` で完全に機能しない
- **LapPE**: 対称化で方向消失 + 固有ベクトルのsign/basis曖昧性 → SignNet/BasisNet (Lim et al., ICLR 2023) が必要
- **推奨**: typed node embeddings + directed shortest-path/reachability attention biases (Graphormer方式) + sign/basis-safe LapPE on symmetrized skeleton

20ノード以下では全ペアの最短路/到達可能性バイアスの計算コストは無視可能。

#### 4. Graph Transformer重み共有のリスク

アーキテクチャ実装の共有は合理的だが、パラメータのデフォルトは分離すべき。

- Generator: 部分グラフ + 展開ノード + 順序依存な履歴を処理
- Encoder: 完成グラフを双方向に処理、順序情報を排除すべき
- 共有重みはschedule情報がエンコーダに漏れる直接的な経路を作る
- 部分グラフ ↔ 完成グラフの分布シフト

**推奨**: 共有backbone + 別adapter/normalization/PE。ただし仮説であり、permutation test・alternate-schedule consistency・graph validity・downstream LLM性能で評価すべき。

#### 5. 関連研究

- **Jin et al. (ICML 2020)** "Hierarchical Generation of Molecular Graphs using Structural Motifs": 最も近い階層的アナロジー。motif選択とattachment解決の分離が `expand(v)` に欠けている設計要素。
- **Liao et al. (NeurIPS 2019)** "GRAN": ブロック単位の同時生成。同時出力には同時分布と順序の明示的扱いが必要。
- **Zhang et al. (NeurIPS 2019)** "D-VAE": DAG-awareな非同期メッセージパッシング。方向/トポロジーの尊重の重要性。
- **Chen et al. (ICML 2021)** "Order Matters": 生成スケジュールの確率的扱いと順序周辺化の必要性。
- **GraphGPS, Graphormer, SignNet/BasisNet**: エンコーダ側の構造的特徴量設計。

#### 6. 分岐履歴の基盤問題

現在のundo/redoは線形履歴（新Op実行でredo suffixが消失）。部分修正（backtrack and re-expand）にはnode-addressable backtracking（分岐履歴/tree of executions）が必要で、engine.pyの履歴構造の変更を伴う。

### 論点の依存関係

```
A. ターゲット分布 p(G) vs p(G,π)        ← 全設計の前提
├── B. 生成文法設計 (production rule)
│   ├── D. Motif妥当性制約
│   ├── E. 重み共有戦略 (B+Cに依存)
│   └── F. 分岐履歴の基盤設計
└── C. エンコーダPE選択
    └── E. 重み共有戦略 (B+Cに依存)
```

### 未決事項（更新）

- [ ] A. ターゲット分布の決定: p(G) vs p(G,π)
- [ ] B. 生成文法設計: expand(v) → production rule（アクション型、co-reference）
- [ ] C. エンコーダ位置符号化の選択（directed SP bias + sign-safe LapPE）
- [ ] D. Motif文法の妥当性制約（CFG固有 → 抽象DAG一般化）
- [ ] E. 重み共有戦略: Generator vs Encoder
- [ ] F. 分岐履歴の基盤設計（線形 → tree of executions）
- [ ] データ生成パイプラインの仕様
- [ ] Motif語彙の粒度（4種で十分か、サブタイプを設けるか）

---
