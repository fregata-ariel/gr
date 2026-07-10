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

## 2026-07-02 セッション4

### 決定: A. ターゲット分布 = p(メタグラフDAG)

GPT-5.5レビューの p(G) vs p(G,π) 二択には第三の選択肢がある: **構築順序 π がグラフの決定的関数 π = f(G)（正準順序）なら、単一トレース学習は正確に p(G) の学習になる**。Chen et al. の質量リークは「多数の有効順序から1つをサンプル」する場合の問題で、グラフ⇔トレースが1対1なら発生しない（GraphGen の正準DFSコードと同じ理屈）。

**gr のトレースは決定的だが、決定性の由来が node_id 依存**:
- ヒープの `(weight, node_id)` タイブレーク
- `_identify_header` の `min(external_entries)`
- Tarjan の `sorted(node_ids)` 起点順

→ 同型グラフでもラベル付け替えで別トレースになり、厳密な正準性はない。

**ただしメタグラフが既にノイズの大半を吸収している**: 等重みターミナルの除去順が入れ替わっても各Motifの `preds`/`succs` は不変で、変わるのは `step` 番号のみ。メタグラフのエッジは依存関係ベースなので **step列が入れ替わってもメタグラフDAGは同型のまま**。step列はDAGの線形拡張の1つにすぎず、メタグラフはタイブレークノイズを商で潰した表現になっている。

**決定**:
- ターゲット分布は **p(メタグラフDAG)**。step番号は損失にもエンコーダにも一切露出させない
- 生成順序はDAGの層順（トポロジカル層ごとのフロンティア展開）
- 「構築順序のバイアスは歓迎」（セッション3）はマクロ構造（SCCの処理順、Kahn順の流れ）について正しく、GPT-5.5の警告はミクロなタイブレークについて正しい — 両者は分離可能

### 議論: header選択の残る穴と棚上げ

タイブレークが「順序」でなく「構造」に漏れる唯一の箇所が `_identify_header`。ただし:

- **headerの曖昧性は不可約ループ（多エントリ）でしか発生しない**。可約ループなら外部入口は定義上1つで `min()` は自明に正準
- 実CFGでは不可約ループは稀（goto由来等）。calisp由来コーパスはほぼ可約、合成データは生成側で制御可能

発生時の正準化案（将来用）:
1. 外部入辺数最大 → 2. SCC内入次数最大 → 3. WLハッシュ → 4. 残る同点は自己同型なので任意で可

ラベルランダム化による周辺化は不採用（header箇所だけ p(G,π) に戻すことになる）。別案として、多エントリループを **Motif語彙のサブタイプ**（meta に全エントリ記録）にする道もある — Motif語彙粒度の議論と接続。

**決定: PoCでは可約ループのみ扱い、header問題は棚上げ**（min-idフォールバックのまま）。

### 議論: 情報の薄まりと分配（層順生成の損失設計）

「枝葉になるにつれ情報量が薄まる」懸念は2つの別問題に分離:

1. **損失配分**: ノード単位の一様損失では、数の多い末端の低情報な決定が損失を支配し、序盤の高情報な決定（ループネスト、分岐骨格）が軽視される → 層ごとの損失正規化 or エントロピー加重で対処。階層メタグラフはLoop内部を別レベルに再帰させるため「深さ」がレベルごとにリセットされ、薄まりが1レベル内に閉じる
2. **サイズ分布**: フロンティア展開型生成に共通の罠。STOP判定の校正を誤ると小さいグラフに質量が偏る（STOP確率のわずかな過大評価が指数的に効く）→ 訓練/生成のサイズ周辺分布の突き合わせを評価指標に最初から組み込む

いずれも設計変更ではなく「損失設計と評価指標に組み込む」事項として生成文法仕様に含める。

### 決定: B. 生成文法 — 「子が親を選ぶ」への反転

GPT-5.5指摘の諸問題（Diamond問題、CREATE/ATTACH区別、種別の遅延決定）は「親が子を前向きに提案する」`expand(v)` フレームの産物。ターゲットを p(メタグラフDAG) に決めた今、基本操作を反転できる:

**新Motifを1つ追加し、既存Motifへの接続（メタグラフ入エッジ）をポインタで宣言する。**

- **Diamond問題は構造的に消滅**: Merge(D) は1回だけ生成され B, C をポインタ参照。CREATE と ATTACH_EXISTING は単一のアトミック操作に融合
- **種別の遅延決定も消滅**: 全親集合と同時に生成されるため、非Loop種別は親数の派生量（entry=親0 / linear=親1 / merge=親2+）。予測すべきは **{node, loop}** の2値 + 接続集合のみ

**文法（PoC版）** — 1レベル分のトークン列:

```
ADD_NODE(parents = {p₁, ..., pₖ})   # kind は k から派生
ADD_LOOP(parents = {p₁, ..., pₖ})   # 外部pred接続
  OPEN ... 子レベルに同じ文法を再帰 ... CLOSE
STOP                                 # このレベル終了
```

**妥当性制約（未決事項Dの最小版）**:

| 制約 | 根拠 |
|---|---|
| parents は既存Motifのみ参照（後方参照） | DAG保証が構文レベルで自動成立 |
| 各レベル先頭は entry（親0のADD_NODE）ちょうど1つ | 単一ソースCFG / Loop header唯一性 |
| ADD_LOOP は親1以上、OPEN/CLOSE 対応 | 可約ループ前提（header棚上げの帰結） |

### 決定: 正準線形化

1 Motifずつの自己回帰生成に必要な線形順序は、**メタグラフ構造自体から決定的に計算**（ラベル非依存）すれば グラフ⇔系列 が1対1のままで、学習対象は正確に p(メタグラフ):

- 大域順序: トポロジカル層順
- 層内順序: (親インデックス集合のソート列, kind) の辞書式 — 親は生成済みで正準インデックスを持つ
- 残る同点兄弟（DiamondのB, C等）は自己同型なので任意で可

GRAN型の層単位セット予測（Hungarian matching）は理論的により忠実だがPoCには過剰。精度頭打ち時の拡張オプションとする。

### 決定: PoCの段階分け

**PoC-0: 正準直列化 + 小型decoder-only LM**
- 語彙: `{ADD_NODE, ADD_LOOP, OPEN, CLOSE, STOP, ptr_0..ptr_N}`（20ノード以下なら絶対インデックストークンで十分）
- 子レベルは **インライン展開**（OPEN/CLOSEで1本のトークン列に埋め込み）— 素のLMがそのまま食える実装最速形
- パイプライン: ローカルで gr → メタグラフ → JSONL、Colabで学習（セッション1の分業どおり)
- 検証対象: データパイプライン一式、正準化の正しさ、文法妥当率、サイズ分布の校正（STOP偏り)

**PoC-1: バックボーンをグラフ条件付けに交換**
- 部分メタグラフへのattention bias（directed reachability等、Graphormer方式）
- 子レベルの独立系列化を検討（階層メタグラフの再帰構造に忠実)
- **アクション語彙とデータフォーマットはPoC-0と共通** — 文法をモデルから分離して設計したことでバックボーン比較が公平にできる

**評価指標（最初から組み込む）**:
1. 文法妥当率（パース成功 + 制約充足）
2. 生成分布の構造統計 vs 訓練分布（Motif種別ヒストグラム、層幅、ループネスト深度）
3. サイズ周辺分布の一致
4. 訓練集合とのメタグラフ同型照合（丸暗記チェック）

### 未決事項（更新）

- [x] A. ターゲット分布の決定 → **p(メタグラフDAG)**、step非露出、正準線形化
- [x] B. 生成文法設計 → 「子が親を選ぶ」ADD_NODE/ADD_LOOP/OPEN/CLOSE/STOP + ポインタ参照
- [ ] C. エンコーダ位置符号化の選択（directed SP bias + sign-safe LapPE）— PoC-1で着手
- [ ] D. Motif文法の妥当性制約 — 最小版は文法制約表として確定、完全版（抽象DAG一般化）は未着手
- [ ] E. 重み共有戦略: Generator vs Encoder
- [ ] F. 分岐履歴の基盤設計（線形 → tree of executions）
- [ ] データ生成パイプラインの仕様 — PoC-0のJSONL直列化仕様として次回具体化
- [ ] Motif語彙の粒度 — 多エントリループのサブタイプ化案が浮上（header問題と接続）
- [ ] header正準化（棚上げ中）: 不可約ループ対応時に構造的タイブレークカスケード or サブタイプ化

---

## 2026-07-04〜06 セッション5

### PoC-0 データパイプライン実装完了（TDD × Codex GPT-5.4 high）

開発プロトコルを確立: **t-wada式TDD** — 仕様書（`docs/plans/`）→ テスト作成（Codex）→
テストレビュー（Claudeセッション）→ 実装（Codex）→ 実装レビュー+検証（Claudeセッション）。

成果物（コミット a32f3d4）:
- `cfg_reducer/gen.py` — `build_cfg()` を main.py から移動
- `cfg_reducer/linearize.py` — `encode` / `decode` / `skeleton_of` / `canonical_order`
- `cfg_reducer/types.py` — `Skeleton`（構造のみのdecode結果）
- `scripts/gen_corpus.py` — 掃引・正準性セルフチェック・dedup・JSONL出力
- テスト: 正準線形化の性質（step非依存、relabel不変、往復、decode拒否6種）

**文法の設計修正**: セッション4の「kindは親数から派生」はSCC縮約で親がデデュープされる
ケース（predsが同一SCC内のMerge等）で破綻 → **kind明示トークン**
`ADD_ENTRY / ADD_LINEAR / ADD_MERGE / ADD_LOOP` を採用。

**正準性セルフチェック**: gen_corpus はノードID順列を変えて再パイプライン実行し、
トークン列不一致なら破棄・計数。ラベル依存正準形をアルゴリズム的検出なしで全捕捉する。
正準直列化の配当: 同型判定=文字列一致、dedup=文字列dedup、丸暗記チェック=ハッシュ照合。

### 実測1: 破棄率45.7% — 「不可約ループは稀」の前提崩壊

| 条件 | 破棄率 |
|---|---|
| 全体（n=6–20, p=0.10–0.30, 1875グラフ） | 45.7% |
| edge_prob 0.10 / 0.20 / 0.30 | 45–46%（無依存） |
| ノード数 6–8 / 12–14 / 18–20 | 20.5% / 39.2% / 73.9% |

原因: スパゲティ化の後方エッジ+前方ジャンプ（ノード数比例）が多エントリSCCを量産。
edge_prob 無依存・ノード数強相関がこの診断を裏付け。生成は毎秒4000グラフ超で計算コストは
無問題、ボトルネックは収率と多様性（小グラフはdedup飽和: n=6–8で生存分の80%が重複）。

**決定 (a)**: 後方エッジを dominator ゲートで可約に矯正（実CFG=構造化言語はほぼ可約なので
calisp分布への接近でもある）。前方ジャンプ [B] を後方エッジ [A] より先に移動
（後からのジャンプはループへの第二エントリを作る）。header正準化前倒し(b)、現状受容(c)は却下。

### 教訓: テスト契約のバグが実装を歪める

仕様書の可約性テスト定式化「サイクルを閉じる全辺が dominator を向く」は可約性より強く、
長さ2以上の全ループを禁止していた（サイクル上の両端点は相互支配不可能）。実装側 Codex は
テスト変更禁止の契約下で**ループを生成しない実装**に到達し green を達成、ただし矛盾を明示的に
報告。掃引出力444件中ループ含み0件で確定。

修正: 正しい定式化「dominator を向く辺（back edge）を全除去した残りが非巡回」+
**ループ非空性テスト**（全グラフがサイクルを最低1つ含む）で「ループ全滅による可約性達成」を
封じた。TDDの教訓: テストは契約であると同時に、誤った契約は静かに設計を破壊する。
実装者が矛盾を報告できるプロンプト契約（「テストと仕様が矛盾したら変更せず報告」）が機能した。

### 実測2: 可約化後も41% — 第二の穴「エントリ含みSCC」

可約化での改善は僅か（45.7%→41.0%）。原因: dominator ゲートは N00（全ノードを支配）を
常に後方エッジ候補にするため、**エントリノードを含むSCC**（プログラム全体ループ）が多発。
このSCCは外部入辺を持たず `_identify_header` が `min(scc)` フォールバック=ラベル依存に落ちる。
可約性による header 一意性は「外部エントリが存在する」ループにしか効かない。

**構造的事実**: entry ∈ SCC ⟺ SCC外部入辺なし（外部から入る辺があれば、その始点は
entry から到達可能なのでサイクルに巻き込まれSCC内に入る — 矛盾）。よって
`external_entries` と `entry ∈ scc` は相互排他。

**決定 (b)**: CFG を **(G, entry)** として扱う。`ReductionAlgorithm(engine, entry=None)` を追加、
`_identify_header` を三段フォールバック化: `min(external_entries)` → `entry ∈ scc なら entry`
→ `min(scc)`。エントリ含みループの意味論的 header はエントリ自身（実コンパイラも entry を
明示指定する）。生成側で N00 をターゲット除外する案(a')は「全体ループ」を分布から失い、
ループ非空性も壊れるため却下。

**結果**（コミット 2e5ae16）: 破棄 769→42（41%→**2.2%**）、42 passed / skip 0、
書き出し549件は全件ループ含み。

### 実測3: 残留2.2% = 層内タイブレークの穴（診断済み・未修正）

n=7, p=0.1, seed=0 で再現・確認: 同一層で（親集合, kind）が同じ2つの linear (N01, N03) が
step 順（ラベル依存）でタイブレークされるが、**両者の子孫構造は異なる**（自己同型でない）。
セッション4の「フルキー同点なら自己同型なので任意で可」は誤り — (親, kind) は未来を見ていない。

**次の一手**: `canonical_order` の層内ソートキーに WL 風構造シグネチャを追加
（kind を初期色に、DAG 双方向の反復精緻化; Loop は子スケルトンの再帰ハッシュを初期色に混入）。
WL が区別できない残留ケースは正準性セルフチェックが破棄するため、**コーパスの正しさは
現状でも保証済み** — 修正は収率を98%→100%へ押し上げる最適化。

### 未決事項（更新）

- [x] データ生成パイプラインの仕様 → 実装完了（linearize + gen_corpus、収率549/1875）
- [ ] 層内タイブレークの WL 精緻化（残留2.2%破棄の解消; 診断済み）
- [ ] PoC-0 学習実験（Colab）: 現収率で開始可能か、コーパス規模・多様性の見極め
- [ ] C. エンコーダ位置符号化（PoC-1）
- [ ] E. 重み共有戦略 / F. 分岐履歴 / Motif語彙粒度 / header正準化（不可約、据え置き）

---

## セッション6: PoC-0 学習実験（Colab T4 本走）と結果 (2026-07-08〜10)

### 実施フロー（上流工程のみ本セッション、実装は Codex 委譲）

ユーザー方針: 本セッションは上流工程（仕様策定・成果物レビュー・レビュー観点整理）に限定し、
詳細計画は GPT-5.5 xhigh、実装は GPT-5.4 high、最終レビューは GPT-5.5 xhigh に委譲。
一括委譲が失敗した場合はステップ分解に切り替える（今回は計画のみ分離、実装は5段階分割が機能）。

1. 上流仕様 `docs/plans/poc0_training_spec.md`（スタック・プラットフォーム分割・EOSなし終端・
   2〜4Mパラメータ帯・T4 30分予算・受入基準3点を固定）
2. 詳細計画 `docs/plans/poc0_training_impl.md`（GPT-5.5 xhigh、実測ベースの語彙41/PTR32/
   系列長80、3.19Mパラメータの算術、5段階委譲プラン付き）
3. Stage 1〜4 を GPT-5.4 high が TDD 実装（Claude がレビューゲート、各段階コミット）
4. 最終レビュー（GPT-5.5 xhigh、読み取り専用）: approve-with-fixes、所見3件すべて CONFIRMED
   → 修正適用（empty_stream 評価区分、空トークン列拒否、ArrayRecord 型検証）
5. Stage 5（Colab 本走）は Codex サンドボックスが colab CLI をブロックしたため Claude が直接実行

### 途中で発見・修正した問題

- **走行間非決定性（重要）**: 同一コマンドの gen_corpus が実行ごとに異なる出力
  （written=20686/20689/20690）。原因は PYTHONHASHSEED — `tarjan_scc` が
  `engine.successors()`（set）をハッシュ順で走査し SCC 列挙順が揺れ、
  `_pick_terminal_scc` の「最初の終端SCC」選択が変動。
  修正: tarjan の子ソート + 終端SCC の明示的最小選択（`a760939`）。
  以後、75kスイープが md5 単位で再現（written=20696、破棄3.2%）。
  「正準直列化 = 同型なら文字列一致」がプロセス・マシンを跨いで成立するようになった。
- **ArrayRecord API の実地バグ2件**（Linux 初実行で顕在化、レビューが警告した「探り書きの脆さ」
  そのもの）: ①トップレベル `array_record` は import 可能だが Writer を公開しない
  → クラス単位の候補探索に修正（`b5d18bc`）; ② `read()` は終端で IndexError
  → `read_all()` 優先に修正（`eb3d6e5`）。修正後スモークテスト 12 passed
  （書き込み→読み戻し→インメモリソースとバッチ完全一致）。
- Colab セッションはネットワーク断で keep-alive が死ぬと孤児化し、無料枠の GPU 1枠を
  塞ぐ（TooManyAssignmentsError）。自然回収まで約30分待ち。

### PoC-0 本走結果（T4、3000 steps / batch 256、N=1000 @ temp 1.0）

| 受入基準 | 結果 | 判定 |
|---|---|---|
| decode() 妥当率 ≥99% | **86.2%** | 未達 |
| 新規率 | **80.6%**（train重複19.4%、有効862中856ユニーク） | 健全 |
| 分布一致 TVD | **0.006〜0.112**（9統計すべて） | 良好 |

- 学習: loss 3.91→0.33、best val 0.549（step≈2000）→ final val 0.641（過学習の尻尾）
- 無効138件の内訳: **decode_error 135** / length_cap 1 / negative_depth 2 / empty_stream 0
- 成果物: `runs/poc0_t4/`（metrics.json、ヒストグラムPNG 9枚、学習ログ）

**解釈**: 分布・非暗記の2基準は明確に合格。構造分布の学習という PoC-0 の核心仮説は支持された。
妥当率は ptr 制約（昇順・重複禁止・範囲内参照）由来とみられる decode_error が 14% を占め未達。
final checkpoint（過学習後）で評価した影響が濃厚 — best checkpoint 評価が第一の改善候補。

### 次の一手（優先順）

1. **再走 + 診断強化**: best checkpoint での評価、無効サンプルのトークン列ダンプ
   （どの文法規則で落ちるかの分類）、checkpoint 重みのローカル回収（~13MB、再評価を可能に）
2. 過学習対策: エポック数削減（現状約40エポック）またはコーパス拡大（n上限拡大で
   ユニーク数はほぼ線形に増える — セッション5の飽和測定参照）
3. 妥当率が best 評価でも 99% に届かない場合: 文法制約サンプリング（ptr マスク）を
   PoC-0.5 として検討（「モデルが文法を学ぶ」原則の緩和になるため要議論）

### 未決事項（更新）

- [x] PoC-0 学習実験（Colab）→ 完走。機構は全段検証済み、妥当率のみ未達（86.2%）
- [ ] PoC-0 再走: best checkpoint 評価 + 無効サンプル診断
- [ ] 層内タイブレークの WL 精緻化（設計済み: docs/plans/wl_tiebreak_design.md、据え置き）
- [ ] C. エンコーダ位置符号化（PoC-1）
- [ ] E. 重み共有戦略 / F. 分岐履歴 / Motif語彙粒度 / header正準化（不可約、据え置き）
