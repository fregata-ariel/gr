# C 詳細設計: 生成器 v2・構造 OOD マトリクス

作成: 2026-09-09。状態: **実装完了**(タスク 1–15, 17。16 は D1 で削除)。§12 に進行役の決定。
上位仕様は [generator_v2.md](generator_v2.md) §3–§8、特に A7。A7 を変更する決定は本書では行わない。
目的は既定モデル mask の構造分布シフト耐性の測定。base と比較し、ptr は family OOD の参考のみ。
本タスクの成果物は本書だけ。以下のファイル名・API・テストは**今後の実装指定**である。

## 1. 固定する境界と配置

- Python 3.13 / uv、追加ランタイム依存なし。新しいデータ型は frozen dataclass、コレクションは tuple を優先する。
- `generate.py`、`store.py`、MetaGraph/Op の形式・縮約順序・永続 step を変更しない。`train_ar.py`、`pyproject.toml`、`uv.lock` も変更対象外。
- 現行 `GeneratorDescriptor(name, fn)` と nodes+edges による同型判定はレビュー 1-5 / 1-6 対応済み。これを再利用する。
- `handoff.md` の古い実装状況より現行コードを優先する。基準は [dataset_generation.md](dataset_generation.md)、[metagraph_schema.md](metagraph_schema.md)。
- `cfg_reducer/generator_types.py`: 本書の生成・採用用データ型。`family_registry.py`: Protocol と登録。`families/{layered,structured,spaghetti}.py`: 族。
- `generate_v2.py`: spec の JSON 変換・dispatch・descriptor アダプタ。`reducibility.py`: CFG 判定。`buckets.py`: 測定・採用。
- `cfg_reducer/structure_features.py`: 現行 `training.structure_features.features_for` と `_walk_levels` をそのまま移す。training 側は再 export し既存 import を保つ。
- `dataset.py` は族を知らない採用機構のみ追加。新 CLI `cfg_reducer/dataset_v2.py` が registry、spec、plan を接続する。
- 既存の `cfg_reducer.dataset` CLI は v1 専用のまま。新族は自身のモジュールと登録だけを追加し、dataset / controlled_eval の分岐を増やさない。

## 2. 族 plugin と型契約

以下のシグネチャ中の省略記号は本文で規定する処理を表す。新 API はすべて引数・戻り値に型を付ける。
`Json = None | bool | int | float | str | list[Json] | dict[str, Json]`、`Features = dict[str, int | float | bool]` を共通型とする。
JSON 入力は明示的な型検査で狭める。bool を int として受理しない。NaN/Infinity、未知キーは ValueError。
frozen 内の dict は入力時に再帰コピーし、公開後に変更しない。JSON 出力は tuple→list、キーを再帰 sort、`allow_nan=False`。

```python
@dataclass(frozen=True)
class GeneratorSpec:
    family: str                   # Literal に固定しない; ^[a-z][a-z0-9_]*$
    num_nodes: int
    params: dict[str, Json] = field(default_factory=dict)
@dataclass(frozen=True)
class CFGShape:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    entry: str
class CFGFamily(Protocol):
    @property
    def name(self) -> str: ...
    def normalize(self, spec: GeneratorSpec) -> GeneratorSpec: ...
    def generate(self, spec: GeneratorSpec, rng: random.Random) -> CFGShape: ...
def register_family(plugin: CFGFamily) -> None: ...
def get_family(name: str) -> CFGFamily: ...
def family_names() -> tuple[str, ...]: ...
def load_plugins(modules: tuple[str, ...]) -> None: ...
def normalize_spec(spec: GeneratorSpec) -> GeneratorSpec: ...
def spec_to_json(spec: GeneratorSpec) -> dict[str, Json]: ...
def spec_from_json(value: dict[str, Json]) -> GeneratorSpec: ...
def generate_cfg_v2(engine: GraphEngine, *, seed: int, spec: GeneratorSpec) -> list[str]: ...
def descriptor_for(spec: GeneratorSpec) -> GeneratorDescriptor: ...
```

- 上位 §3 の族固有フィールドを `params` に移す。たとえば `GeneratorSpec("structured", 24, {"target_depth": 2})`。後の族の独自キーを共通クラスに追加する必要はない。
- 組み込みは `families/__init__.py` で明示登録。外部追加は `--plugin package.module`（反復可）を指定順に import し、そのモジュールの `register_family(...)` を実行する。
- 重複名は同一オブジェクトでも ValueError、未知名も ValueError。暗黙の上書き・ディレクトリ探索はしない。差し替えは新プロセスの登録と version 更新で行う。
- `family_names()` は辞書順。plugin は環境・時刻・グローバル乱数を読まない。registry は登録後に固定し、生成中の登録は禁止する。
- normalize は既定値を埋め、族の型・範囲を検査し、冪等にする。再生成時に既定値が変わっても影響しないよう全有効パラメータを保存する。
- `descriptor_for` は `name="cfg_v2:" + spec.family`、fn は `fn(engine, *, seed: int, spec: dict[str, Json]) -> list[str]` のアダプタを返す。
- builder に渡す config は必ず `{"spec": spec_to_json(normalize_spec(spec))}`。アダプタは受け取った spec を復元し、descriptor と family の一致を検査する。
- `generate_cfg_v2` は空 engine を要求し、`Random(seed)` を一度だけ作って plugin に渡す。shape の ID 重複・不明端点・entry 存在を検査し、nodes 順、辞書順 edges で engine に反映する。
- nodes は配置順の `N00, N01, ...`、edges は重複なし。全ノードが entry から到達可能であることを族の契約とする（一般の重複 API は孤立ノード対応を維持）。
- version は既存どおり CLI の `--version` または git commit。構成アルゴリズム・乱数消費を変更した版は新 version で生成する。
- v2 の provenance は `source=synthetic, generator={name, version, seed, config}` だけ。上位 §4 の固定名案は A7-1 により族別名へ置き換える。

## 3. layered: v1 完全再現を基点にする

params は `max_layer_width=3, edge_prob=0.5, loop_count=2, goto_count=1, span_mode="uniform"`。
n≥1、width≥1、edge_prob∈[0,1]、count≥0。branch_degree / merge_degree / target_depth / spaghetti_rate はこの族では未知キーとして拒否する。
layered の次数は層間接続とジャンプの結果であり、v1 再現時に次数制限を追加してはいけない。degree 制御の実験には structured を使う。

```text
ids を num_nodes 個作る; n<3 は鎖を返す（乱数消費なし）
current=[ids[0]]; remaining=ids[1:-1]
while remaining:
    weights=[1,1,2,2,...,width-1,width-1,width]  # width=1 は [1]
    size=min(len(remaining), rng.choice(weights)); next=先頭 size 個
    current の各 u に rng.choice(next) を接続
    next の各 v に親がなければ rng.choice(current) を接続
    current×next をリスト順に走査し、rng.random()<edge_prob なら未接続辺を追加
    current=next
current の全ノードから EXIT へ接続
loop_count 回: j=rng.randint(2,n-2); i=rng.randint(0,j-2); 未接続なら ids[j]→ids[i]
goto_count 回: i=rng.randint(1,n-3); j=choose_target(i+2..n-1, span_mode); 未接続なら ids[i]→ids[j]
```

- 重複辺になっても再抽選しない。count は**辺追加の試行数**。乱数判定を「未接続か」の後に移すと v1 の列が変わるので禁止。
- uniform の choose_target は `rng.randint(lo, hi)` そのもの。short / long は候補を距離昇順にし、それぞれ最短 / 最長 `max(1, ceil(len/3))` 個から `rng.choice`。
- 回帰 spec は width=3、edge_prob は v1 に渡す値、`loop_count=max(1,int(.15*n))`、`goto_count=max(1,int(.1*n))`、uniform。
- 上位 §3 の `int(...)` 表記は n12–48 では一致するが、小さい n の v1 は `max(1, ...)`。n=3 は v1 が randint で ValueError を出すため、v2 も正 count なら同じく失敗させる。
- n=0 は entry を持てないため v2 では拒否。n=1,2 と n≥4 が回帰対象。n=3 は両 count=0 のとき DAG のみ許す。v1 自体は修正しない。
- 返却 ID、nodes、edges、縮約後の canonical metagraph 部分が v1 と一致することを検査する。v2 は名前/config が違うので UUID の一致は要求しない。

## 4. structured: 再帰テンプレートと予算

params は `max_layer_width=3, branch_degree=2, merge_degree=3, loop_count=2, goto_count=1, span_mode="uniform", target_depth=null`。
width≥1、branch_degree≥2、merge_degree≥2、count≥0、n≥3。width は並列 arm 数の上限で、CFG 全体の幅ではない。
有効 arm 上限 `b=min(width,branch_degree)`。b=1 では条件分岐・loop・goto を要求できず直列のみ。spaghetti_rate は structured では拒否する。
loop_count は構文数、target_depth は構文上の最大 loop 入れ子深さ（0 起点）。L=0 なら D=0、L>0 なら 1≤D≤L。
D=null は L=0 なら0、それ以外は `rng.randint(1,min(L,3))`。実測 `n_loops/max_depth` は縮約器の特徴であり一致を仮定しない。
goto_count は break / continue / return の**合計構文数**とする。一般 goto は入れない。L=0 の場合は return のみ。

### 4.1 内部 AST と配線

`families/structured.py` 内の frozen `Stmt(kind: str, children: tuple[Stmt,...]=())` を使用する。
kind は atom / seq / if / if_else / switch / while / do_while / break / continue / return。種類ごとの子数を検査する。
内部 planner は次の補助 API に分ける（外部 export 不要）。

```python
def plan_structure(spec: GeneratorSpec, rng: random.Random) -> Stmt: ...
def lower_structure(tree: Stmt, merge_degree: int) -> CFGShape: ...
def pad_shape(shape: CFGShape, target: int, mode: str, rng: random.Random) -> CFGShape: ...
```

lower は先に entry/exit の記号を予約し、AST を前順で訪問して記号ノードと辺を作る。最後に ENTRY を先頭、EXIT を末尾として連番を与える。
各断片は入口と通常出口リストを持つ。loop context stack は `(header, continue_port, break_port)`、関数 context は EXIT を持つ。

| 構文 | 接続と最小の構成 |
|---|---|
| atom / seq | atom は1ノード。seq は前の通常出口から次の入口へ接続。空 seq は atom に正規化 |
| if | 条件 T→arm 入口 / join J、arm 通常出口→J。T、J と非空 arm |
| if_else / switch(k) | T→各非空 arm 入口、各通常出口→J。if_else は2 arm、switch は3≤k≤b、fallthrough なし |
| while | H→body / X、body 通常出口→T、T→H。continue_port=T、break_port=X。H,T,X と非空 body |
| do_while | H→body、body 通常出口→T、T→H / X。continue_port=T（条件判定）、break_port=X。同じノード数 |
| break / continue / return | 専用1ノード→最内 loop の X / T / 関数 EXIT。通常出口なし。while の T は副作用のない header 中継 |

- abrupt 文は通常到達可能な atom の直前に `if(abrupt)` として挿入する。ただし専用 join を共有して「条件→abrupt / 元 atom」に下げるため追加2ノード。通常 arm を必ず残す。
- これにより loop の通常経路・back edge と後続文が残り、return を選んでも未到達ノードが生じない。switch 内 break は使わず、break は常に loop に属する。
- 全配線を予約後、入次数 m=merge_degree を超える宛先について、**辺を削除せず**合流中継木を挿入する。
- `join(sources,dst)`: sources を記号作成順で sort。数が m を超える間、先頭 m 個→新中継 J とし、それらを J に置換。残り（≤m）→dst。重複 source は先に除く。
- H は外部前駆と T の2系統だけを持つ。continue は T の前、break は X の前、return は EXIT の前で合流させる。入口を body 内部へ迂回させない。
- 通常の if/switch 合流、loop 出口、EXIT を含め**全 CFG ノード**の入次数≤m。中継木はノード予算に含む。MetaGraph の max_in_degree≤m とは限らない。
- 構成的保証: 直列・分岐は新しい cycle を作らない。loop 内部への外部入口は H のみ。continue は同じ loop 内、break/return は外向きのみ。中継木・辺分割もこの性質を保存する。

### 4.2 有限の生成手順

```text
lo=ceil(.9*n); hi=floor(1.1*n); B=rng.randint(lo,hi)
D を決める; atom を D 重の loop で包む（各種別は [while,do_while] から choice）
残り L-D 個の loop(atom) は最上位 seq の後ろへ順に追加（深さを増やさない）
G 回: 前順の atom 候補から choice; context 内で合法な [break,continue,return] から choice
      選んだ atom の前に条件付き abrupt を挿入
shape=lower(tree,m)   # 合流中継木込みの必要ノード数を正確に数える
len(shape)>hi なら GenerationRejected("node_budget")、B 未満に縮めるため構文を捨てない
B=max(B,len(shape))
K=rng.randint(0,(B-len(shape))//2); 最大 K 回、atom→if(atom), if_else(atom,atom), switch(k atoms) の候補を列挙
候補順は (atom 前順番号, 上記 kind 順, k 昇順); lower 後のサイズ≤B の候補だけから choice
候補なしで終了; 各反復はサイズを増やす; loop/abrupt 数・深さは変えない（残予算は padding に回す）
pad_shape で残差分だけ通常の辺を分割; len(shape)=B を検証して返す
```

- 明らかに不可能な n、D>L、b=1 で制御構文要求、下限 `3+3*L+2*G>hi` は生成前 ValueError。下限通過は実現可能性の証明ではなく、実際の lower 超過は seed ごとの棄却。
- `GenerationRejected(reason: str)` は生成試行の既知の失敗専用例外。builder はこれだけを棄却として捕捉する。バグ、型違反、reducible 保証違反は中断する。
- 任意の構文を消してからノード数を合わせない。指定 L/G/D を守った成功 CFG のみを返し、成功時は整数範囲 lo≤N≤hi を保証する。
- span は CFG の予約順で測る前向き辺長への誘導であり、MetaGraph offset の保証ではない。下記の padding 選択だけに作用させる（構文数は固定）。
- pad_shape は AST 属性を使わず、nodes の位置順から前向き辺を候補とする。測定対象の span 辺は「src の出次数≥2 または位置差≥2」の前向き辺と定義する。
- 「またがる」は分割辺の宛先位置が span 辺 (u,v) の開区間内にあること。各分割で位置を更新し、short は個数の最小群、long は最大群、uniform は全候補から choice。同点候補は (src,dst) 順。
- 挿入位置は分割辺の宛先直前。候補がない場合は ENTRY の通常辺を使用する。ノード追加は辺分割だけなので reducibility と入出次数を保存する。
- ±10% を許容するが分布の均一性は保証しない。mean_offset の short<long も保証条件にせず、pilot の分布表で誘導効果を検証する。

## 5. spaghetti と共通の決定性

params は structured の全項目 + `spaghetti_rate=0.1`（有限、0≤rate≤1）。

```text
base=structured.generate(structured 用に rate を除いた spec, rng)
M=floor(rate*len(base.edges))
candidates=sorted((u,v) for u in nodes for v in nodes
                  if u!=v and (u,v) not in edges and v!=ENTRY and u!=EXIT)
extra=rng.sample(candidates,min(M,len(candidates)))
return 同じ nodes/entry, sorted(base.edges ∪ extra)
```

rate は基底の辺数に対する追加辺比率。forward/backward をまとめた一様な非復元抽出で、irreducible を強制しない。
rate=0 は同 seed の structured と同一 CFG。追加後は branch/merge の上限保証を外すが、ノード予算と entry 到達性は維持する。
全族で乱数源は渡された Random のみ。set は membership にのみ利用し、choice/sample/Tarjan/JSON に渡す順序は必ず固定する。
`tarjan_scc` の既存コメントどおり文字列 set の順序は PYTHONHASHSEED で変わる。seed 固定だけでは決定性テストにならない。

## 6. reducibility 判定

公開 API: `is_reducible(nodes: tuple[str,...], edges: tuple[tuple[str,str],...], *, entry: str) -> bool`。
空 CFG、entry 不在、不明端点、entry から未到達のノードは ValueError（「未到達を除いて true」にはしない）。自己ループは許す。
多重入口は「外から入る**辺数**」でなく「入口**頂点数**」。単一入口の最大 SCC だけ見ても内部の irreducible cycle を見落とす。

```text
pred は元 CFG 全体の predecessor 集合; stack=[全 nodes]
while stack:
    region=pop(); tarjan_scc(region, 元 CFG succ) を求め SCC を sort(tuple(sorted(C))) 順に処理
    for cyclic C (|C|>1 または自己辺あり):
        entries={v∈C | v==entry または pred[v] に C 外の頂点あり}
        if len(entries)!=1: return False
        h=唯一の入口; stack に C-{h} を追加（空なら省略）
return True
```

各再帰段でも pred は元 CFG のまま。除去済み header からの入口を忘れてはいけない。各段で頂点が減るため停止し、最悪 O(V(V+E)) で n≤48 には十分。
テスト fixture: DAG/diamond=true、自己ループ=true、単一入口の nested loops=true、外部2辺が同じ header に入る=true。
多重入口例 `s→a,s→b,a→b,b→a` は false。
隠れた内部多重入口例 `s→h,h→a,h→b,a→b,b→a,a→h` は false（外側 SCC の入口 h は一つ）。
到達不能孤立点は ValueError、全 fixture の辺・ノード挿入順を逆転しても同じ結果を要求する。

## 7. realized・BucketPlan・採用フック

```python
@dataclass(frozen=True)
class Candidate:
    split: str; seed: int; sample_id: str; requested: dict[str, Json]
    nodes: tuple[str, ...]; edges: tuple[tuple[str, str], ...]; mg: MetaGraph
@dataclass(frozen=True)
class BucketDimension:
    feature: str; cuts: tuple[float, ...]; labels: tuple[str, ...]
@dataclass(frozen=True)
class BucketPlan:
    dims: tuple[BucketDimension, ...]
    target_per_bucket: int
    active: tuple[tuple[str, ...], ...] | None = None
@dataclass(frozen=True)
class AcceptanceState:
    counts: tuple[tuple[str, int], ...]    # 現 split の bucket→採用数
@dataclass(frozen=True)
class AcceptDecision:
    accepted: bool; reason: str | None
    bucket: str | None; realized: Features
AcceptHook = Callable[[Candidate, AcceptanceState], AcceptDecision]
def measure_realized(candidate: Candidate) -> Features: ...
def bucket_for(features: Features, plan: BucketPlan) -> str: ...
def bucket_acceptor(plan: BucketPlan) -> AcceptHook: ...
def measurement_acceptor(candidate: Candidate, state: AcceptanceState) -> AcceptDecision: ...
```

- 測定は `features_for(mg)` の既存23特徴 + `num_nodes=len(nodes)` + `reducible=is_reducible(..., entry=nodes[0])`。CFG snapshot は縮約前、MetaGraph 測定は縮約後。
- 判定は plugin 名によらない。追加族にも nodes[0]=entry 契約を要求する。requested は正規化済み config のコピー、realized はこの25特徴。両者を混同しない。
- 既定 dims 順: mean_offset cuts=(1.68,2.03), labels=(low,mid,high); max_depth cuts=(1,2), labels=(0-1,2,3+); max_in_degree cuts=(2,3), labels=(le2,3,ge4); reducible cuts=(), labels=(no,yes)。
- 数値 bin は右閉区間: x≤c0 / c0<x≤c1 / c1<x。reducible だけ bool をラベルに直結する。境界は再推定・丸めしない。定義元の「三分位」と四分位説明の不整合にかかわらず A7 の数値を固定する。
- bucket ID は `mean_offset=low|max_depth=2|max_in_degree=3|reducible=yes` の形式。feature/label に `|` と `=` を禁止する。
- cuts は有限な厳密昇順、labels は一意、数値では len(labels)=len(cuts)+1、feature は測定キーに存在すること。次元の重複も拒否する。
- 全直積は54 bucket。active=null は全54、active 指定時はその集合だけを目標とする。active の重複・未知ラベル・空集合、target≤0 は ValueError。
- active 外は `outside_plan`、充足済みは `bucket_full`、未充足なら採用。欠測・非有限は `invalid_feature`。採用数を増やすのは builder の保存成功後のみ。フックに可変カウンタを隠さない。
- structured に全54を要求しても no 側は埋まらない。structured 用は yes 側を明示選択する。実験に使う active は pilot 後、test 採点前に固定し、空 bucket を結果を見て消さない。
- target は全 split 共通の固定数。異なる train/val/test 件数には別 plan を渡す（builder の accept を split→hook に dispatch する CLI クロージャ）。測定のみなら measurement_acceptor。

### 7.1 build_dataset の拡張順序

既存6引数の意味を保ち、末尾に keyword-only `accept: AcceptHook | None = None, exclude: tuple[CFGReference,...] = ()` を追加する。
`exclude` は accept と併用必須（省略なら ValueError）。quota 不要なら measurement_acceptor を渡す。
`CFGReference` は frozen `(dataset_id: str, sample_id: str, nodes: tuple[str,...], edges: tuple[tuple[str,str],...])`。

```text
split は従来の挿入順、seed は [start,stop) 昇順。全範囲を処理し、充足後も bucket_full を記録
generate → CFG snapshot → WL 候補と有向同型照合（exclude が先、次に全 split の採用済み seen）
accept なし: 従来どおり重複なら即 drop、非重複なら reduce/save
accept あり: 重複でも reduce して測定し、accept(candidate, state) の bucket/realized を取得
重複なら accept の可否より優先して棄却; それ以外は decision に従う
採用なら save_sample → seen 登録 → split 内 counts 増加; 棄却は seen/counts に入れない
```

`GenerationRejected` の場合も attempts に数え、bucket/realized=null。同型による棄却の bucket は測定できるので記録する。
有限 seed range が試行上限であり勝手に seed を延長しない。未充足は `complete=false` と bucket ごとの不足数を記録する。
新 CLI は manifest を保存したうえで未充足なら終了コード2。`--allow-incomplete` のみ0に変更する。例外を握りつぶして成功扱いにしない。

### 7.2 manifest の加算拡張と同一性

accept/exclude 未指定の v1 呼び出しでは既存 manifest・sample を**バイト同一**に保つ。既存 schema_version=1 と sample envelope は変更しない。
新経路だけ `selection={version:1, plans:{split:...}, excluded_datasets:[id,...]}` を追加。quota なし測定経路の plan は null。root にも attempts/accepted/rejected の split 合計を置く。
split の旧 `seed_range, kept, dropped_duplicates, samples, dropped` を残し、次を追加する。

```text
attempts: int; accepted: int; rejected: int; acceptance_rate: float
complete: bool; per_bucket: {bucket: {target: int|null, attempts: int, accepted: int, rejected: int, missing: int|null}}
rejections: [{seed, reason, bucket, realized, duplicate_of?, duplicate_dataset?}]
samples: [{seed, sample_id, requested: {spec: {...}}, realized: {...}, bucket: str|null}]
```

`attempts=accepted+rejected=stop-start`、accepted=kept。rejections の理由は duplicate / cross_dataset_duplicate / node_budget / outside_plan / bucket_full / invalid_feature。
旧 dropped と dropped_duplicates は同型棄却だけを表す（cross_dataset_duplicate も含む）。旧 `kept+dropped_duplicates=試行数` は新経路では成立しない。
per_bucket は全 active をゼロ件でも列挙し、対象外の測定済み bucket も target=null で記録する。bucket 不明試行は別の unbucketed 件数へ集計する。
key は sort、採用・棄却の配列は seed 順。タイムスタンプ・絶対パスを保存せず、code を固定した再実行は全 JSON がバイト同一。
UUID は今の `store.sample_id_for(provenance)` をそのまま使う。requested/realized、bucket/plan、split、commit/dirty の追加情報を provenance に足さない。
旧 manifest の読み手は追加キーを無視し、旧 manifest を読む新コードは selection/realized 不在を許す。`--by-bucket` を旧データに使うときは欠測を明示する（yes/no を推測しない）。

## 8. 別 dataset を用いる OOD の配線

### 8.1 生成・重複・tokenize

新 CLI: `uv run python -m cfg_reducer.dataset_v2 --spec spec.json --plan plans.json --out DIR --split train=0:100000 --version VERSION`。
`--plan` は省略可、内容は `{split: BucketPlanのJSON}`。`--plugin MODULE`、`--exclude-dataset DIR` は反復可。spec と旧 `--num-nodes` 等の二重指定は導入しない。
`load_references(paths: tuple[Path,...], *, version: str) -> tuple[CFGReference,...]` を dataset_v2 に置く。manifest の採用 sample を provenance から再生成して snapshot する。
v1 名は DEFAULT_GENERATOR、`cfg_v2:*` は registry アダプタで解決。未知名は停止。dataset_id は manifest bytes の SHA256。参照した manifest hash を selection に記録する。
load_references は呼び出し側 version と manifest.generator.version の一致、および記録 code.commit と実行 checkout の一致を検証する。不明/不一致なら停止する。
異なる版は対応 checkout で生成した CFGReference の JSON 配列（dataclass の全4フィールド）を `--exclude-cfg FILE` で読み込む。配列の content hash も selection に記録し、現在の plugin で黙って再生成しない。
OOD test 作成時は source train/val dataset を exclude。別名・別 UUID でも同型なら除く。複数 train 族に同じ test を使うなら全 train/val の和集合を除く。

```python
def prepare(dataset_dir: str | Path, out_dir: str | Path,
            window_from: str | None = None, *,
            test_dataset: str | Path | None = None) -> dict: ...
```

`training.prepare_tokens` に `--test-dataset DIR` を追加する。同時指定時は `--window-from train` を必須にし、source の train/val と target の test だけを出力する。
source train は非空必須。source test は置換され、target train/val は読まない。train/val/test の JSONL 形式は既存のまま、UUID を付け替えない。
語彙は source train の `max_offset_needed` の最大（最低1）、`meta.max_len` も source train の token 長最大に固定する。test でモデル形状を広げない。
現 trainer は `gen_max_len` と meta.max_len の大きい方をモデル容量に使う。C では `--gen-max-len=2*train_max_len` を事前固定し、val/test はこの容量超過も除外する。
既存単一 dataset 呼び出しの max_len 計算は互換維持。新 OOD 経路の meta に `max_len_source=train, sequence_capacity=2*max_len` を明記する。
窓超過を先に判定し `over_window`、次に token 長>capacity を `over_length` とする。既存 excluded_over_window を保ち、全除外に sample_id/seed/bucket/needed を追加した sidecar を出す。
`evaluation_index.json` は `{version:1, sources:{split:manifest_sha256}, selection:target_selection, samples:{sample_id:{split,bucket,realized}}, exclusions:[...]}`。
index は採用された全 test と除外を覆い、ID 重複は ValueError。旧 target の bucket 不在は null（集計では __unmeasured__ 群、WF は取得不可）。source の family を推測せず target manifest の値を引き継ぐ。
meta に source hashes、raw/retained/excluded の全体・bucket 別件数を保存。除外0も記録し、比較する全構成で同じ token bundle を使う。
コマンド例: `uv run python -m training.prepare_tokens --dataset data/layered --test-dataset data/structured --window-from train --out data/tokens_family_ood`。
Colab は既存 `--train/--val/--test/--vocab/--meta` にこの bundle を渡す。B の train 頻度 baseline、dev 早期停止を維持し test をチューニングに使用しない。

### 8.2 controlled_eval の bucket 別 NLL

`summarize` の既存引数末尾に `*, by_bucket: bool=False` を追加し、CLI は `--by-bucket`。index は `--tokens` 内から読む。
公開 `bucket_scores(scores: list[dict], index: dict) -> dict[str, dict]` は sample_id で join し、族の名前には分岐しない。
`--by-bucket` 時は index 必須、token/test/score の ID 重複・未知 ID・score 欠落をエラーにする。構成間で全 test ID の一致を検証してから paired_delta を呼ぶ。
各 bucket に n_samples/n_tokens、`sum(nll)/sum(n_tokens)`、sample 平均 NLL、その bootstrap 95% CI、REF NLL/edge accuracy を出す。
構成−base の paired ΔNLL は同一 seed・同一 bucket の sample 対応。sample_id sort 後、既存 bootstrap(seed=0) を使う。空 bucket は件数0、指標/CI=null（NaN 不可）。
run ごとの値と学習 seed 間 mean/sd を既存 summary に加算する。FrequencyBaselines は bundle の source train だけを読み、test の頻度では再学習しない。
窓内 OOD の結果として retained/raw の率を必ず併記する。窓外/長さ超過を失敗 NLL=0 と扱わず、offset OOD の証拠にも使わない。

### 8.3 bucket 別 WF の観測契約（D1）

現行 `samples.json` は**無条件生成**で test sample_id を持たず、`eval.json` は全体集計。test bucket に join できない。
生成成功分だけを detokenize→bucket 分類すると失敗分が分母から落ち、WF が常に100%になる。さらに topology-only Sketch から元 CFG reducibility は復元できない。
したがって既存全体 WF（raw/reference-constrained/fully-constrained）を保持し、追加の bucket WF は**参照 prefix 継続 WF**として別名で設計する。
`training/bucket_eval.py` に frozen `WFProbe(sample_id: str, bucket: str, prefix: tuple[int,...], seed: int)` と次の API を追加する。
`make_wf_probes(rows: list[dict], index: dict, vocab: dict[str,int], *, seed: int) -> tuple[WFProbe,...]`。
各 retained test を sample_id sort し、EOS を含まない先頭 `max(1,len(tokens)//2)` token を prefix とする。seed は基点+その順序番号。bucket は**元の参照 CFG**のもの。
Colab の別呼び出し側が prefix から状態を replay し、各モデル・mask 条件で同一 prefix/seed/総長容量から EOS まで続ける。出力 stream は prefix を含み、条件を manifest hash とともに保存する。
sidecar は各 run の `bucket_samples.jsonl`: `{sample_id,bucket,seed,prefix_length,tokens,wf_kind,protocol:"reference-prefix-v1"}`。
公開 `bucket_wf(rows: list[dict], probes: tuple[WFProbe,...], vocab: dict[str,int]) -> dict[str,dict]` は prefix/ID/seed/bucket 一致と全 probe の応答を検証する。
`eval_samples.classify_stream` が ok の件数/全応答件数、Wilson CI、違反分類を集計する。切断・括弧失敗も分母に含む。参照自体の WF や生成後推測 bucket は使用しない。
controlled_eval は sidecar があれば読む。なければ bucket の `wf=null, wf_unavailable_reason="missing_reference_prefix_samples"`。全体 eval.json の率をコピーしない。
この値は B の無条件 WF と直接差分を取らず、ID/OOD の**同じ prefix protocol**間でのみ比較する。torch-free 集計は実装可能だが、prefix 継続の Colab 呼び出し追加は D1 承認後の別作業である。

## 9. 実験セルと固定手順

| セル | train / test と採用条件 | 構成・確認 |
|---|---|---|
| ID | layered v1 回帰 spec / 同族の別 seed | base,mask。既存 B を参照点とし、同じ新窓内条件の ID も確保 |
| family OOD | layered / structured yes | 最優先、base,mask,参考 ptr。n24 主、n32 確認 |
| family OOD′ | structured / layered と spaghetti を別 test にする | base,mask。逆方向を混ぜて1平均にしない |
| depth OOD | structured の realized max_depth≤1 / 2–3 | target_depth で誘導。3+ bin は4以上も含むため別途実測≤3で絞る |
| degree OOD | structured CFG merge_degree≤2 / 3–4 | 採用も realized max_in_degree≤2 / 3–4。4+ bin だけでは上限4を保証しない |
| balanced-k | mean_offset の3 binを等数採用 / 同じ採用分布 | 他次元は事前固定した active で揃える。個々の REF k の均等化とは異なる |
| offset OOD | C′へ延期 | 窓なし pointer を C では実装しない |

実測の上限フィルタは CLI の plan JSON に `ranges:{feature:[lower,upper]}` を持つラッパー（両端包含、null は無制限）で実装し、不一致は outside_plan とする。
pilot seed と本番 seed は分離。pilot で採用率・到達 bucket・CFG/MetaGraph 次数・構文深さとの差を確認し、spec/plan/seed 範囲/version を凍結してから本番生成する。
目標数の初期案は train≈2000 / val≈200 / test≈200。等数採用なので各 split は `target_per_bucket×active数` の実数を記録し、200へ無作為に切り詰めない（D2）。
学習 seed=0,1,2、mask が既定、全比較は対応 seed。family OOD 確定時には n=12,16,24,32,48 の節目スイープを行う（A7-5）。
ID→OOD の NLL 上昇は分布間の差、構成間 paired ΔNLL とは別欄。WF は protocol を併記。合否閾値は設けず、族×因子の表と除外率を残す。
`eval_axes` は source val の canary、`sketch_stats` は制約生成の分布忠実度に継続利用する。test の WF/分布を見て spec を調整する場合は新 version と独立 test を切る。

## 10. 順序付き実装タスクと受け入れ条件

各行は1回の実装依頼単位。依存は原則上から順。表の API は本文の完全シグネチャを使い、公開追加なしは「なし」とする。
各タスクで表の pytest に加え `uv run ty check` と `git diff --check` を実行。全テストは torch の import/インストールなし。新しいテスト関数名は以下で固定する。

| 順 | 変更ファイル（新規を含む）・公開 API | 受け入れ条件と確認コマンド |
|---|---|---|
| 1 | `generator_types.py`, `family_registry.py`, `tests/test_family_registry.py`。§2 の型、CFGFamily、register/get/names/load | normalize 冪等・未知/重複名・独自 params の往復、`test_registry_contract`。`uv run pytest tests/test_family_registry.py` |
| 2 | `generate_v2.py`, `families/__init__.py`, 同テスト。normalize/spec JSON/descriptor/generate の §2 API | toy plugin の異なる名前・同 seed が異なる UUID、config 再生、未知キー拒否。`test_plugin_descriptor_roundtrip`。同上 |
| 3 | `families/layered.py`, `tests/test_generate_v2.py`。公開追加なし、CFGFamily 契約を実装 | `test_layered_v1_exact`: n=1,2,4,6,12,24,32,48×seed0–49×p=0,.18,.5,1、ID/辺/MG一致、n3エラー。`uv run pytest tests/test_generate_v2.py -k layered` |
| 4 | `reducibility.py`, `tests/test_reducibility.py`。§6 is_reducible | `test_reducibility_cases` と `test_nested_multi_entry` が§6全例を検証。`uv run pytest tests/test_reducibility.py` |
| 5 | `families/structured.py`, `tests/test_structured_templates.py`。内部 Stmt/lower のみ | 各テンプレートと abrupt の接続、continue の while/do差、m=2合流木の手書き期待辺、到達性/reducible。`test_template_edges`, `test_merge_router`。`uv run pytest tests/test_structured_templates.py` |
| 6 | 同 structured、`tests/test_structured_planner.py`。内部 plan_structure のみ | L=0,1,2,4 / D=0,1,2,3 / G=0,1,3 の合法組合せでAST構文数/深さ一致、予算不可能の明示失敗。`test_plan_counts_depth`。`uv run pytest tests/test_structured_planner.py` |
| 7 | 同 structured、登録、`tests/test_generate_v2.py`。内部 pad_shape、plugin.generate | `test_structured_reducible_many_seeds`: n=12,24,48各500 seedの実現可能spec、成功CFG全件reducible、成功ゼロ禁止。`test_node_budget` は整数±10%、CFG次数、到達性、padding前後比較。`uv run pytest tests/test_generate_v2.py -k 'structured or node_budget'` |
| 8 | `families/spaghetti.py`, 登録、同テスト。公開追加なし | `test_spaghetti_rate_zero`, `test_spaghetti_edge_budget`: rate0完全一致、追加辺数/候補制約、全seedがirreducibleとは要求しない。`uv run pytest tests/test_generate_v2.py -k spaghetti` |
| 9 | `cfg_reducer/structure_features.py`, `training/structure_features.py`, `buckets.py`, 型、`tests/test_buckets.py`。§7 measure_realized | 既存23特徴を不変で移設、CFGノード数/reducible追加。`test_realized_features`。`uv run pytest tests/test_structure_features.py tests/test_buckets.py` |
| 10 | `buckets.py`, 型、同テスト。BucketPlan/bucket_for/acceptor/state/decision | `test_bucket_boundaries`（境界直前/一致/直後）、`test_bucket_plan_determinism`（充足/対象外/欠測/状態非変更）、ゼロtarget拒否。`uv run pytest tests/test_buckets.py` |
| 11 | `dataset.py`, 型、`tests/test_dataset_v2.py`。§7.1 accept/exclude 引数 | 固定toy生成で重複がquotaを消費しない、棄却がseenを汚さない、全失敗も有限、未充足集計。`test_accept_order_and_counts`。`uv run pytest tests/test_dataset.py tests/test_dataset_v2.py` |
| 12 | `dataset_v2.py`, `tests/test_dataset_v2.py`。`main(argv: list[str] | None=None) -> None`, load_references | spec/plan/ranges/plugin CLI、別dataset同型排除（孤立点付きfixture含む）、hash/version検査、未充足exit2。`test_v2_cli_and_cross_dataset_dedup`。同上 |
| 13 | `tests/test_dataset_v2.py`, `tests/fixtures/generator_v1_manifest.json`。公開追加なし | `test_manifest_legacy_bytes` は変更前v1の固定version/code fixtureと一致。`test_manifest_hashseed_determinism` はPYTHONHASHSEED=1,77の子プロセスで3族の全JSON比較、plan変更で同一候補UUID不変。`uv run pytest tests/test_dataset_v2.py -k manifest` |
| 14 | `training/prepare_tokens.py`, `tests/test_training.py`。§8.1 prepare 拡張 | `test_prepare_cross_dataset`: sourceだけから窓/容量、target test置換、byte同一vocab、長距離/長列の除外、index/除外分母、旧API回帰。`uv run pytest tests/test_training.py -k prepare` |
| 15 | `training/controlled_eval.py`, `tests/test_controlled_eval.py`。§8.2 summarize/bucket_scores | `test_bucket_nll_and_pairing`: 手計算の重み付きNLL、空bucket、ID不一致拒否、train頻度の維持、旧CLI出力。`uv run pytest tests/test_controlled_eval.py` |
| 16 | `training/bucket_eval.py`, `training/controlled_eval.py`, `tests/test_bucket_eval.py`。§8.3 WFProbe/make_wf_probes/bucket_wf | `test_bucket_wf_denominator`: valid/invalid各1で1/2、欠落/偽bucket拒否、sidecarなしnull、protocol名分離。モデルを呼ばずfixture使用。`uv run pytest tests/test_bucket_eval.py` |
| 17 | `tests/test_generator_v2_integration.py`。公開追加なし | `test_v2_ood_pipeline` はtoy plugin追加→生成→除外→tokenize→合成score/継続stream集計、旧sample読込まで実行。`uv run pytest tests/test_generator_v2_integration.py` |

最終確認は `uv run python -m pytest tests/`、`uv run ty check`、`git diff --check`。import で torch/matplotlib を引き込まないことも integration の子プロセスで確認する。
実験実行は上記ローカル実装とは別依頼。D1 未解決時も生成・NLL・WF集計器は実装できるが、bucket WF の実測完了とは報告しない。

## 11. Claude が判断する未決事項

- **D1（WF の意味・採取範囲）**: 参照 prefix 継続 WF を bucket WF として採用するか。採用なら §8.3 の Colab 呼び出し側の追加を別タスク化する（train_ar.py は変更しない）。無条件 WF を意図する場合、現行出力では4次元 bucket 別率を同定できないため、A7 の評価要件かモデルの条件付けを改訂する判断が必要。
- **D2（本番 quota と seed 予算）**: pilot で到達する active と split 別 target_per_bucket、有限 seed 範囲を確定する。推奨は train約2000 / val,test約200を維持し、54直積の充足を前提にしない。境界値は A7 承認済みのまま。
- **D3（span の効果と節目）**: §4.2 の辺分割による span 誘導が pilot で十分か、より強い構文配置制御を次版に入れるか。推奨は最初の family OOD 確定を n12–48 スイープの節目とし、offset OOD と窓なし pointer は C′へ留める。

## 12. 進行役(Claude)の決定(2026-09-09)

本節は §11 の D1–D3 への回答と、実装前に固定する簡略化である。**本節が上位の各節と
矛盾する場合は本節を優先する。**

| # | 決定 | 影響箇所 |
|---|---|---|
| D1 | **bucket 別 WF は採用しない**(参照 prefix 継続 protocol は導入しない)。WF は従来どおり test dataset 単位の全体率(raw / reference-constrained / fully-constrained)で、OOD セルごとに test dataset が異なるためセル別 WF はそれで得られる。bucket 別指標は teacher-forced の NLL / edge accuracy のみ | §8.3 全体と **タスク 16 を削除**。§8.2 の `wf` フィールドは出力しない。タスク 17 の統合テストから継続 stream 集計を外す |
| D2 | quota は **pilot 後に固定**。第一段は各族の自然分布(`measurement_acceptor`)で n24 の pilot(train 2000 / val 200 / test 200 相当の seed 範囲)を生成し、bucket 占有と採用率を見てから、セルごとに `active` / `ranges` / `target_per_bucket` を決める。ID・family OOD は自然分布、depth / degree OOD は `ranges` フィルタ、balanced-k のみ mean_offset の等数採用 | §7・§9。54 直積の充足は前提にしない |
| D3 | span 誘導は §4.2 の辺分割のままで pilot に進む。より強い構文配置制御は次版 | §4.2 |
| D4 | **別 dataset 同型排除の簡略化**: `load_references` は manifest の provenance(name / version / seed / config)から registry 経由で CFG を再生成し、再生成した provenance の `sample_id_for` が manifest の `sample_id` と一致することだけを検証する(生成器版の同一性を直接確認できる)。`--exclude-cfg FILE`、content hash、`code.commit` と checkout の一致検査は**導入しない**。manifest の sha256 は情報として selection に残す | §8.1、タスク 12 |
| D5 | **manifest の肥大化防止**: `rejections` の配列は manifest に置かず、sidecar `rejections.jsonl`(seed / reason / bucket / realized / duplicate_of)に出す。manifest には split ごとの `attempts / accepted / rejected / acceptance_rate / complete` と `per_bucket`(target / attempts / accepted / rejected / missing)、`rejected_by_reason` の件数だけを残す。sample エントリの `requested / realized / bucket` は維持 | §7.2、タスク 11–13 |
| D6 | `build_dataset` の `accept` と `exclude` は**独立**の keyword-only 引数(併用必須にしない)。`exclude` のみでも同型排除だけを行う | §7.1、タスク 11 |
| D7 | **テスト規模を既定では小さく**: `test_layered_v1_exact` は n ∈ {1, 2, 4, 6, 12, 24, 32, 48} × seed 0–9 × p ∈ {0.18, 0.5}、`test_structured_reducible_many_seeds` は n ∈ {12, 24, 48} × 100 seed。環境変数 `GR_SLOW_TESTS=1` のときだけ原案の規模(seed 0–49 × 4 p、500 seed)に拡大する。テストスイート全体を数秒以内に保つ | タスク 3・7 |
| D8 | 実装は §10 の順序で、1 依頼につき 1〜2 タスク。各依頼の受け入れは Claude が `uv run pytest` / `uv run ty check` / `git diff --check` を再実行して確認し、コミットする。タスク 16 は削除、タスク 17 は D1 に合わせて縮小 | §10 |

補足: `GeneratorSpec.params` は族固有キーを持つ dict であり、族の追加は
`families/<name>.py` の追加と `families/__init__.py` の登録行 1 つで完了すること(A7-1)を、
タスク 17 の統合テスト(toy plugin)で確認する。
