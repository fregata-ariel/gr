# 逐次ベイズ実験計画 詳細設計(`training/seqdesign.py`)

作成: 2026-09-26。状態: 実装前。上位の決定は [`seqdesign.md`](seqdesign.md) §2。
本書は有限候補集合・独立 GP・既定 TS・seed 反復・既存 Observation / Plan の再利用を具体化する。今回の成果物は本書のみ。シミュレーション、遡及評価、実機実験の結果はまだ存在しない。

## 1. 目的・適用範囲

固定10点と大域 Scheffé 多項式を、候補全体の不確実性を残す局所的な代理モデルに置き換える。
最小値の追求だけでなく応答の特徴を調べるため、TS による乱択と反復を維持する。
`mixture_doe.md` §8.1 の純 structured → layered の生 NLL 3.814 と、少量混合時の急減は
頂点近傍の特異性である。二次・特殊三次の lack-of-fit は正規化後も残り、滑らかな GP でも
自動的に解消する保証はない。頂点と内部を分けた予測区間の検証を必須とする。
主応答は §9.1 と `obs_xt.json` の target 固有基準を引いた超過 NLL、負ほど良い。
生 NLL、source 条件付き超過 NLL、異なる size / dataset version は同じ campaign に混ぜない。

## 2. モジュールと API

| 場所 | 責務 |
| --- | --- |
| `training/seqdesign.py` | 型・検証・JSON、GP、獲得関数、状態更新、報告、合成検証、5サブコマンド |
| `experiments/mixture_doe/materialize.py` | mixture 因子から spec・コマンド列・既存 Plan を生成 |
| `tests/test_seqdesign.py` | 決定的な数値・状態・CLI・materializer の契約試験 |
| `experiments/seqdesign_sim/` | 後日の合成検証と遡及評価の Markdown / JSON / 任意の図 |

GP・獲得・状態処理の外部 import は標準ライブラリと `numpy` のみ。scipy / torch は使わない。
`Observation` は再定義せず `training.mixture_doe` から再利用し、JSON 入出力も同モジュールの
`observations_from_json` / `observations_to_json` に委譲する。接続部分で遅延 import する。
既存モジュールは import 時に matplotlib と cfg_reducer を読み込むため、観測接続を含む
プロセス全体まで numpy 単独になるわけではない。既存依存を増やさず、数値コアを独立させる。
任意の図だけ matplotlib(Agg)を使う。既存モジュールの依存整理や Observation の移設は範囲外。

```python
@dataclass(frozen=True)
class Space:
    factor_names: tuple[str, ...]
    candidates: tuple[dict[str, float], ...]
    simplex: tuple[str, ...] | None = None

@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    candidate_id: int
    point: int
    factors: dict[str, float]
    seeds: tuple[int, ...]
    acquisition: str
    observation_revision: int
    campaign_revision: int

@dataclass(frozen=True)
class Campaign:
    version: int
    campaign_id: str
    revision: int
    observation_revision: int
    space: Space
    responses: tuple[str, ...]
    objective: dict[str, object]
    input_mode: str
    context: dict[str, object] | None
    observations: tuple[Observation, ...]
    proposals: tuple[Proposal, ...]
    registry: tuple[dict[str, object], ...]
    rng_state: dict[str, int]
```
補助型 `GPFit`(X、標準化量、超パラメタ、L、alpha、診断)、`Posterior`(mean、covariance)も frozen。
配列・dict は生成時にコピーし、公開後は変更しない。更新は `dataclasses.replace` で新しい値を返す。
公開 API は `validate_space`、`campaign_from_json/to_json`、`observe(campaign, observations)`、
`fit_gp(X, y)`、`posterior(fit, Xstar)`、`draw_joint(posterior, rng)`、
`propose(campaign, k, seeds, acq)`、`report(campaign, out, fig_dir=None)`、`simulate(...)`、`main(argv)`。
`propose` は `(new_campaign, proposal_payload)` を返す。数値関数からファイルや runner を操作しない。

## 3. 空間・観測・永続 JSON

### 3.1 `space.json` と因子の対応

候補 ID は入力配列の0始まり index。順番を保存し、初期化後に変更しない。
因子名は非空・一意、候補は非空、各 dict は同じキー集合、値は有限の数値(bool を除く)。重複候補は禁止。比較キーは `factor_names` 順の float tuple(-0.0 は 0.0 に正規化)とする。
`simplex` が非 null なら指定因子は非負、和は1との差が `1e-9` 以下。他の因子も併存できる。
カテゴリは事前に one-hot 化し、有効な組合せだけ列挙する。連続最適化や候補への勝手な丸めは行わない。

```json
{
  "factor_names": ["lay", "str", "spa"],
  "simplex": ["lay", "str", "spa"],
  "candidates": [
    {"lay": 0.5, "str": 0.25, "spa": 0.25},
    {"lay": 0.0, "str": 1.0, "spa": 0.0},
    {"lay": 0.0, "str": 0.0, "spa": 1.0}
  ]
}
```

mixture の通常空間は `simplex_grid(0.05)` の231点を同順で dict 化する。
格子生成に既存関数を使う場合、`round(1/step)*step` が1に一致する step のみ許可する。

### 3.2 観測 JSON と registry

入力は既存 collect の **配列そのもの**。例は形を説明する架空の1行である。

```json
[{"point":100,"seed":0,"x":[0.495,0.268,0.237],
  "y":{"lay":-0.454,"str":-0.376,"spa":-0.333},
  "wf":0.87,"n_train":2139,"y_raw":{"lay":0.799,"str":0.364,"spa":0.629}}]
```

`observations_from_json` に渡す前に型・長さ・有限性を検証する(既存 parser の int 変換で
小数の点番号を受理しない)。全 responses が y に必要。y の追加キーは保存するが学習しない。
`wf` は null または [0,1]、`n_train` は非負整数、point / seed は非負整数。
`y_raw` は参照専用で、y を再正規化しない。全観測の保存に既存 JSON 往復を使用する。
同一 `(point, seed)` の同一内容は no-op、値の違う再投入は全体を拒否。訂正は別 campaign で行う。

- `input_mode="mixture"` は因子名と simplex が lay / str / spa の3成分であることを要求する。
  GP の学習座標は `Observation.x`(lay,str,spa 順の**実現組成**)を因子順へ変換する。
  予測座標は候補の**名目組成**。実現組成が格子外でも受理し、最近傍候補に置き換えない。
- `input_mode="registry"` は一般数値因子用。学習座標は registry の `factors` から取得する。
  Observation.x の3成分制約は変更しない。一般ドライバは x を `[0,0,0]` として保存し、
  GP は x を参照しない。未知 point は拒否し、暗黙に因子を推定しない。
- registry の1要素は `{"point":100,"candidate_id":0,"factors":{"lay":0.5,"str":0.25,"spa":0.25}}`。
  名目パターンと point は一対一。候補外の過去点は `candidate_id:null` で登録できる。
  mixture の未知な過去点は observe 時に candidate_id / factors を null として仮登録できる。
  この場合も実現組成で学習できるが、名目パターンを推測して再実体化してはならない。
- 既知の過去点を再提案できるよう、init の任意 `--registry registry.json` に上記要素の配列を渡す。
  p1〜p12 は既存 specs の重みを正規化して作る。一般空間では初期観測の point 対応をここで与える。
  同一点の mixture 反復は x / n_train が一致すること。データを作り直したものは反復に混ぜない。

### 3.3 `campaign.json`

全キー必須。version は schema version の整数1であり、dataset の git commit とは別物。
下例は init 直後の完全な小規模 campaign。context の null は実験条件未設定を表す。

```json
{
  "version": 1, "campaign_id": "adaptive24", "revision": 0, "observation_revision": 0,
  "space": {"factor_names":["lay","str","spa"],"simplex":["lay","str","spa"],
    "candidates":[{"lay":0.5,"str":0.25,"spa":0.25},
                  {"lay":0.0,"str":1.0,"spa":0.0},{"lay":0.0,"str":0.0,"spa":1.0}]},
  "responses": ["lay","str","spa"],
  "objective": {"kind":"bal","responses":["lay","str","spa"],"direction":"minimize"},
  "input_mode": "mixture", "context": null,
  "observations": [], "proposals": [], "registry": [],
  "rng_state": {"seed":20260926,"draws":0}
}
```

objective.kind は `bal|max|response`、
response の場合 responses は1要素、bal/max は campaign.responses 全体。同名応答との衝突は禁止。
`--objective lay` のような指定は response に展開する。任意 Python 式や重み付き合成は初版範囲外。
context は init の任意 `--context context.json` から読み込む不変の実験条件。mixture の例:

```json
{"size":24,"prefix":"a24","data_root":"data","max_offset":20,
 "response_scale":"excess_target","baseline_dir":"experiments/mixture_doe/baselines",
 "dataset_version":"<generation git commit>","test_manifests":{
   "lay":"<sha256 of data/s24_layered/manifest.json>",
   "str":"<sha256 of data/s24_structured/manifest.json>",
   "spa":"<sha256 of data/s24_spaghetti/manifest.json>"}}
```

山括弧内は実際の commit/hash に置換する。baseline_dir は既存基準ファイルの実所在を指定し、
`base_{lay,str,spa}.jsonl` の SHA256 も実体化時の manifest に記録する。
collect JSON 自体には scale/version 情報がないため、observe だけでは混入検出を保証できない。
実験条件を固定し、materializer が生成する collect コマンドと manifest を確認して取り込む。
JSON は UTF-8、`sort_keys=True, indent=2, allow_nan=False`、末尾改行。未知 schema / キーは拒否。
観測は `(point,seed)` 順、registry は point 順、proposal は作成順。時刻は再現性のキーにしない。

### 3.4 乱数・更新・提案出力

乱数源は注入した `random.Random(seed)` だけ。numpy.random と Python hash() は使わない。
`rng_state` は初期 seed と `random()` の消費回数で状態を表す。復元は同じ seed から同回数進める。
正規乱数1個は2個の一様乱数から Box–Muller の cos 側で生成し、残りは捨てる(キャッシュなし)。
`u1=1-rng.random()` として log(0) を防ぐ。一様候補選択は `floor(M*rng.random())`。
初版の規模では再生コストを許容する。report 用 RNG は別の固定 seed 0、campaign の状態を進めない。
再現性は同じ Python/numpy/BLAS 環境で保証し、環境差の浮動小数点の完全一致は要求しない。

propose は選択順に `q000001` から proposal_id を払い出す。新パターンは未使用の point >=100 の
最小値を割り当て registry に保存する。既存登録パターンならその point を使う。
各提案の seed は当該 point の観測・過去提案・今回の予約の最大 seed +1 から連続で確保する
(未使用なら0)。重複候補を統合せず別 Proposal とし、新 seed を割り当てる。
proposal の完了数は対応する観測キーから導出する。再実行は保存済み提案を使い、新たな propose を呼ばない。

```json
{"version":1,"campaign_id":"adaptive24","campaign_revision":1,
 "context":null,"proposals":[
   {"proposal_id":"q000001","candidate_id":0,"point":100,
    "factors":{"lay":0.5,"str":0.25,"spa":0.25},"seeds":[0,1],
    "acquisition":"ts","observation_revision":0,"campaign_revision":1}]}
```
出力 envelope の context は campaign の完全なコピー。materializer は非 null の mixture context を要求する。
observe は新規観測があると両 revision を1増加、propose は revision のみ1増加する。
更新は検証→メモリ上で計算→同じ directory の一時ファイル→`os.replace`、その後 stdout 出力。
失敗時は元 campaign を保持する。単一 writer 運用とし、同時更新はサポートしない。
各 Proposal に作成後の campaign_revision を保存。`propose --replay-revision N` はそのバッチを再出力し、状態を変更しない。

## 4. GP の数値仕様

### 4.1 前処理・ノイズ

応答ごとに独立な GP を当てる。seed を入力因子にも固定効果にもせず、全反復行を残す。
各因子は候補集合の min/max で [0,1] に線形変換、一定列はカーネルから除く。
観測も同じ変換を使う(範囲外へ clip しない)。simplex の全成分を残し、対数比変換はしない。
応答は全行の平均 m と母 SD s で `z=(y-m)/s`。s < `1e-8` なら s=1 とする。
観測0件は m=0,s=1 の事前分布。超過 NLL の符号を変えたり log にしたりしない。

同じ学習座標にある反復を group g とし、応答ごとに
`v_pe = sum_g sum_i (y_gi - mean_g)^2 / sum_g(n_g-1)` を計算する。
分母>0なら標準化ノイズ分散 `v=clip(v_pe/s²,1e-10,10)` を固定して振幅・長さだけを最尤化する。
反復がなければ v も周辺尤度で推定する。初期値0.01、範囲 `[1e-10,10]`。
観測が2座標未満なら最尤化を行わず初期値を使う(反復があればその v を使う)。
初版は応答別の等分散ノイズ。p2 の大きな seed 分散で内部の誤差を過大評価し得ること、
反復なしでは短い長さと大きなノイズを識別しにくいことを report に記す。異分散 GP は追加しない。

### 4.2 カーネル・周辺尤度最大化

`kf(x,x') = a * exp(-0.5 * sum_d ((x_d-x'_d)/ell_d)^2)`、a は信号**分散**。
観測共分散は `K = kf(X,X) + v*I`。同じ x でもノイズは別 seed なら独立。
初版は ARD RBF のみ。Matérn 5/2 は有限の滑らかさの比較には価値があるが、頂点の急変を
それだけで解決しないため初版には含めない。カーネル追加は校正結果を見た後の独立タスクとする。

探索変数は自然対数 `log(a), log(ell_1..ell_D)` と、反復なしの場合 `log(v)`。
範囲は a=`[1e-4,100]`、ell=`[0.02,2]`、v は上記。3つの決定的な restart を使う:
`a=1, ell_d=0.05/0.3/1.0(各 restart で全次元同値), v=0.01`。
各 restart は有界座標探索: 各 log 範囲の1/4を初期 step とし、a→因子順ell→v の順に
現在値・clip した ±step を比較、LML が `1e-8` より改善した候補へ移動する。
1 sweep で改善がなければ全 step を半減。最大30 sweep または全 step<`1e-3` で終了。
最高 LML の restart を採用、同点は先の restart / 現在値 / 負方向 / 正方向の順。
全反復で再標準化・再 fit する。warm start による呼出し履歴依存を入れない。

`L=cholesky(K+jI)`, `alpha=solve(L.T,solve(L,z))` とし、
`LML = -0.5*z.T@alpha - sum(log(diag(L))) - n/2*log(2*pi)` を最大化する。
逆行列・行列式の直接計算はしない。j は `max(1,a+v)*1e-10` から10倍ずつ `1e-4` 倍まで。
Cholesky 失敗・非有限 LML はその試行を不採用(診断は null と理由を記録)、全試行失敗なら例外。
採用 jitter とパラメタの境界到達を report する。jitter を観測ノイズの推定値に足して報告しない。

### 4.3 候補全体の事後分布

`B=kf(X,C)`, `V=solve(L,B)`, `mu_z=B.T@alpha`, `Sigma_z=kf(C,C)-V.T@V`。
逆標準化は `mu=m+s*mu_z`, `Sigma=s²*Sigma_z`。これはノイズなしの潜在応答 f の分布。
将来1 seed の観測予測は `Sigma + s²*v*I`、r seed 平均の予測分散は対角へ `s²*v/r` を足す。
候補だけでなく任意の観測座標にも同じ posterior API を使えるようにする。
Sigma は `(Sigma+Sigma.T)/2` に対称化。標準化単位で `1e-8*max(1,a)` 以下の負固有値は
丸め誤差として0へ修正し、それ以上なら失敗する。修正後、上と同じ jitter 再試行で Cholesky 化。
サンプルは `f=mu+Lposterior@normal_vector`。候補ごとの独立な正規乱数を直接応答に使わない。
潜在 SD と CI の表示は sampling jitter を除く。応答間には共分散を仮定しない。
計算量は応答あたり fit が探索回数×O(n³)、候補 sampling factor が O(M³)、保存が O(n²+M²)。
231〜235候補と数十観測を初版の対象とし、各 posterior の分解をバッチ内で再利用する。

## 5. 獲得関数・推奨点

全て最小化。合成は**元の応答単位に戻したサンプル**で `bal=mean_r(f_r)` または `max=max_r(f_r)`。
`max(mean_r)` は max の事後期待値ではない。bal の mean/covariance は独立 GP の和 / R、和 / R²
で厳密にも求められるが、獲得比較では同じ合成サンプル API を使い、別の bal GP は fit しない。

- `ts`(既定): 各応答から全候補にわたる関数を1本ずつ joint sample、合成の argmin を選ぶ。
  バッチ k は同じ事後から独立に k 回描く。pending 点の除外・fantasy 更新・重複排除はしない。
- `ei`: S=1024本の joint sample で `EI(c)=mean_s(max(b-g_s(c),0))` を推定し argmax。
  候補と既観測座標を連結して描き、b は既観測座標での標本平均の最小(実現組成も含む)。観測ノイズの
  下振れを閾値にしない。これは plug-in incumbent の MC-EI であり完全な noisy EI ではない。
- `ucb`: 最小化用の下側信頼境界 `mean_s(g(c))-2*sd_s(g(c))` の argmin。CLI 名は ucb を維持。
  SD は ddof=0。EI/UCB はバッチに1つの1024本集合を使い、選んだ同じ点を k 回反復する。
- `random`: 候補から独立な一様復元抽出。`fixed` は §8 の固定10点を順に循環(simulate 専用)。
  観測0件の EI は incumbent がないため random に退避し、診断にその理由を記録する。
同点は候補 index の最小。TS の乱数消費順は draw→responses 順→候補順とする。
推奨点は TS の最後の選択ではなく、合成事後期待値が最小の候補。
report は独立の固定 RNG で4096本を描き、推奨点と95%等裾 credible interval を求める。
この区間は選択された候補の潜在応答の区間であり、真の最適点の位置や最小値の信頼区間ではない。

## 6. CLI と report

`seqdesign.md` §3 の5コマンドをそのまま実装する。全例の起動は `uv run python -m training.seqdesign`。

```text
init     --campaign C.json --space space.json --responses lay,str,spa --objective bal
observe  --campaign C.json --obs obs.json
propose  --campaign C.json --k 3 --seeds 2 --acq ts
report   --campaign C.json --out report.md [--fig-dir DIR]
simulate --surface scheffe --rounds 8 --k 3 --repeats 20 --out sim.md
```

init は `--seed 20260926`、`--input-mode mixture|registry`(既定 mixture)、任意 `--registry` / `--context`
も受ける。campaign_id は C の stem、既存ファイルは上書きしない。responses の順を固定する。
propose の既定は k=3,seeds=2,acq=ts、acq は ts/ei/ucb/random。正の整数だけ許可する。
propose の stdout は §3.4 の JSON のみ(リダイレクトして `proposal.json` とする)、他の診断は stderr。
simulate の追加引数は `--seeds 2 --seed 20260926`、遡及用 `--obs PATH`。詳細は §8。
成功0、入力・数値計算・I/O エラー2。report / simulate は campaign とその RNG を変更しない。

report の本文は日本語、以下を Markdown で出す。数値丸めは表示のみで、内部値は丸めない。

1. context、schema、観測数・点数・予約中 seed 数、input_mode、名目/実現座標の意味。
2. 応答別の m/s、a/ell/v、pure-error df、LML、jitter、境界到達と未観測時の注記。
3. **全候補**の因子、各応答と bal/max の潜在 mean/SD、登録 point と観測 seed 数。
4. 目的別の推奨候補・95%区間、将来1 seed の予測区間(潜在区間とは別欄)。
5. proposal_id、観測 revision、acq、point、名目因子、seed と完了/未完了の履歴。
6. mixture で p7 と共有 seed があるときは `paired_delta` の lay/str/spa/bal/max を補足する。
   これはモデルによらない観測対比。共有 seed が1本の NaN CI は JSON に出さず「算出不能」とする。

図は3成分 simplex のみ、`--fig-dir` 指定時に mean/SD の三角図を保存する。
座標 `(str+spa/2, sqrt(3)*spa/2)` に名目候補・実現観測・推奨点を区別して描く。
非三角形の小集合や共線点は散布図に退避。図なしでも表は全情報を保持する。

## 7. mixture materializer と Plan

```text
uv run python experiments/mixture_doe/materialize.py --proposal proposal.json --out-dir experiments/mixture_doe/adaptive/r01
```

入力は §3.4 の出力。context.size は既定24、max_offset は n24=20(n48 を使う場合37)で固定。
context の必須条件を init 時に設定し、materializer では変更しない。未対応 size は拒否する。
同一 point の重複提案は dataset/spec/bundle を共有し、全予約 seed の train jobs を同じ Block に入れる。
出力は `specs/spec_p<P>.json`、`generate.sh`、`tokenize.sh`、`collect.sh`、`plan.json`、`manifest.json`。
生成・学習は実行しない。manifest は `{version:1, campaign_id, campaign_revision, context,
proposal_ids:[...], files:{relative_path:sha256,...}}`(自身の hash を除く)とし再実体化を監査する。
同じ出力は再利用、異なる既存ファイルとの衝突は拒否する。既存 dataset を削除する命令は生成しない。

spec は既存 `spec_p7.json` の成分設定を再利用し、順は layered / structured / spaghetti。
重み0の成分を除き、1成分なら既存 p1/p2/p3 同様に純族 spec、複数なら mixture spec とする。
`num_nodes=24`、loop_count=3、goto_count=2、layered は max_layer_width=3 / edge_prob=0.18 /
span_mode=uniform、spaghetti_rate=0.1。n48 は loop_count=7 / goto_count=4 にする。
正重みは提案の値そのもの。例えば候補0なら p7 の weight を順に0.5/0.25/0.25へ置換する。

生成コマンドは repository root 基準、引数を shlex.quote して出力し、`set -euo pipefail` を付ける。
以下は n24 / p100 の例(出力先 spec は実際の out-dir に展開する):

```bash
uv run python -m cfg_reducer.dataset_v2 --spec experiments/mixture_doe/adaptive/r01/specs/spec_p100.json --out data/d24_p100 \
  --split train=800000:802150 --split val=802150:802370 --split test=802370:802400 \
  --exclude-dataset data/s24_layered --exclude-dataset data/s24_structured --exclude-dataset data/s24_spaghetti \
  --allow-version-mismatch --allow-incomplete
uv run python -m training.prepare_tokens --dataset data/d24_p100 --test-dataset data/s24_layered --max-offset 20 --out data/tok_d24_p100_lay
```

tokenize は同じ train/val に対して structured→`_str`、spaghetti→`_spa` も生成する。
`allow-version-mismatch` は既存の固定 test 除外集合を参照するための従来規約であり、学習 dataset の
version 混在を認めるものではない。生成前に git commit と context.dataset_version を照合する。
既存 dataset/bundle は spec、version、split、test hash、max_offset が一致するときだけ再利用する。
accept 数と実現組成を記録し、collect で manifest から得る x を学習に使う。2150は試行数である。

Plan は `training.runner.types` の `Plan/Block/TrainJob/RescoreJob` を構築し、
`validate_plan` → `plan_to_json` → `plan_from_json` 往復検証で保存する。独自キーを足さない。
下例は紙幅のため1 seed の完全な Plan。実際の既定 seeds=2 なら同じ設定の seed1 も含める。

```json
{"plan_id":"adaptive24_r1","blocks":[{
 "source_bundle":"data/tok_d24_p100_lay",
 "train":[{"name":"a24_s24_p100_mask_n24_s0","epochs":300,"patience":20,
   "num_samples":400,"constrained_samples":400,"seed":0,"sample_seed":1000,
   "extra":["--ref-legal-mask"]}],
 "rescore":[
   {"source_run":"a24_s24_p100_mask_n24_s0","target_bundle":"data/tok_d24_p100_str","out_run":"a24_s24_p1002str_mask_n24_s0"},
   {"source_run":"a24_s24_p100_mask_n24_s0","target_bundle":"data/tok_d24_p100_spa","out_run":"a24_s24_p1002spa_mask_n24_s0"}]}]}
```

一般形は `<prefix>_s<size>_p<P><suffix>_mask_n<size>_s<S>`、suffix は空または `2str/2spa`
(実際には p<P> 直後に区切りを挟まず連結)。sample_seed=1000+S。Block 順は point、job 順は seed。
collect は run prefix を引数にできるが dataset 名は `d<size>_p<P>` 固定なのでこれを維持する。
複数 campaign は別 data_root を使うか point 範囲を分け、既存パターンと衝突すれば停止する。
生成 collect.sh は point ごとの実際の seed 集合を使って `collect --prefix a24 --size 24 --runs runs
--data data --points 100 --seeds 0,1 --baseline-dir <context値> --baseline-mode target --out obs_p100.json`
を `uv run python -m training.mixture_doe` で呼ぶ。各出力を observe に渡せばよい。
runner は `uv run python -m training.runner run --plan <plan.json> --dry-run` で先に確認する。

## 8. Validation の実行プロトコル

### 8.1 合成曲面・予算・指標

全手法共通の候補は0.05 simplex 格子231点 + 固定設計の重心・軸点4点 =235点。
追加4点の順は p7,p8,p9,p10。既存固定10点 §3 を正確に含め、最近傍格子に丸めない。
`a=(0.40,0.25,0.35)`, `q(x)=-0.4+0.12*sum_i(x_i-a_i)^2`, `d=x_lay-x_str` として
`f_lay=q-0.06*d`, `f_str=q+0.04*d`, `f_spa=q+0.02*d` を基礎曲面にする。
これは simplex 上で Scheffé 二次(線形係数 `-0.4+0.12*(1+sum(a²)-2*a_i)`、交互作用係数-0.24、
上記応答別線形項を加算)。bal の内部は平坦で、有限集合の真の最適値は全候補列挙で得る。

`--surface scheffe` は以下の3条件を一括実行する: 基礎曲面のみ、
`h=exp(-(1-x_str)/0.025)` を lay に3.0h / spa に0.9h 加えた頂点急変、
同じ加算を `h=1[x_str==1]` にした頂点不連続。後二者は厳密な二次でないことを明記する。
各条件で独立 Gaussian noise の SD を0 / 0.005 / 0.03、さらに頂点急変に限り
`sd_lay=0.005+0.125*h`、他応答0.005の異分散条件を追加し、等分散 GP の限界も測る。
目的は bal と max を別 campaign として全組合せ評価する。
各条件・目的・手法は **8 round × k=3提案 × 2学習 seed =48観測、20 repeats**。
warm-up は設けず空 campaign から開始。1 round 内にフィードバックせず6観測をまとめて追加する。
TS/EI/UCB/random/fixed を比較し、fixed は p1〜p10 の順を循環して24提案を使い切る。
同点再訪も予算に数える。fixed に追加の初期10点や最適化情報を与えない。
repeat seed は base+r(r=0..19)。応答ノイズは候補→反復番号→応答の順に Random で事前生成し、
全手法が同じ候補の第j反復に同じノイズを受ける。獲得用 RNG は別インスタンスとする。
毎 round 全手法で同じ GP/report 推奨則を使い、観測済み最良値とは別に次を記録する。

- simple regret: `g_true(recommended)-min_C(g_true)`。横軸は累積観測数6,12,...,48。
- 最適近傍への予算比率: `g_true(c)-g_true_min <= 0.01` の候補で消費した seed 数 / 総 seed 数。
  平坦面を考慮し、座標距離でなく目的値の差(nats/token)で近傍を定義する。
- 校正: 観測前に次 round の各 y の95%予測区間を凍結し、到着後に被覆率と区間幅を計算する。
  別に全候補の潜在真値に対する95% credible interval 被覆率も計算する。
  応答別・合成別、頂点(`max(x)>=0.95`) / 内部別に集計。合成の観測区間はサンプルへ noise を足す。
20反復の平均・中央値・10/90百分位、平均差の paired SE を出す。優勝手法を合格条件にしない。
滑らかな低ノイズ条件で regret 減少を確認し、95%被覆率のずれは反復単位のSEとともに評価する。
頂点条件の失敗・過信も validation の結果として残す。過信時に実機へ自動進行しない。
`sim.md` と同 stem の `sim.json` を保存し、JSON は `{version:1, config:{...}, results:[...]}`。
config は係数、候補、noise条件、全 seed、環境版、各CLI値。results は condition/objective/method/
repeat/round/budget/recommended/regret/near_fraction/coverage/interval_width を持つ行配列。

### 8.2 `obs_xt.json` の遡及と実機確認

`simulate --surface scheffe ... --obs experiments/mixture_doe/obs_xt.json` は合成検証に加え遡及を出す。
36観測(12点×seed0,1,2)を point=1..12、同点内 seed=0..2 の固定順で、1点の3行ずつ投入する。
候補は §8.1、GP 学習は実現組成。p1〜p12 の名目 registry は既存 specs から作る。
6点(18行)と12点(36行)で state をコピーし、各コピーの提案 RNG を `Random(20260926)` に設定、
TS k=3,seeds=2 の proposal JSON と report を保存する。コピー上の提案を元の遡及系列へ戻さない。
それぞれ bal/max で行い、何を提案したかを候補因子・point・事後 mean/SD とともに実測記録する。
未観測候補の反実仮想の y や regret は計算しない。6点時点から残り6点への予測区間被覆率も報告する。
元データの照合値: p2 の lay=+2.561091、bal=+0.885077、p7 bal=-0.386427、
p8=-0.384226、p10=-0.381217、p11=-0.387800、p12=-0.388062(seed平均)。
この内部約0.007幅の平坦さを再現するか、混合内部の max を spa が支配する領域を再現するか、
structured 頂点を不当に内部へ広げないかを調べる。原点数が少ない6点時点で一致を強制しない。
実機は上位決定どおり n24 の2 round×3提案×2 seed。Colab 実行と collect→observe→report を
オーケストレータが担当し、bal 平坦・相対的に spa が苦手という既存結論との整合を記録する。

## 9. テストと実装タスク

`tests/test_seqdesign.py` の通常テストは小集合と固定 seed のみ、torch import / GPU / 実験データを不要にする。

| 試験 | 受け入れ条件 |
| --- | --- |
| 1-D 関数復元 | 41点 `[0,1]`、`sin(2*pi*x)` を9等間隔点で観測、held-out RMSE<0.10(最尤 fit も通す) |
| 観測点の事後収縮 | 固定 a=1,ell=0.2,v=1e-10 で平均誤差<1e-5、潜在分散<1e-7、重複行でも有限 |
| TS の最適点 | 11点格子の `(x-0.3)^2` を各点2回無雑音観測、固定 RNG で10提案が全て x=0.3 |
| joint sampling / 合成 | 固定小共分散で標本共分散を許容誤差内に復元、max(mean) と mean(max) の差も検査 |
| JSON・再開 | Observation 往復、campaign 往復後の次提案完全一致、report 前後の campaign byte 不変 |
| 状態の検証 | 再 observe は no-op、矛盾入力は不変、格子外実現組成受理、一般因子 registry 対応、NaN 拒否 |
| バッチ予約 | 重複候補でも seed 不重複、既存点の再利用、新点>=100、未完了提案を含む再開 |
| CLI smoke | tmp_path で init→observe→propose→report、小さな simulate、異常系で終了2・元JSON不変 |
| materializer | 頂点/辺/内部、重複点2提案の4 train+8 rescore、Plan 往復と validate、固定窓・run名・spec値 |

数値試験の閾値は実装に合わせて緩めず、失敗時に数値仕様を調査する。異なる PYTHONHASHSEED で
提案JSONが一致する subprocess 試験も小規模で行う。20反復 sweep は `GR_SLOW_TESTS=1` のみ。

| # | 1回の Codex low / OpenCode に渡す範囲 | 完了条件 |
| --- | --- | --- |
| T1 | seqdesign.py の型・検証・JSON・前処理・GP、対応テスト(§2〜4) | 関数復元・収縮・ノイズ推定・JSON往復が合格、既存 Observation 無変更 |
| T2 | 同ファイルの獲得・状態更新・CLI・report・simulate と対応テスト(§5,6,8) | 決定的再開、CLI smoke、小規模な全手法比較、遡及の6/12点出力が生成可能 |
| T3 | materialize.py と契約テスト(§7) | spec・コマンド・Plan の生成、重複反復と再実体化が安全、runner 形式検証が合格 |

T1→T2→T3 の順。各タスク後に `uv run pytest -q`、`uv run ty check`、`git diff --check`。
全検証実行と成果物の保存、上位 `seqdesign.md` §5 への実行記録は実装後にオーケストレータが行う。
各 run は割当範囲だけを変更し、依存追加・既存生成器/runner/mixture_doe の変更・commit は行わない。

## 10. オーケストレータへの確認事項

- 実機 campaign の prefix / data_root、生成 commit、固定 test・baseline の hash を実行前に確定する。
- 既存12点を実機初期履歴に使う場合、生成 version が一致する範囲を確認する。不一致なら遡及専用とする。
- 校正結果のどの程度の過信を実機開始の見送り条件とするか(まず頂点/内部別の結果を確認する)。

## 11. オーケストレータの回答(2026-09-26)

- §10-1: 実機 campaign は prefix `a24`、data_root `data`、size 24、max_offset 20、baseline_dir
  `experiments/mixture_doe/base_target`(既存の target 固有基準)、固定 test は `data/s24_{layered,structured,spaghetti}`。
  dataset_version は実体化時の HEAD。既存 DoE データ(d24_p*)は生成器の family 実装が変わっていないため
  `--allow-version-mismatch` の従来規約で同一版とみなす。
- §10-2: 既存 12 点 × 3 seed(`obs_xt.json`)を実機 campaign の初期履歴として使う(registry は specs から作る)。
- §10-3: 実機開始の見送り条件は「滑らかな低ノイズ条件で内部候補の 95% 区間被覆率が 80% 未満」。頂点の過信は
  記録するだけで見送り条件にしない(関心は内部)。
- 実装順: T1 → T2 → T3(§9)。T1 は OpenCode 用 worktree で、pretrain の T1 と並行して進める。
