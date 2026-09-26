# CFG 生成器 v2 コードレビュー報告書（2026-09-15）

## 1. 総評

**生成器と採用機構の主要部分は動作しているが、実験集計の完全性を保証するには修正が必要である。指摘は高 1 件、中 4 件、低 1 件。** 特に `summarize_c.py` が不完全な test スコア・学習 seed を通常の結果として集計する点は、モデル比較の結論を変え得るため、再利用前に対応したい。

対象は `feat/controlled-eval`（`a51cb002a9651248b03205630b4a48a0d4f99fb1`）から `feat/generator-v2` の HEAD（`1439828836ea18c45f0694d54813395e22ff3ca8`）までの三点差分、17 コミット。設計は `generator_v2.md` と `generator_v2_detailed.md`、後者 §12 の D1–D8 を優先した。D1 により削除された bucket WF／タスク16は欠落として数えない。D4 の checkout 検証・外部 CFG ファイルの省略、D7 の通常テスト縮小も承認済みの仕様として扱った。

レビュー開始時から `pyproject.toml` と `uv.lock` に未コミット変更があった。これらは対象ブランチのコミット差分には含まれず、変更・復元していない。コード・テスト・コミットは変更しておらず、追加成果物は本報告書だけである。再現スクリプトと比較用出力は `/tmp` に作成した。

### 検証結果

パッケージをインストールしないため `UV_NO_SYNC=1`、既定 uv キャッシュが書き込み不可のため `UV_CACHE_DIR=/tmp/gr-review-uv` を指定した。Python は 3.13.12。最初の既定キャッシュでの pytest／ty 起動はキャッシュのロック作成に失敗し、その後の下記実行は成功した。

| 確認 | 結果 |
|---|---|
| `UV_CACHE_DIR=/tmp/gr-review-uv UV_NO_SYNC=1 uv run pytest -q` | **533 passed in 6.64s** |
| `UV_CACHE_DIR=/tmp/gr-review-uv UV_NO_SYNC=1 uv run ty check` | **All checks passed!** |
| `UV_CACHE_DIR=/tmp/gr-review-uv UV_NO_SYNC=1 GR_SLOW_TESTS=1 uv run pytest -q tests/test_generate_v2.py` | **1706 passed in 14.91s** |
| `git diff --check` | 出力なし |
| 旧ブランチとの v1 再生成比較 | fixture 条件の manifest と全サンプルがバイト一致。固定 code メタデータ付きでも一致 |
| reducibility の独立照合 | 4 頂点・自己辺なしの有向グラフ全 4096 個を確認。全頂点到達可能な 2432 個は支配関係による判定と全件一致、残りは ValueError |
| structured の追加検証 | 固定乱数源 `Random(20260915)` で 1000 組の spec を選択。成功 798、normalize 拒否 114、GenerationRejected 88。成功全件で到達性・独立判定による reducibility・辺重複なし・予算・入次数上限・出次数上限を確認 |
| 手元の C 実験データ | `summarize_c.CELLS` の 11 セル・各構成・seed 0,1,2 の全スコアを走査。test ID 集合／件数と token trace 長／n_tokens の不一致なし |

追加 structured 検証の母集団は n∈{12,16,24,32,48}、L∈{0,1,2,4}、D=0 または 1..L、G∈{0,1,3,5}、merge∈{2,3,4}、branch=width=4、span∈{short,long,uniform}。生成 seed は試行番号 0..999。独立判定では `networkx.immediate_dominators` から「宛先が始点を支配する辺」を除去し、残りが DAG かを調べた。全パラメータの証明ではない。

手元の `data/` と `runs/` はコミット差分外の補助証拠である。既存 C の数値がスコア欠落によって誤っていたという証拠は得ていない。以下では、実際に再現した不具合と、この確認限界を区別する。

## 2. 指摘

### R1【高】C 集計が test ID・学習 seed の欠落を許し、不揃いの比較を正常な結果として出す

**該当箇所:** `training/summarize_c.py:169`、`:174`、`:184`。接続先は `training/controlled_eval.py:372`、`:377`、`:409`、`:69`。

**根拠:** 新しい C 集計は `summarize(..., by_bucket=False)` の既定経路を呼ぶ。厳密な ID 検査とスコアファイル必須化は `by_bucket=True` の場合だけであり、通常経路は欠けた run をスキップし、paired Δ は sample_id の共通部分だけを使う。Markdown の冒頭には要求した `seeds=[0,1,2]` が表示されるが、各行に実際の seed 数は出ない。全 run がなくても空 summary は例外にならない。さらに、発生した検証エラーも包括的な `except Exception` により pending 扱いとなる。

**影響:** 片方のモデルから難しい sample や失敗した学習 seed が落ちると、全体 NLL の比較母集団が異なる。1 seed・1 sample の Δ が「符号一致 yes」と表示され得る。これは `summarize_c.py` の新規呼び出し側の問題であり、旧 B 経路の仕様を一律変更する必要はない。

同じ呼び出しのため C の結果には bucket 集計と retained/raw も残らない。手元の `runs/c_summary.json` には str2lay の n_test=198 はあるが raw=200／除外=2 がなく、depth は 214/216、merge は 227/230 の分母が失われている。各 token bundle の meta にはこれらの件数が存在する。

**再現手順:** リポジトリ直下で次を実行する。出力は `/tmp` のみ。前半が R1、後半が R5 の再現である。

```bash
UV_CACHE_DIR=/tmp/gr-review-uv UV_NO_SYNC=1 uv run python - <<'PY'
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from cfg_reducer.model_input import build_vocab
from training.controlled_eval import summarize
from training.data_utils import write_jsonl

with TemporaryDirectory(prefix='gr-review-') as tmp:
    root = Path(tmp)
    bundle = root / 'tokens'
    bundle.mkdir()
    vocab = build_vocab(1)
    (bundle / 'vocab.json').write_text(json.dumps(vocab))
    tokens = [vocab[x] for x in ('BOS', 'KIND_ENTRY', 'EOS')]
    write_jsonl(bundle / 'train.jsonl', [{'sample_id': 'train', 'tokens': tokens}])
    write_jsonl(bundle / 'val.jsonl', [])
    write_jsonl(bundle / 'test.jsonl', [
        {'sample_id': sid, 'tokens': tokens} for sid in ('a', 'b')])
    index = {'version': 1, 'samples': {
        'train': {'split': 'train', 'bucket': None},
        'a': {'split': 'test', 'bucket': 'x'},
        'b': {'split': 'test', 'bucket': 'x'}}, 'exclusions': []}
    (bundle / 'evaluation_index.json').write_text(json.dumps(index))
    def score(sid, value):
        return {'sample_id': sid, 'n_tokens': 2, 'nll': 2*value,
                'nll_per_token': value, 'token_nll': [value, value]}
    for cfg, rows in (
        ('base', [score('a', 1), score('b', 9)]),
        ('mask', [score('a', .5)]),
    ):
        run = root / f'x_{cfg}_n24_s0'
        run.mkdir()
        write_jsonl(run / 'test_scores.jsonl', rows)
    args = (root, 'x_', 24, bundle, ['base', 'mask'], 'base')
    report = summarize(*args, [0, 1, 2])
    print(report['n_test'], report['summary'])
    # 完全な2件のtestに対しmaskは1件、学習seedも0のみ。
    # n_test=2、base NLL=5.0、mask NLL=0.5、paired n=1、
    # mean_delta=-0.5、all_same_sign=True が正常に返る。
    try:
        summarize(*args, [0], by_bucket=True)
    except ValueError as exc:
        print('strict:', exc)  # score IDs must match all retained test IDs

    # R5: IDは揃えるが、本来2 tokenのtraceを1 tokenで切る。
    for cfg in ('base', 'mask'):
        rows = [dict(score(sid, 1), n_tokens=1, nll=1, token_nll=[1])
                for sid in ('a', 'b')]
        write_jsonl(root / f'x_{cfg}_n24_s0/test_scores.jsonl', rows)
    report = summarize(*args, [0], by_bucket=True)
    print(report['runs']['base'][0]['by_bucket']['x'])
    # retained=2/2、n_tokens=2（本来4）、NLL=1.0として受理。
PY
```

**推奨する修正:** C 用集計で、要求した全構成・全 seed の存在、test ID の重複なし・完全一致を集計前に必須確認する。bucket 出力の有無と完全性検査を分離し、ID セルにも適用する。不足がある場合は通常の比較表を生成せず、部分集計を許すなら明示オプションと実際の seed／sample 件数を出す。pending は未到着ファイルに限定し、破損・検証エラーは区別する。OOD 出力には retained/raw と除外理由を引き継ぐ。R4 解消後は bucket 指標も接続する。

### R2【中】abrupt の同一 atom への反復挿入で join が余分に残り、実現可能な予算を全棄却する

**該当箇所:** `cfg_reducer/families/structured.py:67`、`:162`、`:172`、`:255`。誤った期待値を固定しているテストは `tests/test_structured_planner.py:65`–`:70`。

**根拠:** 設計 §4.1 は通常経路を残す `if(abrupt)` の挿入を追加2ノードと規定する。反復時の planner は `seq(if, if, ..., atom)` に平坦化するが、lower の join 共有は直後の child が atom の場合だけである。G 個を同じ atom に挿入すると、最後以外の G−1 個に余分な join が残り、合流中継を除いて追加ノードは `3G−1` になる。既存テストも L1/G3 の 14 ノードを期待しており、仕様の 12 ノードを検査していない。

**再現手順:** 次は L=0、G=5、merge=99 で合流中継の交絡を除いた例。予算上限13に対して必要数が17と計算される。

```bash
UV_CACHE_DIR=/tmp/gr-review-uv UV_NO_SYNC=1 uv run python - <<'PY'
from random import Random
from cfg_reducer.generator_types import GeneratorSpec, GenerationRejected
from cfg_reducer.families.structured import Structured, plan_structure, lower_structure
s = GeneratorSpec('structured', 12,
                  {'loop_count': 0, 'goto_count': 5, 'merge_degree': 99})
t = plan_structure(s, Random(0))
print([c.kind for c in t.children])
print(len(lower_structure(t, 99).nodes))  # 17、設計上は3+2*5=13
rejected = 0
for seed in range(100):
    try:
        Structured().generate(s, Random(seed))
    except GenerationRejected:
        rejected += 1
print(rejected)  # 100
PY
```

**影響:** G≥2 が同じ atom に挿入される spec の採用率・実現分布が変わる。上の例は仕様どおりの条件鎖なら13ノードで構成できるにもかかわらず成功ゼロ。成功 CFG の reducibility や abrupt 構文数が壊れるという証拠ではない。

**推奨する修正:** 連続する guarded abrupt の通常側を、次の条件または元 atom に直接つなぎ、不要な join を作らないよう planner/lower 間の契約を統一する。単発だけでなく反復挿入と、その後の atom 展開でもノード数を検証する。乱数消費や出力構造が変わるため修正後は generator version を更新する。

### R3【中】plugin の明示 entry が descriptor 境界で失われ、測定側の暗黙の先頭 entry と食い違う

**該当箇所:** `cfg_reducer/generate_v2.py:48`、`:70`、`cfg_reducer/dataset.py:337`、`cfg_reducer/buckets.py:29`。

**根拠:** `CFGShape` は entry を独立フィールドとして持ち、dispatch はその存在だけを検査する。ところが戻り値は nodes のみで、builder→Candidate→measure_realized は `nodes[0]` を entry とみなす。詳細設計 §2 の共通契約は entry 存在・entry からの到達性を要求しているが、`entry == nodes[0]` の制約を明記・検証していない。structured 自身の ENTRY 先頭保証だけでは他の plugin の契約を閉じられない。

**再現手順:** nodes の名前・配置順は N00,N01 のまま、入口を N01 とする、全ノード到達可能な plugin。

```bash
UV_CACHE_DIR=/tmp/gr-review-uv UV_NO_SYNC=1 uv run python - <<'PY'
from tempfile import TemporaryDirectory
from cfg_reducer import GraphEngine
from cfg_reducer.generator_types import CFGShape, GeneratorSpec
from cfg_reducer.family_registry import register_family
from cfg_reducer.generate_v2 import generate_cfg_v2, descriptor_for, spec_to_json
from cfg_reducer.dataset import build_dataset
from cfg_reducer.buckets import measurement_acceptor
class Plugin:
    name = 'review_entry'
    def normalize(self, spec): return spec
    def generate(self, spec, rng):
        return CFGShape(('N00', 'N01'), (('N01', 'N00'),), 'N01')
register_family(Plugin())
s = GeneratorSpec('review_entry', 2)
print(generate_cfg_v2(GraphEngine(), seed=0, spec=s))  # 正常終了
with TemporaryDirectory(prefix='gr-review-') as out:
    build_dataset(out, {'test': (0, 1)}, {'spec': spec_to_json(s)},
                  'review', descriptor_for(s), accept=measurement_acceptor)
# ValueError: all CFG nodes must be reachable from entry
PY
```

**影響:** 組み込み3族には影響しない。Protocol の型を満たし明示 entry から到達可能な外部族が、生成単体では成功して dataset CLI では失敗する。A7-1 の分岐なし追加という配置はできているが、plugin の動作契約は不足している。

**推奨する修正:** 先頭 entry を共通の明示契約として validation で強制するか、entry を Candidate まで渡すかを決める。前者なら engine 変更前に説明的な ValueError とし、後者なら entry が先頭以外の plugin で生成→測定→再生を確認する。

### R4【中】通常の token 準備では index を作らず、ID／balanced-k の bucket 評価につながらない

**該当箇所:** `training/prepare_tokens.py:30`、`:81`–`:94`、`:124`、`training/controlled_eval.py:354`、`training/summarize_c.py:42`。

**根拠:** `evaluation_index.json` と selection／realized／bucket の引き継ぎは `--test-dataset` 経路だけである。単一 dataset の通常経路は v2 manifest に bucket があっても捨てる。一方、`--by-bucket` は index を必須とする。詳細設計 §9 の「balanced-k を同じ採用分布で評価する」経路に、この組み合わせを埋める処理がない。

**再現手順:** 手元の本番 bundle では以下が `FileNotFoundError: data/tok_c_balanced/evaluation_index.json` となった。

```bash
UV_CACHE_DIR=/tmp/gr-review-uv UV_NO_SYNC=1 uv run python -m training.controlled_eval \
  --runs runs --prefix c_balanced_ --size 24 --tokens data/tok_c_balanced \
  --configs base,mask --seeds 0,1,2 --by-bucket
```

本番データに依存せず、任意の train/val/test を持つ v2 dataset を `/tmp/ds` に生成し、`uv run python -m training.prepare_tokens --dataset /tmp/ds --window-from train --out /tmp/tok` を実行しても index は作られない。手元では5つの ID bundle 全てで不在を確認し、6つの cross-dataset bundle には存在した。

**影響:** ID 対 OOD の同じ bucket の比較や balanced-k の bin 別集計が標準の準備手順だけではできない。R1 の `by_bucket=True` を単に追加するだけでは C の全セルを集計できない。

**推奨する修正:** 旧 token／vocab／meta の互換性を維持しつつ、単一 v2 dataset の selection と test ID から index を生成する追加経路を設ける。既存ファイルを変えない sidecar 追加または明示オプションが候補。暫定的には source と同じ dataset を `--test-dataset` にも指定できるが、max_len／容量・除外基準が旧経路と変わるため、既存 ID モデルとの互換性を確認してから使う。

### R5【中】by-bucket の完全性検査は ID までで、切れた token trace を完全な採点として集計する

**該当箇所:** `training/controlled_eval.py:95`、`:316`–`:335`、`:378`–`:387`。

**根拠:** score の ID 集合は厳密に検査されるが、`len(token_nll) == len(tokens)-1 == n_tokens` を確認しない。REF 集計は `zip(tokens[1:], token_nll)` なので短い trace の末尾を無言で落とす。nll／nll_per_token と trace の整合、ref_pos／ref_k／ref_correct の対応も検査されない。

**再現手順:** R1 の再現スクリプト後半。2 sample、それぞれ2 target token の test に対して各 trace を1 tokenに切り、対応する n_tokens/nll も1にすると、`by_bucket=True` でも retained=2/2、n_tokens=2、NLL=1.0 として正常終了する。本来の採点対象は合計4 tokenである。

**影響:** 切断・別バージョンの採点ファイル・REF メタデータ不整合が、同じ sample_id を持つ場合に検出できず、NLL／edge accuracy の分母が縮む。現行 trainer がこの壊れた trace を通常生成することは未確認。今回走査した C の実スコアには長さ不一致はなかった。

**推奨する修正:** 集計入口で token 長・n_tokens・trace 長・有限値・nll 合計を検証し、REF の位置／実 token／配列長も一致させる。検証済みの同じ score を全体集計と bucket 集計で共有する。

### R6【低】invalid_feature を棄却しても sidecar に NaN を書ける

**該当箇所:** `cfg_reducer/dataset.py:359`–`:363`、`:383`、`:405`、`cfg_reducer/buckets.py:138`–`:142`。

**根拠:** 採用結果は accepted と reason の整合だけを確認し、realized の JSON 適合性を検査しない。JSON 出力は `allow_nan=False` を指定していない。bucket_acceptor も非有限値に対する invalid_feature 決定に、元の realized を保持する。詳細設計 §2 の「NaN/Infinity を出さない」に反する。

**再現手順:** 正常な1ノード generator と、非有限特徴を invalid_feature として棄却するフックを使用する。

```bash
UV_CACHE_DIR=/tmp/gr-review-uv UV_NO_SYNC=1 uv run python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
from cfg_reducer.dataset import build_dataset
from cfg_reducer.buckets import AcceptDecision

def generate(engine, *, seed):
    engine.add_node('N00')
    return ['N00']
def accept(candidate, state):
    return AcceptDecision(False, 'invalid_feature', None,
                          {'mean_offset': float('nan')})
with TemporaryDirectory(prefix='gr-review-') as out:
    build_dataset(out, {'test': (0, 1)}, {}, 'review', generate, accept=accept)
    print((Path(out) / 'rejections.jsonl').read_text())
# realized: {"mean_offset": NaN} が保存され、生成処理は成功する。
PY
```

**影響:** 拡張フックや将来の測定処理が不正値を返した場合、厳密な JSON reader がログを読めない。組み込みの正常な `features_for` 出力で NaN が出る例は得ておらず、通常運用への影響は限定的。

**推奨する修正:** 不正値を棄却ログにどう表現するか（欠測 null と別診断など）を定義し、非有限値を残さない。採用結果の検証と `allow_nan=False` を併用し、測定バグを単なる採用失敗として隠さない。

## 3. 要求 1〜8 ごとの充足状況

| 要求 | 判定 | 根拠と限界 |
|---|---|---|
| 1. 構造分布シフト・plugin 追加 | **部分** | params は族固有 dict、registry／descriptor／CLI は族名分岐を増やさず追加可能。toy の追加テストあり。R3 の entry 契約と R4 の評価接続が不足。各因子の「独立」は設定を別々に持てる意味では成立するが、予算共有による実現分布の交絡は残る。span が MetaGraph offset を保証しない点は D3 の承認仕様 |
| 2. 決定性・再現性・後方互換 | **部分** | 組み込み族の RNG と順序制御、hashseed=1/77 の全 JSON 比較、plan 変更時の候補 UUID 不変は確認。v1 の実際のバイト比較も成功。R6 の非有限 JSON と R3 の外部族契約は残る。provenance からの UUID 決定性は確認したが、D4 の UUID 検査はコード内容の同一性証明ではない |
| 3. layered による v1 完全再現 | **充足** | 通常・拡大テストで返却 ID、nodes、edges、縮約 sketch 一致。所定 spec は width=3、対応 edge_prob、L=max(1,int(.15n))、G=max(1,int(.1n))、uniform。n1/2 の鎖、n3 の正 count 時 ValueError、両 count=0 の DAG も確認 |
| 4. structured の構成的保証 | **部分** | 成功 CFG の到達性・reducibility・重複辺なし・整数予算・入次数上限を通常／拡大／追加検証で確認。planner の L/G/D の数え上げテスト、バグ例外の伝播あり。ただし R2 が追加2ノード契約と実現可能予算を損なう。CFG のループ構文数と縮約後 n_loops/max_depth の一致は設計上の保証対象ではない |
| 5. reducibility | **充足** | 元の pred を保持して header 除去後の SCC に再帰する。最大 SCC の内部に隠れた多重入口、同じ header への複数辺、自己ループ、到達不能、不明端点、順序反転のテストあり。独立な全4096グラフ照合でも矛盾なし |
| 6. 採用機構・manifest | **部分** | built-in bucket/ranges は状態を変更せず、counts を読むだけ。cross-dataset→seen の順で同型照合し、採用後だけ seen/counts を更新。重複も測定するため per_bucket に計上でき、GenerationRejected は unbucketed。attempts=accepted+rejected、未充足、理由別件数と sidecar はテストで整合。R6 の JSON 境界は不足。quota なしで complete=True は「指定 quota の未充足なし」であり、採用数が正との保証ではない |
| 7. OOD 配線・評価 | **部分** | cross-dataset 窓と max_len は source train のみ、capacity=2*max_len。窓超過を先に分類し、長さ超過・分母・source hash・採用 index を記録。by-bucket の ID join、全構成の ID 一致、ID sort 後の paired Δ は正常。R1/R4/R5 により C 全体の評価経路は不十分。bucket WF を出さない点は D1 適合 |
| 8. 運用制約 | **充足** | Python3.13/uv で検証、追加 runtime 依存なし。対象差分で pyproject.toml／uv.lock／train_ar.py は不変。torch/matplotlib の軽量 import テスト成功。今回もインストール、コード変更、テスト追加、コミットなし |

### 決定性と構造保証を支持する実装上の根拠

- layered は v1 と同じ順序で乱数を消費し、重複辺でも再抽選しない。choice の候補は順序付きの nodes/layers、出力 edges は sort される。
- structured の atom 候補は前順、構文候補は固定順、合流 source は記号番号順。padding は候補を sort してから渡された RNG だけで choice する。src/dst の既存次数を変えず、挿入ノードの入出次数は1。辺の細分化なので到達性と reducibility を保存し、最終ノード数は budget と一致する。
- spaghetti は基底 edges を維持し、`u != v`、既存辺なし、`v != entry`、`u != EXIT` の全候補を sort して非復元抽出する。追加数は `min(floor(rate*|E_base|), 候補数)`。rate=0 の完全一致と候補制約はテスト済み。追加後の次数上限と irreducible の強制は仕様外である。
- reducibility は set を Tarjan に渡すが、既存 `algorithm.py:78` と `:102` の内部で開始頂点・後続頂点を sort する。返された SCC も sort するため、set の反復順は結果に漏れない。単に呼び出し引数が set であることを不具合とは判定しなかった。
- `structure_features.py` の23特徴は旧 training 実装から計算式を変えず移設されている。subgraphs の巡回順に依存する出力列は作らず、集計結果だけを返す。**max_in_degree は MetaGraph の特徴**であり、structured の CFG merge_degree と同一ではない。degree OOD の ranges が前者を使うのは詳細設計 §9 に明記された仕様。
- `_build_selected` は重複に対しても accept を呼ぶが、built-in は副作用を持たず counts を増やすのは保存時だけである。重複理由は bucket_full／outside_plan より優先される。棄却された CFG を seen に入れないことは専用 fixture で確認されている。

### v1 バイト互換の根拠と追試手順

`generate.py`、`store.py`、`algorithm.py`、`engine.py`、`motif.py`、`metagraph.py` は対象差分に変更がない。`build_dataset` は accept/exclude が無効なら追加前の本体に入る。checked-in fixture は n=8、edge_prob=.3、train=[0,4)、val=[4,6)、version=fixture、code なし。

レビューでは `git archive feat/controlled-eval` を `/tmp/gr_review_base` に展開し、旧モジュールと現モジュールを別プロセスで import して、この条件の dataset を `/tmp/gr_review_old` と `/tmp/gr_review_new` に生成した。再生成用スクリプトは以下と同じ内容で、`uv run python /tmp/gr_review_legacy.py <ソースroot> <出力dir>` として実行した。

```python
import sys
sys.path.insert(0, sys.argv[1])
from cfg_reducer.dataset import build_dataset
for suffix, code in (('', None), ('_code', {'commit': 'fixed', 'dirty': False})):
    build_dataset(sys.argv[2] + suffix,
                  {'train': (0, 4), 'val': (4, 6)},
                  {'num_nodes': 8, 'edge_prob': .3}, 'fixture', code=code)
```

両出力ディレクトリの `diff -rq`（code なし／ありの両方）、旧 manifest と `tests/fixtures/generator_v1_manifest.json` の `cmp` は全て差分なしだった。manifest だけでなくサンプル内容も比較した。ただし全既存 dataset の保存済みバイトを総当たりしたわけではない。

## 4. テストの空白

§10 のタスクと、D1/D7 適用後の受け入れ条件を基準にした。

| タスク | 担保済みの部分 | 空白・追加してほしい確認 |
|---|---|---|
| 1–2: registry／descriptor | 冪等 normalize、独自 params 往復、未知／重複族、異なる族の UUID、形状不正、登録禁止 | entry が先頭以外の族の end-to-end（R3）。任意の plugin の冪等性・乱数源・副作用までは Protocol が保証しないため、plugin 作者向け共通契約テストが必要 |
| 3: layered | D7 の通常・拡大マトリクス、n3、RNG を消費しない小サイズ | 拡大ケースは通常 CI では走らない。今回は拡大側も実行済み |
| 4: reducibility | 設計の具体例、順序反転、不正入力 | 一般グラフとの独立な照合は既存 suite にない。今回の4頂点全列挙で補助確認した |
| 5–6: templates／planner | 単発 abrupt 全種類、while/do の continue、nested context、merge=2 の手書き期待辺、L/G/D マトリクス | 反復 abrupt の追加2ノード契約。現テストはむしろ R2 の余分な join を固定。branch 展開前後での guarded abrupt 接続の期待辺も不足 |
| 7: structured／padding | D7 の100 seed×3サイズ、予算、入次数、辺数増分、padding 候補の手書き例 | many-seeds は L2/G1/merge2 が中心。G3以上、L4、予算境界を跨いだ成功 CFG の組合せ、最終展開後の構文契約は薄い。padding の出次数保存を明示検査していない。今回の追加検証は一部を補うが恒久テストではない |
| 8–9: spaghetti／features | rate0、辺数、ENTRY/EXIT 制約、到達性、23特徴の既存テスト | 候補集合の確率分布を統計検定していないが、固定された `rng.sample` 使用なので受け入れ上の重大な空白とはしない |
| 10–12: bucket／builder／CLI | 境界、欠測・非有限の bucket 判定、状態不変、重複優先、seen 非汚染、孤立点を含む一般同型 API、ranges、未充足exit2、provenance不一致拒否 | 非有限 realized を実際の JSON 出力まで通す検証（R6）。D4 により「同じ version だが実装が変わった」検出は仕様として未実装 |
| 13: manifest 互換／hashseed | v1 固定 manifest、3族×hashseed=1/77 の全 JSON、quota変更でもUUID不変 | fixture テストは code なしで、サンプルは読込／ID確認のみ。旧ブランチの sample bytes との比較は suite にない。今回それを code あり／なしで補助確認 |
| 14: prepare | source train の窓・容量、target置換、byte同一vocab、over_window/over_length、index／分母、重複ID、空train | 単一 v2 dataset／balanced-k の index 出力（R4）。capacity ちょうどの境界、同時に窓・長さ超過する sample の優先理由、val 側の長さ超過は明示 fixture が薄い |
| 15: controlled_eval | 手計算の重み付きNLL、空bucket、ID不一致、seed別集計、source train頻度、旧CLI互換 | 切断／不整合trace（R5）。`summarize_c.py` を介した欠落seed／欠落ID／分母の保存（R1）は未テスト。まとめスクリプト自身のテストがない |
| 17: integration | built-in layered→structured の除外・token化・合成score集計、旧sample読込、軽量import | toy plugin は生成確認までで、その toy 自体を exclusion→tokenize→score まで流す経路ではない。toy の特有契約不整合を end-to-end で拾えない |

`GR_SLOW_TESTS=1` 限定なのは layered の seed 10–49 と p=0/1 を含む拡大マトリクス、および structured の seed 100–499。今回これらは通過したが、abrupt反復・入口情報・C集計の欠落検証を増やすものではない。seed 数だけを増やしても R1–R6 は埋まらない。

## 5. 質問

1. **plugin の entry は必ず nodes[0] とするか。** 現在の builder コメントの制約を共通契約・validation に昇格するのか、明示 entry を保持するのかを決めたい（R3）。
2. **abrupt 挿入の追加2ノードを維持するか。** 現在の反復ケースのテスト期待値は仕様と矛盾する。現実装を意図したなら、必要予算と実現分布の仕様を改める判断が必要である（R2）。
3. **C の正式な集計入口は `summarize_c.py` か、`controlled_eval --by-bucket` か。** 現在は前者が検証・分母・bucket 表を迂回し、後者は ID bundle を読めない。完全な比較条件を一箇所で保証したい（R1/R4）。
4. **D4 の version 運用をどう固定するか。** `load_references` の sample_id は入力 provenance だけから算出される。同じ name/version/seed/config のまま plugin 実装を変えれば、別 CFG でも UUID 検査は通る。「UUID 一致がコード版の同一性を直接確認する」という D4 の説明は強すぎる。コード変更ごとの version 更新を必須運用とする理解でよいか。実際の source/target 間でこの問題が起きたかは未確認。承認済み D4 を無視して checkout 検査を追加すべきという指摘ではない。
5. **C のセル別 WF はどの生成条件を指すか。** D1 に従い bucket WF は評価しなかった。無条件生成は target test の差し替えだけでは条件が変わらないため、source run の全体 WF と OOD target に条件付けた WF を混同しない説明が必要である。別途の Colab 再採点／生成スクリプトの動作はこの差分では未確認。
6. **「因子を独立に振る」の評価単位をどこまで要求するか。** 実装は要求 L/G/D と次数上限を独立のキーとして持つが、有限予算・同型排除・ranges によって realized 分布は連動する。詳細設計どおり同じ他因子の active/ranges と除外率を併記する基準でよいか。n12/16/32/48 の family OOD 学習実験と、因子間交絡の統計的な解消は本レビューでは未確認である。

## 6. 対応方針(進行役 Claude、2026-09-15)

指摘はすべて妥当と判断した。対応の順序は「混合族データの実験(既存の単族結果と同じ生成器・
同じ target で比較する)→ レビュー修正をまとめて適用」とする。生成器の出力を変える R2 を
先に入れると、単族結果との比較条件が崩れるため。

| 指摘 | 対応 | 時期 |
|---|---|---|
| R1(高)C 集計の完全性 | `summarize_c.py` は集計前に全構成・全 seed・test ID の完全一致と trace 長を必須検査し、欠落は表を出さずに明示する。`by_bucket` の厳密検査を ID セルにも適用し、retained / raw と除外理由を引き継ぐ。既存の C 表は reviewer の走査で欠落なしと確認済み(§1)なので数値の訂正は不要 | 混合族実験後 |
| R2(中)反復 abrupt の余分な join | 設計の「追加 2 ノード」契約に合わせ、連続する guard の通常側を次の条件または元 atom に直結する。乱数消費と構造が変わるため generator version(git commit)が変わることを文書化し、以後の dataset は新 version で生成する。既存 C データは旧 version のまま参照可能 | 混合族実験後 |
| R3(中)entry 契約 | `generate_cfg_v2._validate_shape` で `entry == nodes[0]` を共通契約として強制し(説明的な ValueError)、設計 §2 に明記する | 混合族実験後(組み込み族は影響なし) |
| R4(中)単一 dataset の index | `prepare_tokens` の単一 dataset 経路でも v2 manifest から `evaluation_index.json` を書く(旧 token / vocab / meta は不変)。balanced-k の bin 別集計を標準経路に載せる | 混合族実験後 |
| R5(中)trace 完全性 | 集計入口で `len(token_nll) == len(tokens) − 1 == n_tokens`、有限値、REF メタデータの整合を検証し、全体集計と bucket 集計で同じ検証済み score を使う | R1 と同時 |
| R6(低)NaN の sidecar | `allow_nan=False` と採用結果の検証。非有限 realized は `null` + 理由で記録 | R1 と同時 |

質問への回答: (1) entry は nodes[0] を共通契約とする。(2) 追加 2 ノード契約を維持し R2 を修正する。
(3) 正式な集計入口は `summarize_c.py` とし、厳密検査を `controlled_eval` 側の共通関数に置いて
両方から使う。(4) 生成器コードを変えるたびに version(git commit)が変わる運用を必須とし、
AGENTS.md に明記する。同一 version での実装変更は禁止。(5) セル別 WF は source モデルの無条件
生成 WF(モデルの性質)であり、target には条件付けない。表の見出しで明示する。(6) 詳細設計
どおり「同じ他因子の active / ranges と除外率を併記」を基準とする。

## 7. 対応記録(2026-09-15)

| 指摘 | コミット | 内容 |
|---|---|---|
| R1 / R5 | `4a1fc9f` | `controlled_eval.validate_scores`(trace 長・有限性・nll 整合・REF メタデータ・test ID の完全一致)を全経路に適用。`summarize_c` は strict 既定、pending / error を区別、retained / raw / excluded を併記、21 セル + target 別比較表 |
| R3 / R4 / R6 | `95ac0f2` | `entry == nodes[0]` を共通契約として強制。単一 dataset 経路でも `evaluation_index.json` を出力(token / vocab / meta はバイト不変)。manifest / sidecar を `allow_nan=False`、非有限 realized は `null` + `invalid_features` |
| R2 | `97ce3eb` | 連続 guard の join 共有(追加 2G ノード)。structured の出力が変わるため以後の dataset は新 version。再現例は 13 ノード・100/100 成功 |

いずれも Codex(gpt-6-astra、low)が実装し、進行役が pytest / ty / diff-check と再現例で受け入れた。
残る「テストの空白」(§4)のうち、plugin 作者向け共通契約テスト・padding の出次数保存・
反復 guard 後の atom 展開の期待辺は R2 のテスト追加で一部を担保。残りは次の実装機会に回す。
