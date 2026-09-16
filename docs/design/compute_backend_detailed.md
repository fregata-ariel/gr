# 計算バックエンド・ルーター詳細設計(`training/runner/`)

作成: 2026-09-16。概要と決定は `compute_backend.md`、`handoff_questions.md` Q8 / A8。
状態: 設計確定(実装タスク §9)。Codex が利用上限のため、本設計は Claude が執筆し、
実装は OpenCode(bounded なタスク)または Claude が行う。本書の §10「決定」が上位節より優先。

## 0. 目的(このプログラムに要求していること)

1. **計画と実行先の分離**: 実験計画(どの run を、どのバンドルで、どの `train_ar` 引数で
   学習し、どのバンドルで再スコアするか)を `Plan` として JSON で持ち、実行先(Colab /
   ローカル GPU コンテナ)は起動時の方針で選ぶ。計画側のコードに実行先の名前が現れない。
2. **現行フローの忠実な再現**: Colab アダプタは現行の `run_sweep.sh` と同じリモートパス、
   同じラッパー本文(byte 同一)、同じタイムアウト、同じ冪等規則、同じログ標識を用いる。
3. **冪等・再開可能**: run は `runs/<name>/test_scores.jsonl` の有無で完了判定。失敗した run
   の部分ディレクトリは削除。セッション喪失後は同種の実行先を再取得して途中から続ける。
4. **実行先の記録**: 各 run に `backend.json`(種別、セッション/イメージ、GPU 名、時刻、plan_id)
   を残し、実行先が混ざった場合に判別できる。
5. **オフライン検証**: `--dry-run` で全コマンド列と生成スクリプト本文を出力し、Colab が
   使えなくても動作を検証できる。テストは FakeBackend と注入したコマンド実行器のみを使う。
6. **制約**: 標準ライブラリのみ。`torch` も `cfg_reducer` も import しない(ローカル後処理は
   `uv run python -m training.eval_samples` を subprocess で呼ぶ)。凍結 dataclass + tuple、
   JSON はキー昇順。テストは torch 非依存で速い。

## 1. モジュール構成

```
training/runner/
  __init__.py    再エクスポート(Plan, Block, TrainJob, RescoreJob, Backend, Router, PlanExecutor)
  types.py       Plan / Block / TrainJob / RescoreJob(凍結 dataclass)、JSON 入出力
  scripts.py     リモートで実行する Python スクリプト本文の生成(学習ラッパー、再スコア)
  plans.py       節目スイープの Plan ビルダー(既存ラッパー 30 本を再生成できる)
  backend.py     Backend Protocol、例外、CommandRunner(subprocess 注入点)、run_command 既定実装
  colab.py       ColabBackend(colab CLI アダプタ、ファイルロック、再試行)
  docker.py      DockerBackend(ローカル GPU コンテナ、ステージング dir、GPU ロック)
  fake.py        FakeBackend(テスト用、呼び出し記録 + 出力ファイルの捏造)と DryRunBackend
  router.py      方針の解決(CLI > 環境変数 GR_BACKEND > 既定 colab)、取得・再取得
  executor.py    PlanExecutor(ブロック単位のステージング、学習、ローカル評価、再スコア)
  __main__.py    CLI: run / sweep サブコマンド
training/docker/Dockerfile   FROM pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime(それ以外なし)
tests/test_runner_*.py
```

## 2. 型(`types.py`)

```python
@dataclass(frozen=True)
class TrainJob:
    name: str                 # run 名。runs/<name>/ と <root>/<name>/ に対応
    epochs: int
    patience: int
    num_samples: int
    constrained_samples: int
    seed: int
    sample_seed: int
    extra: tuple[str, ...] = ()   # モデル構成フラグ。例 ("--ref-legal-mask",)、
                                  # ("--pointer", "--pointer-legal", "--pointer-dist-bias",
                                  #  "--pointer-dist-bias-mode", "context")

    def argv(self, root: str) -> tuple[str, ...]:
        # 順序固定。現行ラッパーと byte 同一になる
        return ("--out", f"{root}/{self.name}", "--epochs", str(self.epochs),
                "--patience", str(self.patience), "--num-samples", str(self.num_samples),
                "--test", f"{root}/test.jsonl", "--constrained-samples", str(self.constrained_samples),
                "--seed", str(self.seed), "--sample-seed", str(self.sample_seed), *self.extra)

@dataclass(frozen=True)
class RescoreJob:
    source_run: str      # 学習済み run 名(同じ Block の TrainJob.name)
    target_bundle: str   # 例 "data/tok_s32_lay2str"。その test.jsonl を採点する
    out_run: str         # 例 "c_s32_lay2str_mask_n32_s0"。runs/<out_run>/test_scores.jsonl

@dataclass(frozen=True)
class Block:
    source_bundle: str             # 例 "data/tok_s32_lay"(train/val/test/vocab/meta を持つ)
    train: tuple[TrainJob, ...]
    rescore: tuple[RescoreJob, ...] = ()

@dataclass(frozen=True)
class Plan:
    plan_id: str
    blocks: tuple[Block, ...]
```

- `bundle_name(path) = PurePosixPath(path).name`(例 `tok_s32_lay`)。リモートの平坦名に使う。
- `plan_to_json(plan) -> str`: `json.dumps(asdict, sort_keys=True, indent=2) + "\n"`。
  `plan_from_json(text) -> Plan`: list は tuple に戻す。未知キーは `ValueError`。
- 検証(`validate_plan`): run 名は Plan 内で一意、`RescoreJob.source_run` は同 Block の
  TrainJob に存在、`out_run` は TrainJob 名と重複しない。違反は `ValueError`。

## 3. リモートスクリプト(`scripts.py`)

```python
def train_wrapper(job: TrainJob, root: str) -> str
```
本文(root = "/content" のとき現行ラッパーと byte 同一。引数は Python の repr で
シングルクォート、`', '` 区切り、末尾改行 1 つ):

```python
import sys
sys.path.insert(0, '/content')
for _m in ('train_ar', 'grammar_mask'):
    sys.modules.pop(_m, None)
import train_ar
train_ar.main(['--out', '/content/c_s32_lay_mask_n32_s0', '--epochs', '300', '--patience', '20', '--num-samples', '400', '--test', '/content/test.jsonl', '--constrained-samples', '400', '--seed', '0', '--sample-seed', '1000', '--ref-legal-mask'])
```

```python
def rescore_script(root: str) -> str
```
現行 `rescore_c.py` と同内容で `/content` を `root` に置換したもの。`<root>/rescore_jobs.json`
を読み、各 job の `config`(= `<run>_samples.json` の `config`)でモデルを組み、`model` を
読み、`test` を採点して `out` に書き、`RESCORED <run> <test> <n>` と最後に `RESCORE-DONE` を出力。

```python
def rescore_jobs(block: Block, root: str) -> list[dict[str, str]]
```
各 RescoreJob について
`{"run": source_run, "model": f"{root}/{source_run}_model.pt", "config": f"{root}/{source_run}_samples.json",
  "vocab": f"{root}/vocab_{B}.json", "meta": f"{root}/meta_{B}.json",
  "test": f"{root}/test_{T}.jsonl", "out": f"{root}/{source_run}__{T}.jsonl"}`
(B = source バンドル名、T = target バンドル名)。JSON はキー昇順。

`train_ar.py` の既定パス: `--train/--val/--vocab/--meta` の既定値 `/content/...` を
`os.environ.get("GR_ROOT", "/content")` 基準に変える(Colab では無変更の挙動)。Docker
バックエンドはコンテナに `GR_ROOT=/work` を渡す。これが `train_ar.py` への唯一の変更。

## 4. Backend Protocol(`backend.py`)

```python
class BackendUnavailable(RuntimeError): ...   # 取得失敗(割当なし、GPU 使用中)
class SessionLost(RuntimeError): ...          # 取得後に実行先が消えた
class RemoteFailure(RuntimeError): ...        # 実行先は生きているがコマンド失敗

@dataclass(frozen=True)
class CommandResult:
    returncode: int
    output: str          # stdout + stderr

CommandRunner = Callable[[Sequence[str], int], CommandResult]   # (argv, timeout_s)

def run_command(argv, timeout_s) -> CommandResult:
    # subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)。
    # TimeoutExpired は CommandResult(124, "timeout after <s>s")

class Backend(Protocol):
    kind: str            # "colab" | "local" | "fake" | "dry"
    root: str            # "/content" | "/work"
    def acquire(self) -> None: ...            # BackendUnavailable
    def release(self) -> None: ...
    def alive(self) -> bool: ...
    def put(self, local: Path, remote: str) -> None: ...      # SessionLost | RemoteFailure
    def get(self, remote: str, local: Path) -> bool: ...      # 無ければ False。SessionLost
    def run(self, script: str, name: str, timeout_s: int) -> str: ...  # 抽出済み出力。SessionLost | RemoteFailure
    def describe(self) -> dict[str, object]: ...              # backend.json に載せる
```

## 5. ColabBackend(`colab.py`)

```python
@dataclass(frozen=True)
class ColabTimeouts:
    new: int = 120; sessions: int = 60; upload: int = 300; download: int = 300
    exec_outer: int = 6000; exec_inner: int = 5400; stop: int = 90

class ColabBackend:
    kind = "colab"; root = "/content"
    def __init__(self, session: str, gpu: str = "T4", *, run_command=run_command,
                 lock_path: Path = Path.home()/".cache"/"gr-runner"/"colab.lock",
                 attempts: int = 36, sleep_s: int = 600, timeouts=ColabTimeouts(),
                 sleep=time.sleep, log=print, script_dir: Path | None = None)
```

- **ロック**: すべての colab CLI 呼び出しは `fcntl.flock(LOCK_EX)` で直列化する(同一ロック
  ファイルを他プロセスの runner も使う)。`acquire()` は `colab new` と直後の `colab sessions`
  を 1 つのロック区間で行う(`colab new` と他コマンドの同時実行がセッション登録簿を壊すため)。
- `acquire()`: `attempts` 回まで `["colab","new","--gpu",gpu,"-s",session]` → `alive()`。
  成功で `SESSION-OK HH:MM:SS` を log。失敗(`Service Unavailable` など)は
  `attempt i/N HH:MM:SS: <出力の最終行>` を log して `sleep(sleep_s)`。全失敗で
  `GIVE-UP HH:MM:SS` を log し `BackendUnavailable`。
- `alive()`: `["colab","sessions"]` の出力に `^\[<session>\]` で始まる行があるか。
- `put(local, remote)`: `["colab","upload","-s",session,str(local),remote]`。returncode≠0 →
  `alive()` が偽なら `SessionLost`、真なら `RemoteFailure(出力)`。
- `get(remote, local)`: `["colab","download","-s",session,remote,str(local)]`。出力に
  `not found`(大小無視)があれば False。他の失敗は put と同じ判定。
- `run(script, name, timeout_s)`: `script_dir/<name>.py` に本文を書き、
  `["colab","exec","-s",session,"-f",path,"--timeout",str(exec_inner)]` を `exec_outer`
  (引数 `timeout_s` があればそれ)で実行。出力から
  `re.compile(r"early stop|test NLL|Error|Traceback|lost|RESCORED|RESCORE-DONE")` に一致する行を
  抽出して返す。出力に `lost`(大小無視)または returncode≠0 のときは `alive()` を確認し、
  偽なら `SessionLost`、真なら `RemoteFailure`。
- `release()`: `["colab","stop","-s",session]`(失敗は無視して log)。
- `describe()`: `{"gpu": gpu, "kind": "colab", "session": session}`。

## 6. DockerBackend(`docker.py`)

```python
class DockerBackend:
    kind = "local"; root = "/work"
    def __init__(self, image: str, staging: Path, *, gpus: str = "all", run_command=run_command,
                 lock_path: Path = Path.home()/".cache"/"gr-runner"/"local-gpu.lock",
                 uid_gid: str | None = None, log=print)
```

- `acquire()`: `lock_path` を `LOCK_EX | LOCK_NB` で取り、取れなければ `BackendUnavailable`
  (ローカル GPU は 1 Plan 専有)。続いて
  `["docker","run","--rm","--gpus",gpus,image,"nvidia-smi","--query-gpu=name","--format=csv,noheader"]`
  で GPU 名を取得(失敗は `BackendUnavailable`)。`staging` を作成。
- `put(local, remote)`: `remote` は `root/` で始まること(違反は `ValueError`)。
  `staging / remote[len(root)+1:]` へ `shutil.copyfile`(親 dir 作成)。
- `get(remote, local)`: 対応ファイルが無ければ False、あればコピーして True。
- `run(script, name, timeout_s)`: `staging/_scripts/<name>.py` に本文を書き、
  `["docker","run","--rm","--gpus",gpus,"--user",uid_gid,"-e","HOME=/tmp","-e","GR_ROOT=/work",
    "-v",f"{staging}:/work",image,"python",f"/work/_scripts/{name}.py"]` を `timeout_s` で実行。
  `uid_gid` の既定は `f"{os.getuid()}:{os.getgid()}"`。出力の抽出は Colab と同じ正規表現。
  returncode≠0 → `RemoteFailure(出力末尾 40 行)`。`SessionLost` は発生しない。
- `alive()`: 常に True。`release()`: ロック解放。
- `describe()`: `{"gpu": <名>, "image": image, "image_id": docker image inspect の Id(取れなければ ""),
  "kind": "local"}`。
- 学習コードもバンドルも Colab と同様に `put` でステージングへ置く(Colab と同じ実行手順に
  するため、コードをマウントで見せる方式は採らない)。

## 7. Router(`router.py`)

```python
DEFAULT_POLICY = "colab"
POLICIES = ("colab", "local", "auto")
def resolve_policy(cli: str | None, env: Mapping[str, str] = os.environ) -> str
    # cli > env["GR_BACKEND"] > DEFAULT_POLICY。不正値は ValueError

class Router:
    def __init__(self, factories: Mapping[str, Callable[[], Backend]], policy: str,
                 *, allow_switch: bool = False, log=print)
    def acquire(self) -> Backend
        # 順序: policy が colab/local ならその 1 つ。auto なら (DEFAULT_POLICY, もう一方)。
        # 各候補で factory().acquire()。BackendUnavailable は次へ。全滅で BackendUnavailable
    def reacquire(self, previous: Backend) -> Backend
        # 同種を再取得。失敗時 allow_switch が偽なら BackendUnavailable、真なら他方を試す
```

## 8. PlanExecutor(`executor.py`)

```python
class PlanExecutor:
    def __init__(self, plan: Plan, router: Router, *, repo_root: Path, runs_dir: Path,
                 local_eval: Callable[[list[str]], int] | None = None,   # 既定 subprocess(uv run ...)
                 log=print, dry_run: bool = False, now=datetime.now)
    def run(self) -> int      # 終了コード: 0 完了 / 2 実行先取得不能 / 3 run 失敗
```

処理(現行 `run_sweep.sh` の再現):

1. `backend = router.acquire()`(失敗 → `GIVE-UP` は Backend 側が log、戻り値 2)。
2. `stage_code(backend)`: `training/train_ar.py` → `<root>/train_ar.py`、
   `training/grammar_mask.py` → `<root>/grammar_mask.py`。
3. Block ごとに:
   - 完了判定: 全 TrainJob と全 RescoreJob の `test_scores.jsonl` が揃っていれば
     `skip <source_bundle> (complete)` を log して次へ。
   - `backend.alive()` が偽なら `SESSION-LOST before <bundle>` を log → `reacquire` → `stage_code`。
   - `stage_bundle`: `train.jsonl val.jsonl test.jsonl vocab.json meta.json` を `<root>/<同名>` へ、
     さらに `vocab.json` → `<root>/vocab_<B>.json`、`meta.json` → `<root>/meta_<B>.json`。
   - TrainJob ごと(完了済みは `skip <name>`):
     `=== <name> start HH:MM:SS ===` → `backend.run(train_wrapper(job, root), name, exec)` →
     `runs/<name>/` を作り `samples.json history.json model.pt val_scores.jsonl test_scores.jsonl
     samples_constrained.json` を `get`。`test_scores.jsonl` があれば
     `local_eval(["uv","run","python","-m","training.eval_samples","--samples",...,"--vocab",
     "<bundle>/vocab.json","--train-tokens","<bundle>/train.jsonl","--out","runs/<name>/eval.json"])`
     を実行し、`backend.json` を書き、`=== <name> done HH:MM:SS ===`。無ければ
     `=== <name> FAILED HH:MM:SS ===`、`runs/<name>` を削除して `RemoteFailure`。
   - 再スコア(未完了の RescoreJob があるときだけ): target バンドルごとに
     `<target>/test.jsonl` → `<root>/test_<T>.jsonl`;source run ごとに `runs/<run>/model.pt` →
     `<root>/<run>_model.pt`、`runs/<run>/samples.json` → `<root>/<run>_samples.json`(ローカル
     `runs/` から。VM 上のファイルは使わない);`rescore_jobs(block, root)` を JSON にして
     `<root>/rescore_jobs.json` へ;`=== rescore <B> HH:MM:SS ===`;
     `backend.run(rescore_script(root), f"rescore_{B}", 2400)`;各 job の `out` を
     `runs/<out_run>/test_scores.jsonl` へ `get`(無ければ空 dir を削除)、`runs/<source>/eval.json`
     を `runs/<out_run>/eval.json` にコピー、`backend.json` を書く。
4. **障害処理**: `SessionLost` は Block 単位で捕捉し、`SESSION-LOST at <name>` を log →
   `router.reacquire` → `stage_code` → 同じ Block を先頭からやり直す(冪等なので完了分は skip)。
   同じ Block で `SessionLost`/`RemoteFailure` が 2 回続けば `release()` して 3 を返す。
   `BackendUnavailable`(再取得失敗)は 2 を返す。
5. 正常終了: `backend.release()`、`RUN-DONE HH:MM:SS` を log、0。
6. `backend.json`: `{"finished": ISO 時刻, "kind", "plan_id", "started": ISO 時刻, **describe()}`
   をキー昇順で `runs/<name>/backend.json` に書く(再スコア run にも書く)。
7. **dry-run**: `DryRunBackend(root="/content")` を使い、`put`/`get`/`run` を
   `PUT <local> -> <remote>` / `GET <remote> -> <local>` / `RUN <name> timeout=<s>` + スクリプト本文
   として log に出す。`get` は常に True を返すがファイルは書かない。Executor は `dry_run=True` の
   とき完了判定をメモリ上の集合で行い、`runs/` に触れず、`local_eval` は argv を log するだけ。
   `describe()` は `{"kind": "dry"}`。

## 9. CLI(`__main__.py`)と実装タスク

```
uv run python -m training.runner run --plan plan.json [--backend colab|local|auto]
    [--session NAME(既定 sw)] [--image TAG] [--staging DIR] [--runs-dir runs]
    [--attempts N] [--sleep-s S] [--dry-run] [--allow-backend-switch]
uv run python -m training.runner sweep --sizes 12,16,24,32,48 [--sources lay,mix] --out plan.json
```
- `--backend` 省略時は `resolve_policy(None)`(環境変数 `GR_BACKEND`、無ければ colab)。
- `--image` 既定 `pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime`(環境変数 `GR_DOCKER_IMAGE`
  で上書き)。`--staging` 既定 `<repo>/.runner_staging`(`.gitignore` に追加)。
- factories: `colab` → `ColabBackend(session, attempts=, sleep_s=)`、`local` → `DockerBackend(image, staging)`。
- 終了コード: 0 / 2 / 3、引数エラー 4。

### 実装タスク(各タスクは独立に検証可能。1 タスク = 1 回の委譲)

| # | 内容 | ファイル | テスト |
| --- | --- | --- | --- |
| T1 | `types.py`, `scripts.py`, `plans.py`, `__init__.py` | 上記 | `tests/test_runner_types.py`: JSON 往復・キー昇順・validate_plan の違反検出;`tests/test_runner_scripts.py`: §3 のラッパー本文と **byte 同一**(mask / base(extra=()) / pointer の 3 例をテスト内に埋め込む)、`rescore_jobs` の辞書、`rescore_script("/work")` に `/content` が残らない;`sweep_plan((12,16,24,32,48))` が TrainJob 30・RescoreJob 75、run 名一致;環境変数 `GR_SWEEP_WRAPPERS=<dir>` があればその dir の `w_<name>.py` 全件と byte 比較(無ければ skip) |
| T2 | `backend.py`, `colab.py` | 上記 | `tests/test_runner_colab.py`: 注入した runner が受けた argv を完全一致で検証(new / sessions / upload / download / exec / stop)、`Service Unavailable` を N 回返すと N 回 sleep して `BackendUnavailable`、途中成功で `SESSION-OK`、`alive` の行解析、upload 失敗 + sessions 不在 → `SessionLost`、download の `not found` → False、exec 出力の抽出、ロックファイルが作られコマンド実行中は他者が `LOCK_NB` で取れない |
| T3 | `docker.py`, `fake.py`, `training/docker/Dockerfile`, `train_ar.py` の `GR_ROOT` 既定 | 上記 | `tests/test_runner_docker.py`: run の argv 完全一致、put/get のステージング配置、`root` 外の remote は `ValueError`、2 つ目の `acquire` は `BackendUnavailable`;`train_ar.py` はテストで import しない(`grep` で既定値の文字列を確認する程度) |
| T4 | `router.py`, `executor.py` | 上記 | `tests/test_runner_router.py`: 方針解決(CLI > env > 既定 colab)、auto の順序と fallback、reacquire の同種固定と `allow_switch`;`tests/test_runner_executor.py`(FakeBackend): 正常系で 6 ファイル + `eval.json`(local_eval 呼び出し引数)+ `backend.json`、完了済み run は `run` が呼ばれない、失敗で部分 dir 削除 → 再取得 → 再試行、2 回失敗で 3、SessionLost 後の再ステージング(put の呼び出し列)と再開、再スコアの put/run/get 列と `eval.json` コピー、ログ標識の文言 |
| T5 | `__main__.py`, `.gitignore`, `compute_backend.md` §4 更新、`CLAUDE.md` に `GR_BACKEND` の注記 | 上記 | `tests/test_runner_cli.py`: `sweep` が JSON を書く、`run --dry-run` の出力に PUT/RUN/GET と `RUN-DONE` が含まれ `runs/` に何も作らない、不正 `--backend` は 4 |
| T6 | 受け入れ試験(委譲しない): ローカル GPU で 2 run の小 Plan(n12、epochs 2)を `--backend local` で end-to-end 実行し、`runs/` と `backend.json` を確認。続いて n32/n48 の Plan を `--backend local` で実行 | — | — |

検証コマンド(各タスク): `uv run pytest -q`、`uv run ty check`、`git diff --check`。

## 10. 決定(上位節より優先)

- D1 既定方針は `colab`。このマシンでは `GR_BACKEND=local` を環境変数で設定して使う(A8 と
  ユーザー補足)。テストと dry-run は実行先に触れない。
- D2 Docker バックエンドは T3 で Colab アダプタ直後に実装する。実機の受け入れ試験(T6)は
  ローカル GPU で行う。イメージは `pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime` を pin。
- D3 リモートの平坦名はバンドル名(`tok_s32_lay` など)で組む。現行の `vocab_32_lay.json` とは
  異なるが VM 上の一時名なので互換不要。ラッパー本文は byte 同一を要求する。
- D4 `train_ar.py` への変更は既定パスの `GR_ROOT` 化のみ。Colab での挙動は不変。
- D5 `SessionLost` の再試行は同一 Block で 2 回まで。実行先の種別切替は `--allow-backend-switch`
  を付けたときだけ。
- D6 学習コードもステージング経由で置く(マウントで見せない)。Colab と Docker で
  Executor の手順を完全に共有するため。
