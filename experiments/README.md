# experiments/ — 実験の再現に必要なメタデータ(git 管理)

Claude Code セッションの scratchpad(`/tmp`、再起動で消える)に置いていた spec、Plan JSON、観測値、集計スクリプトを
2026-09-26 にここへ移した(Q11-3)。`data/` と `runs/` は従来どおり git 管理外で、バックアップ(`tools/devenv/`)が持ち運ぶ。
以後、新しい実験の spec / Plan / 観測値はここに置き、scratchpad には一時ファイルだけを置く。

| ディレクトリ | 内容 | 対応する記録 |
|---|---|---|
| `sweep/` | ノード数スイープの spec `spec_<family>_n<N>.json`(別の n の spec はここから params を写す)、生成・実行スクリプト、`sweep_summary.py`、再スコア job 一覧、Colab wrapper | `docs/design/generator_v2.md` §14、`runs/sweep_summary.md` |
| `mixture_doe/` | 混合重み DoE の spec(`specs/`、n24 は p1–p12、n48 は spec48_p*)、Plan(`plan_doe_*.json`)、観測値(`obs*.json`、`obs48_raw.json`)、基準符号長 `base_*/`(git 管理外、`training.info_baseline` で再計算)、生成スクリプト、実行ログ | `docs/design/mixture_doe.md` §8–§9 |
| `runner/` | ルーターの受け入れ Plan(local / colab)、n32–n48 スイープ Plan、クロスバックエンド確認 Plan と比較スクリプト、実行ログ | `docs/design/compute_backend.md` §5 |
| `controlled/` | 実験 C(構造 OOD 行列)の spec / Plan / 生成・実行・カナリアスクリプト、集計スクリプト、wrapper、ログ | `docs/design/controlled_eval.md`、`runs/c_summary.md` |
| `pilot/` | v2 生成器のパイロット spec(3 family) | `docs/design/generator_v2.md` 初期節 |
| `legacy/` | ルーター導入前(Colab 手動運用)の wrapper と Plan、集計・再スコアスクリプト(`runs/b_*`, `runs/n24_*`, `runs/sweep_*` の記録) | `docs/design/ar_baseline.md`、`scale_experiments.md`、`representation_experiments.md` |

再現の入口:

```
uv run python -m training.runner run --plan experiments/mixture_doe/plan_doe_n48.json --dry-run
uv run python -m training.mixture_doe collect --runs runs --data data --points 1,2,3,7,8 --seeds 0,1,2 --size 48 --prefix d --out experiments/mixture_doe/obs48_raw.json
```

ログ(`*.log`)と基準符号長(`base_*/`)は gitignore だが同じ場所に置く(バックアップの対象にはなる)。
