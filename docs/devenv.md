# 開発環境のバックアップと移行(Coder Web IDE)

作成: 2026-09-26。対象: `gr` の開発環境一式(リポジトリ、実験データ、Claude Code セッション、周辺ツール)。
目的は (1) 何を持ち運ぶべきかの範囲確定、(2) 継続的なバックアップ、(3) Coder ワークスペースへの移行手順の準備。
calisp(pyClangAST)側の `backup.sh` + manifest + systemd timer 方式をそのまま踏襲し、スクリプトは `tools/devenv/` に置く。

## 1. 範囲確定(棚卸し 2026-09-26)

区分は manifest(`tools/devenv/backup-manifest.txt`)の接頭辞と対応する。

### A. git 管理下(コピー: `~/Projects/Compiler/gr`、`.git` 込み)

| 項目 | 内容 | 規模 |
|---|---|---|
| 追跡ファイル | 115 件(cfg_reducer, training, tests, docs) | 約 6 MB(.git 含む) |
| 未コミット | `pyproject.toml` / `uv.lock`(ipykernel, ipython。方針によりコミットしない) | — |
| `data/` | 188 ディレクトリ。spec+seed から再生成できるが、古いものは当時のコミットが必要 | 2.4 GB |
| `runs/` | 465 run ディレクトリ + サマリ。`model.pt` 170 個(再スコアリングに必要) | 836 MB |
| 除外 | `.venv`(`uv sync`)、`.runner_staging/`(runs/ と同一内容を確認済み、583 MB)、`__pycache__`、`.pytest_cache` | — |

Python 依存: `matplotlib`, `networkx`(実行時)、`pytest`, `ty`(開発)、Python 3.13、`uv 0.11.2`。torch は Colab / Docker コンテナ内のみ。

### B. git 外の記録と Claude Code セッション(コピー)

| 項目 | パス | 備考 |
|---|---|---|
| セッション本体 | `~/.claude/projects/-home-fischeri-Projects-Compiler-gr/` | transcript `b84df06a-….jsonl`(15 MB)、tool-results、**memory/** |
| セッション付随 | `~/.claude/file-history/<sid>`, `~/.claude/session-env/<sid>` | undo 用履歴、セッション開始フック |
| 共有設定 | `~/.claude/settings.json`, `plugins/`(openai-codex 1.0.6), `skills/`, `commands/` | |
| エージェント設定 | `~/.agents/skills/agmsg`, `~/.config/opencode/opencode.jsonc`, `~/.codex/config.toml`, `~/.config/colab-cli/settings.json` | 認証情報は含まない |
| バックアップ設定 | `~/.config/gr-devenv/env.sh`, `~/.config/systemd/user/gr-backup.{service,timer}` | |

### C. セッションの scratchpad(`/tmp`、再起動で消える → 世代の `tmp/` 配下にコピー)

`/tmp/claude-1000/-home-fischeri-Projects-Compiler-gr/<sid>/scratchpad/`(25 MB)。実験の再現に必要なもの
(スイープ spec、DoE の spec / Plan / 観測値、ルーターの Plan、集計スクリプト、実験 C とルーター導入前の記録)は
2026-09-26 に `experiments/` へ移して git 管理にした(A11-3、`experiments/README.md`)。基準符号長と実行ログは
gitignore のまま `experiments/` に置く(チェックアウトごとバックアップされる)。scratchpad に残るのは委任プロンプト、
OpenCode の出力、smoke/probe の一時物で、記録としてコピーはするが作業再開には要らない。

### D. 再生成できるもの(コピーしない)

`.venv`、`~/.cache/uv`、Docker イメージ `pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime`(7.4 GB、ローカル GPU 用のみ)、
uv tool(`google-colab-cli` 0.6.0、`mdtools`)、node v24(nvm)、codex CLI、opencode、`/tmp/oc-wt`(OpenCode 用 worktree)、`.runner_staging/`。

### E. 外部ソース(clone / login で復元)

`git@github.com:fregata-ariel/gr.git`(全ブランチ push 済み)、companion の `pyClangAST`(読み取り専用参照、独自にバックアップ)、Colab ランタイム。

### S. 秘密情報(バックアップに含めない。移行先で手入力、または Coder の secrets)

`~/.config/colab-cli/token.json`、`~/.colab-cli-oauth*.json`、`~/.local/share/opencode/auth.json`、`~/.codex/auth.json`、
`~/.config/gh/hosts.yml`、`~/.claude/.credentials.json`、`~/.ssh`。

### 移行先で必要なシステムツール

`git`, `uv`(Python 3.13 は uv が取得), `gh`, `node`(codex companion plugin 用), `codex`, `opencode`, `rsync`,
`uv tool install google-colab-cli==0.6.0`(ルーターの Colab アダプタは 0.6.0 のコマンド体系で検証済み。0.7.2 への更新は別途検証)。
ローカル GPU バックエンドを使う場合のみ `docker` + `nvidia-container-toolkit`。既定バックエンドは `colab` なので GPU 無しでも動く。

## 2. 継続的バックアップ(導入済み 2026-09-26)

- スクリプト: `tools/devenv/backup.sh`(manifest 駆動、世代 = `<BACKUP_DEST>/<YYYYMMDD-HHMM>/`、前世代とハードリンク、`BACKUP_KEEP=14`)。
- 導入: `tools/devenv/install.sh` が `~/.config/gr-devenv/` にコピーし、`gr-backup.timer`(毎日 03:30、calisp の 03:00 と重ならない)を有効化する。
  manifest やスクリプトを変えたら再実行する。
- 実行先: `BACKUP_DEST=/mnt/data/backups/gr`(`~/.config/gr-devenv/env.sh`)。
- 手動: `~/.config/gr-devenv/backup.sh now | verify | list | prune`。
- 初回世代 `20260926-0422`: 3.3 GB、37 秒、`verify` の必須項目(V 行)すべて ok。
- 注意: manifest の `file-history/<sid>`、`session-env/<sid>`、`T` 行はセッション ID に依存する。新しいセッションに移ったら書き換える。

## 3. 復元と移行(`tools/devenv/restore.sh`)

```
DRY_RUN=1 tools/devenv/restore.sh /mnt/data/backups/gr/latest /home/coder/gr   # 確認
tools/devenv/restore.sh /mnt/data/backups/gr/latest /home/coder/gr             # 実行
```

- チェックアウト(data/, runs/ 込み)を新パスへ、Claude Code のプロジェクト状態を新パスのキー
  (`/` → `-`、例 `-home-coder-gr`)へ、scratchpad を `/tmp/claude-<uid>/<新キー>/<sid>/scratchpad` へ再配置する。削除はしない。
- 続いて手作業: 秘密情報のログイン、`uv sync`、`uv run pytest -q`、`git worktree prune`、必要なら Docker イメージの pull。
  再開は `claude --resume <sid>`。transcript とメモリ内の旧絶対パスは移行先では無効なので、再開時に新パスを伝える。
- 状態: 手元で dry-run(疑似的な新パス)まで確認。実機での復元は Coder ワークスペース側で行う(段階 3)。
- A11-5: 移行先でも同じパス `/home/fischeri/Projects/Compiler/gr` を使う(ユーザー `fischeri`、uid 1000)。
  このときセッションのキーも scratchpad のパスも変わらないので、`restore.sh <世代>`(第 2 引数なし)で足りる。

## 4. 段階計画

| 段階 | 内容 | 状態 |
|---|---|---|
| 1 | 範囲確定(本書 §1、manifest) | 完了 |
| 2 | 継続バックアップ(timer + verify) | 完了、運用中 |
| 3 | 移行スクリプト(restore.sh)と Coder 側の確認 | スクリプト作成・dry-run 済み、実機未検証 |
| 4 | Coder 前提の整理(docs/handoff_questions.md Q11 / A11) | 回答済み: GPU/Docker はワークスペース外の Pod、data/runs 持ち込み、experiments/ を git 管理、Colab CLI 0.6.0 固定、同一パスで resume |

次の段階: Coder ワークスペースを用意したら `restore.sh` を実機で通し、`claude --resume` を確認する。GPU Pod ワーカーは
`compute_backend.md` の 3 つ目のバックエンドとして設計する(Coder から Pod を起動して Plan を流す)。
