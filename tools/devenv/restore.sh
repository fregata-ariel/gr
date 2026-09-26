#!/usr/bin/env bash
# tools/devenv/restore.sh — restore a backup generation on this or another machine (e.g. a Coder workspace),
# relocating the checkout and the Claude Code session to a new project path.
#
#   restore.sh <generation-dir> [<new-project-path>]     default path: $HOME/Projects/Compiler/gr
#   DRY_RUN=1 restore.sh …                                 print what would be copied, copy nothing
#
# A generation is <gen>/home/… and <gen>/tmp/… as written by backup.sh. Claude Code keys its per-project
# state by the project path with "/" replaced by "-" (e.g. /home/coder/gr -> -home-coder-gr), and the session
# scratchpad lives under /tmp/claude-<uid>/<encoded path>/<session id>/scratchpad, so both are re-keyed here.
# Nothing is deleted on the target; existing files are overwritten by the backup's copy.
set -euo pipefail
gen=${1:?usage: restore.sh <generation-dir> [<new-project-path>]}
new=${2:-$HOME/Projects/Compiler/gr}
gen=$(cd "$gen" && pwd); new=${new%/}
[ -d "$gen/home" ] || { echo "not a generation directory: $gen"; exit 1; }
dry=(); [ "${DRY_RUN:-0}" = 1 ] && dry=(-n)
rs() { rsync -a "${dry[@]}" --info=stats1 "$@" | { grep -E 'Number of (files|created files|regular files transferred)' || true; } | sed 's/^/    /'; }

# 1. locate the checkout inside the generation (the single directory holding .git under home/Projects)
old_rel=$(cd "$gen/home" && find . -maxdepth 4 -name .git -type d -printf '%h\n' | sed 's#^\./##' | head -1)
[ -n "$old_rel" ] || { echo "no git checkout found under $gen/home"; exit 1; }
old_enc=$(cd "$gen/home/.claude/projects" && ls -d -- -* | head -1)         # e.g. -home-fischeri-Projects-Compiler-gr
new_enc=$(printf '%s' "$new" | sed 's#/#-#g')
sid=$(basename "$(ls "$gen/home/.claude/projects/$old_enc"/*.jsonl | head -1)" .jsonl)
old_tmp=$(cd "$gen/tmp" && find . -maxdepth 4 -type d -name "$sid" -printf '%p\n' | sed 's#^\./##' | head -1)   # claude-1000/<old_enc>/<sid>
new_tmp="/tmp/claude-$(id -u)/$new_enc/$sid"
echo "generation : $gen"
echo "checkout   : ~/$old_rel  ->  $new"
echo "claude dir : ~/.claude/projects/$old_enc  ->  ~/.claude/projects/$new_enc"
echo "session id : $sid"
echo "scratchpad : /tmp/$old_tmp/scratchpad  ->  $new_tmp/scratchpad"
[ "${DRY_RUN:-0}" = 1 ] && echo "(dry run — nothing is written)"

echo "== 1. checkout (data/, runs/ included; .venv is re-created by 'uv sync')"
[ "${DRY_RUN:-0}" = 1 ] || mkdir -p "$(dirname "$new")"; rs "$gen/home/$old_rel/" "$new/"
echo "== 2. Claude Code project state (transcript, tool-results, memory)"
mkdir -p "$HOME/.claude/projects"; rs "$gen/home/.claude/projects/$old_enc/" "$HOME/.claude/projects/$new_enc/"
echo "== 3. shared Claude Code config (settings, plugins, skills, commands, per-session env / file history)"
for d in settings.json plugins skills commands "session-env/$sid" "file-history/$sid"; do
  [ -e "$gen/home/.claude/$d" ] || continue
  mkdir -p "$HOME/.claude/$(dirname "$d")"; rs "$gen/home/.claude/$d" "$HOME/.claude/$(dirname "$d")/"
done
for d in .agents/skills/agmsg .config/opencode/opencode.jsonc .codex/config.toml .config/colab-cli/settings.json .config/gr-devenv/env.sh; do
  [ -e "$gen/home/$d" ] || continue
  mkdir -p "$HOME/$(dirname "$d")"; rs "$gen/home/$d" "$HOME/$(dirname "$d")/"
done
echo "== 4. session scratchpad (/tmp; re-keyed to the new project path)"
if [ -n "$old_tmp" ]; then [ "${DRY_RUN:-0}" = 1 ] || mkdir -p "$new_tmp"; rs "$gen/tmp/$old_tmp/scratchpad/" "$new_tmp/scratchpad/"; else echo "    (no scratchpad in this generation)"; fi

cat <<MSG
== 5. manual steps on the target
  - secrets (never in the backup): gh auth login; colab login (google-colab-cli 0.6.0 via uv tool); opencode auth; codex login; ssh key for github
  - cd $new && uv sync && uv run pytest -q
  - git remote -v ; git status   (compare with $gen/git-state.txt)
  - git worktree prune          (the /tmp/oc-wt worktree does not exist here)
  - GPU backend only: docker pull pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime ; otherwise GR_BACKEND stays colab
  - old absolute paths inside the transcript/memory (~/$old_rel and /tmp/$old_tmp on the source machine) refer to this machine: tell Claude the new paths
  - resume: cd $new && claude --resume $sid
MSG
