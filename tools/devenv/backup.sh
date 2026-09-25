#!/usr/bin/env bash
# tools/devenv/backup.sh — generation backup of the gr development environment.
#
#   backup.sh now      take a new generation <DEST>/<YYYYMMDD-HHMM>/ (hard-linked against <DEST>/latest), then prune
#   backup.sh verify   check that the newest generation holds the must-have items ("V" lines) and print a summary
#   backup.sh list     list generations with their sizes
#   backup.sh prune    remove generations beyond BACKUP_KEEP
#
# What is copied is defined by backup-manifest.txt next to this script:
#   "+ path"   include, relative to $HOME          "- pattern"  rsync exclude pattern
#   "T /tmp/…" absolute /tmp path, copied under <gen>/tmp/…   "V path"  must-have item for `verify`
#   "S"/"E"/"D" lines are documentation only (never copied).
# Machine-specific values come from env.sh next to this script (copy env.sh.example; env.sh is git-ignored).
# Layout of a generation: <gen>/home/<path relative to $HOME>, <gen>/tmp/<path relative to /tmp>, <gen>/git-state.txt
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$here/env.sh"
: "${BACKUP_DEST:?set BACKUP_DEST in env.sh}" "${BACKUP_KEEP:=14}"
manifest="$here/backup-manifest.txt"
cmd=${1:-now}

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$BACKUP_DEST/backup.log" >&2; }

parse_manifest() {
  # emits: kind<TAB>value  with trailing comments stripped
  sed -E 's/[[:space:]]+#.*$//; /^[[:space:]]*(#|$)/d' "$manifest" \
    | awk '{k=$1; $1=""; sub(/^ /,""); print k "\t" $0}'
}

git_state() {
  # git status of every included checkout, so a later restore knows what was uncommitted and what was pushed
  local incl=$1 d
  while read -r d; do
    [ -d "$HOME/$d/.git" ] || continue
    echo "== ~/$d"
    git -C "$HOME/$d" status --short --branch 2>/dev/null | head -40
    echo "HEAD $(git -C "$HOME/$d" rev-parse --short HEAD 2>/dev/null)"
    git -C "$HOME/$d" for-each-ref --format='%(refname:short) %(upstream:short) %(upstream:track)' refs/heads/ 2>/dev/null
    echo "-- worktrees"; git -C "$HOME/$d" worktree list 2>/dev/null
    echo "-- stash"; git -C "$HOME/$d" stash list 2>/dev/null | head -5
  done <"$incl"
}

do_now() {
  mkdir -p "$BACKUP_DEST"
  exec 9>"$BACKUP_DEST/.lock"; flock -n 9 || { log "another backup is running; exit"; exit 1; }
  local gen latest="$BACKUP_DEST/latest"
  gen="$BACKUP_DEST/$(date +%Y%m%d-%H%M)"
  [ -e "$gen" ] && { log "generation $gen already exists; exit"; exit 1; }
  mkdir -p "$gen/home" "$gen/tmp"
  local incl="$gen/.include-home" excl="$gen/.exclude" tincl="$gen/.include-tmp"
  : >"$incl"; : >"$excl"; : >"$tincl"
  while IFS=$'\t' read -r kind value; do
    case "$kind" in
      +) [ -e "$HOME/$value" ] && printf '%s\n' "$value" >>"$incl" || log "WARN missing (skipped): ~/$value" ;;
      -) printf '%s\n' "$value" >>"$excl" ;;
      T) [ -e "$value" ] && printf '%s\n' "${value#/}" >>"$tincl" || log "WARN missing (skipped): $value" ;;
    esac
  done < <(parse_manifest)
  git_state "$incl" >"$gen/git-state.txt" 2>&1 || true
  local link=(); [ -d "$latest/home" ] && link=(--link-dest="$latest/home")
  log "generation $gen: home items $(wc -l <"$incl"), tmp items $(wc -l <"$tincl")"
  # --files-from does not imply recursion even with -a, hence the explicit -r
  ( cd "$HOME" && rsync -arR --delete --delete-excluded --exclude-from="$excl" "${link[@]}" --files-from="$incl" ./ "$gen/home/" )
  if [ -s "$tincl" ]; then
    # entries are "tmp/…" relative to /, so -R recreates them under <gen>/tmp/…
    local tlink=(); [ -d "$latest/tmp" ] && tlink=(--link-dest="$latest")
    ( cd / && rsync -arR --delete --delete-excluded --exclude-from="$excl" "${tlink[@]}" --files-from="$tincl" / "$gen/" )
  fi
  ln -sfn "$gen" "$latest"
  log "done: $(du -sh "$gen" | cut -f1) in $gen (hard-linked against the previous generation)"
  do_prune
}

do_prune() {
  local gens; mapfile -t gens < <(ls -1d "$BACKUP_DEST"/[0-9]*-[0-9]* 2>/dev/null | sort)
  local n=${#gens[@]}
  while [ "$n" -gt "$BACKUP_KEEP" ]; do
    log "prune ${gens[0]}"; rm -rf "${gens[0]}"; gens=("${gens[@]:1}"); n=${#gens[@]}
  done
}

do_list() {
  for g in "$BACKUP_DEST"/[0-9]*-[0-9]*; do [ -d "$g" ] && printf '%s  %s\n' "$(du -sh "$g" | cut -f1)" "$g"; done
  echo "latest -> $(readlink "$BACKUP_DEST/latest" 2>/dev/null)"
}

do_verify() {
  local g="$BACKUP_DEST/latest"; [ -d "$g" ] || { echo "no generation under $BACKUP_DEST"; exit 1; }
  local rc=0
  check() { if [ -e "$g/$1" ]; then printf 'ok      %s\n' "$1"; else printf 'MISSING %s\n' "$1"; rc=1; fi; }
  for v in $(parse_manifest | awk -F'\t' '$1=="V"{print $2}'); do check "$v"; done
  for t in $(parse_manifest | awk -F'\t' '$1=="T"{print $2}'); do check "tmp/${t#/tmp/}"; done
  echo "--- git-state.txt"; cat "$g/git-state.txt"
  echo "--- size $(du -sh "$g/" | cut -f1)  generations $(ls -1d "$BACKUP_DEST"/[0-9]*-[0-9]* | wc -l)"
  return $rc
}

case "$cmd" in
  now) do_now ;; verify) do_verify ;; list) do_list ;; prune) do_prune ;;
  *) echo "usage: $0 [now|verify|list|prune]"; exit 2 ;;
esac
