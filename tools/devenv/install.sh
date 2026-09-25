#!/usr/bin/env bash
# tools/devenv/install.sh — install backup.sh + manifest under ~/.config/gr-devenv and enable the daily timer.
# Re-run after editing the manifest or the script; env.sh is created from env.sh.example only if absent.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cfg="$HOME/.config/gr-devenv"; units="$HOME/.config/systemd/user"
mkdir -p "$cfg" "$units"
install -m 755 "$here/backup.sh" "$cfg/backup.sh"
install -m 644 "$here/backup-manifest.txt" "$cfg/backup-manifest.txt"
[ -e "$cfg/env.sh" ] || { install -m 644 "$here/env.sh.example" "$cfg/env.sh"; echo "created $cfg/env.sh — edit BACKUP_DEST if needed"; }
install -m 644 "$here/gr-backup.service" "$units/gr-backup.service"
install -m 644 "$here/gr-backup.timer" "$units/gr-backup.timer"
systemctl --user daemon-reload
systemctl --user enable --now gr-backup.timer
systemctl --user list-timers gr-backup.timer --no-pager
echo "installed: $cfg  (run '$cfg/backup.sh now' for a first generation, then 'verify')"
