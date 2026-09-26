import json, sys
from pathlib import Path
NODES = [12, 16, 20, 24, 28, 32, 40, 48]
print(f"{'n':>3s} {'train':>5s} {'maxlen':>6s} {'refW':>4s} {'bestEp':>6s} {'run':>4s} "
      f"{'bestVL':>7s} {'bestVA':>6s} {'WF%':>6s} {'uniq%':>6s} {'novel%':>6s} {'top violations'}")
for n in NODES:
    d = Path(f"runs/sweep_n{n}")
    if not (d / "eval.json").exists():
        print(f"{n:>3d} (pending)"); continue
    ev = json.loads((d / "eval.json").read_text())
    hist = json.loads((d / "history.json").read_text())
    smp = json.loads((d / "samples.json").read_text())
    meta = json.loads(Path(f"data/tokens_sweep_n{n}/meta.json").read_text())
    best = min(hist, key=lambda h: h["val_loss"])
    top = ", ".join(f"{k}:{v}" for k, v in list(ev["violations"].items())[:3])
    print(f"{n:>3d} {meta['splits']['train']:>5d} {meta['max_len']:>6d} {meta['max_offset']:>4d} "
          f"{smp.get('best_epoch', best['epoch']):>6d} {len(hist):>4d} {best['val_loss']:>7.4f} {best['val_acc']:>6.3f} "
          f"{ev['well_formed_rate']*100:>5.1f}% {ev['unique_rate']*100:>5.1f}% {ev['novelty_rate']*100:>5.1f}%  {top}")
