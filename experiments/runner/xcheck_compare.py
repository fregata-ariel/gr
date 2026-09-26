import json, math, sys
from pathlib import Path
def nll(path):
    rows=[json.loads(l) for l in open(path)]
    tok=sum(r.get("n_tokens", r.get("tokens", 0)) for r in rows); tot=sum(r.get("nll_sum", r.get("nll", 0)) for r in rows)
    if tok: return tot/tok, len(rows)
    # fallback: mean of per-row nll/token
    key=[k for k in rows[0] if "nll" in k][0]
    return sum(r[key] for r in rows)/len(rows), len(rows)
pairs=[("c_s24_lay_mask_n24_s0","x_s24_lay_mask_n24_s0"),("c_s24_lay2str_mask_n24_s0","x_s24_lay2str_mask_n24_s0"),("c_s24_lay2spa_mask_n24_s0","x_s24_lay2spa_mask_n24_s0"),
       ("c_s24_mix_mask_n24_s0","x_s24_mix_mask_n24_s0"),("c_s24_mix2str_mask_n24_s0","x_s24_mix2str_mask_n24_s0"),("c_s24_mix2spa_mask_n24_s0","x_s24_mix2spa_mask_n24_s0"),("c_s24_mix2lay_mask_n24_s0","x_s24_mix2lay_mask_n24_s0")]
print(f"{'cell':34s} {'T4':>8s} {'2080Ti':>8s} {'diff':>8s}")
for a,b in pairs:
    pa=Path("runs")/a/"test_scores.jsonl"; pb=Path("runs")/b/"test_scores.jsonl"
    if not (pa.exists() and pb.exists()): print(f"{a:34s} pending"); continue
    na,_=nll(pa); nb,_=nll(pb); print(f"{a:34s} {na:8.4f} {nb:8.4f} {nb-na:+8.4f}")
for a,b in pairs[:1]+pairs[3:4]:
    for r in (a,b):
        h=json.load(open(Path("runs")/r/"history.json")) if (Path("runs")/r/"history.json").exists() else None
        e=json.load(open(Path("runs")/r/"eval.json")) if (Path("runs")/r/"eval.json").exists() else None
        if h and e: print(f"{r:34s} epochs={len(h)} best_val={min(x['val_loss'] for x in h):.4f} WF={e.get('well_formed_rate')}")
