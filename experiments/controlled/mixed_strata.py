"""Per-component NLL of the mixed-family models on their own test split (component via mixture.component_for)."""
import json, sys
from statistics import mean
from cfg_reducer.generate_v2 import spec_from_json
from cfg_reducer.families.mixture import component_for
m = json.load(open("data/c_mixed/manifest.json"))
spec = spec_from_json(m["generator"]["config"]["spec"])
names = [c["family"] for c in spec.params["components"]]
comp = {e["sample_id"]: names[component_for(spec, e["seed"])] for e in m["splits"]["test"]["samples"]}
train_comp = [names[component_for(spec, e["seed"])] for e in m["splits"]["train"]["samples"]]
print("train component counts:", {n: train_comp.count(n) for n in names}, "| test:", {n: sum(1 for v in comp.values() if v == n) for n in names})
for cfg in ("base", "mask"):
    rows = {}
    for s in range(3):
        try:
            for r in map(json.loads, open(f"runs/c_mixed_{cfg}_n24_s{s}/test_scores.jsonl")):
                rows.setdefault(r["sample_id"], []).append(r["nll_per_token"])
        except FileNotFoundError:
            print(f"  {cfg}: seed {s} pending"); 
    groups = {}
    for sid, v in rows.items(): groups.setdefault(comp.get(sid, "?"), []).append(mean(v))
    if groups: print(f"  {cfg:5s} mixed-ID test by component: " + "  ".join(f"{k}: {mean(v):.3f} (n={len(v)})" for k, v in sorted(groups.items())))
