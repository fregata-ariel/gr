import sys, json, argparse
sys.path.insert(0, "/content")
for _m in ("train_ar", "grammar_mask"):
    sys.modules.pop(_m, None)
import torch, train_ar
# /content/rescore_jobs.json: [{"run": name, "vocab": path, "meta": path, "test": path, "out": path}, ...]
JOBS = json.load(open("/content/rescore_jobs.json"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for job in JOBS:
    run = job["run"]
    cfg = json.load(open(job["config"]))["config"]
    args = argparse.Namespace(**cfg)
    vc = train_ar.Vocab(json.load(open(job["vocab"])))
    meta = json.load(open(job["meta"]))
    gen_max_len = args.gen_max_len or 2 * meta["max_len"]
    model = train_ar.build_model(vc, meta, args, gen_max_len).to(device)
    model.load_state_dict(torch.load(job["model"], map_location=device))
    aux_ctx = args.pointer or getattr(args, "ref_legal_mask", False)
    scores = train_ar.score_rows(model, train_ar.load_rows(job["test"]), vc, device, args.struct_pos, aux_ctx)
    open(job["out"], "w").write("".join(json.dumps(r) + "\n" for r in scores))
    print("RESCORED", run, job["test"], len(scores))
print("RESCORE-DONE")
