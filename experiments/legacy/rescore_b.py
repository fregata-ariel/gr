import sys, json, argparse
sys.path.insert(0, "/content")
for _m in ("train_ar", "grammar_mask"):
    sys.modules.pop(_m, None)
import torch, train_ar
RUNS = json.load(open("/content/rescore_list.json"))
device = torch.device("cpu")
for run in RUNS:
    cfg = json.load(open(f"/content/{run}_samples.json"))["config"]
    args = argparse.Namespace(**cfg)
    vc = train_ar.Vocab(json.load(open("/content/vocab.json")))
    meta = json.load(open("/content/meta.json"))
    gen_max_len = args.gen_max_len or 2 * meta["max_len"]
    model = train_ar.build_model(vc, meta, args, gen_max_len).to(device)
    model.load_state_dict(torch.load(f"/content/{run}_model.pt", map_location=device))
    aux_ctx = args.pointer or getattr(args, "ref_legal_mask", False)
    scores = train_ar.score_rows(model, train_ar.load_rows("/content/test.jsonl"), vc, device,
                                 args.struct_pos, aux_ctx)
    open(f"/content/{run}_test_scores.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in scores))
    print("RESCORED", run, len(scores))
print("RESCORE-DONE")
