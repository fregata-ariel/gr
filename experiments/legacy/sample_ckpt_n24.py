import json, sys
sys.path.insert(0, "/content")
for _m in ("train_ar", "grammar_mask"):
    sys.modules.pop(_m, None)
import torch, train_ar, grammar_mask

PTR2 = ["--pointer", "--pointer-legal", "--pointer-dist-bias", "--pointer-dist-bias-mode", "context"]
CONFIGS = [("ptr3_n24", PTR2), ("ptr5_n24", PTR2 + ["--struct-pos", "--struct-pos-mode", "depth_only"])]

vc = train_ar.Vocab(json.load(open("/content/vocab.json")))
meta = json.load(open("/content/meta.json"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen_max_len = 2 * meta["max_len"]
for name, flags in CONFIGS:
    args, _ = train_ar.build_parser().parse_known_args(flags)
    model = train_ar.build_model(vc, meta, args, gen_max_len).to(device)
    model.load_state_dict(torch.load(f"/content/{name}_model.pt", map_location=device))
    torch.manual_seed(1)
    samples = [train_ar.sample_stream(model, vc, device, gen_max_len, 1.0, 0,
                                      state=grammar_mask.GrammarState(vc.vocab),
                                      struct=args.struct_pos, pointer=args.pointer)
               for _ in range(400)]
    json.dump({"samples": samples}, open(f"/content/{name}_constrained.json", "w"))
    print("SAMPLED", name, len(samples), "on", device)
