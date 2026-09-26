import json
import sys

sys.path.insert(0, "/content")
import torch
import train_ar
import grammar_mask

vocab = json.load(open("/content/vocab.json"))
meta = json.load(open("/content/meta.json"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen_max_len = 2 * meta["max_len"]
model = train_ar.ARBaseline(
    vocab_size=len(vocab),
    max_len=max(meta["max_len"], gen_max_len),
    pad_id=vocab["PAD"],
).to(device)
model.load_state_dict(torch.load("/content/model.pt", map_location=device))

torch.manual_seed(1)
samples = [
    train_ar.sample_stream(
        model, vocab, device, gen_max_len, 1.0, 0,
        state=grammar_mask.GrammarState(vocab),
    )
    for _ in range(200)
]
json.dump({"samples": samples},
          open("/content/samples_constrained.json", "w"))
print("wrote", len(samples), "constrained samples on", device)
