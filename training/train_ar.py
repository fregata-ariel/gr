"""
AR baseline trainer — runs on Colab, torch only, no cfg_reducer
dependency (docs/design/ar_baseline.md).

Single-file by design: it is the only script uploaded to the VM.
Defaults match the upload paths in the design note, so it can run with
no arguments after `colab upload`:

    python train_ar.py

Consumes prepare_tokens.py outputs (vocab.json / meta.json / *.jsonl)
and writes <out>/model.pt, <out>/history.json, <out>/samples.json.
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

# torch is a Colab-side dependency only; it is deliberately absent from
# the local environment (pyproject stays untouched).
import torch                  # ty: ignore[unresolved-import]
from torch import nn          # ty: ignore[unresolved-import]


# ── data ─────────────────────────────────────

def load_streams(path: str | Path) -> list[list[int]]:
    return [
        json.loads(line)["tokens"]
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def iter_batches(seqs, batch_size, pad_id, device, rng=None):
    order = list(range(len(seqs)))
    if rng is not None:
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [seqs[j] for j in order[i:i + batch_size]]
        width = max(len(s) for s in chunk)
        batch = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        for row, seq in enumerate(chunk):
            batch[row, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        yield batch.to(device)


# ── model ────────────────────────────────────

class ARBaseline(nn.Module):
    def __init__(self, vocab_size, max_len, pad_id,
                 d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        length = x.size(1)
        positions = torch.arange(length, device=x.device)
        hidden = self.tok_emb(x) + self.pos_emb(positions)[None]
        causal = nn.Transformer.generate_square_subsequent_mask(
            length, device=x.device
        )
        hidden = self.encoder(
            hidden, mask=causal, src_key_padding_mask=x.eq(self.pad_id)
        )
        return self.head(hidden)


# ── train / eval ─────────────────────────────

def run_epoch(model, seqs, batch_size, pad_id, device,
              optimizer=None, rng=None):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    total_loss = total_tokens = total_correct = 0
    with torch.set_grad_enabled(training):
        for batch in iter_batches(seqs, batch_size, pad_id, device, rng):
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = model(inputs)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            mask = targets.ne(pad_id)
            n_tokens = int(mask.sum())
            total_loss += float(loss) * n_tokens
            total_tokens += n_tokens
            total_correct += int(
                (logits.argmax(-1).eq(targets) & mask).sum()
            )

    return total_loss / total_tokens, total_correct / total_tokens


@torch.no_grad()
def sample_stream(model, vocab, device, max_len, temperature, top_k):
    model.eval()
    ids = [vocab["BOS"]]
    for _ in range(max_len - 1):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x)[0, -1] / temperature
        logits[vocab["PAD"]] = float("-inf")
        logits[vocab["BOS"]] = float("-inf")

        if top_k > 0:
            keep = torch.topk(logits, min(top_k, logits.size(-1))).indices
            filtered = torch.full_like(logits, float("-inf"))
            filtered[keep] = logits[keep]
            logits = filtered

        next_id = int(torch.multinomial(torch.softmax(logits, -1), 1))
        ids.append(next_id)
        if next_id == vocab["EOS"]:
            break
    return ids


# ── main ─────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the AR baseline.")
    parser.add_argument("--train", default="/content/train.jsonl")
    parser.add_argument("--val", default="/content/val.jsonl")
    parser.add_argument("--vocab", default="/content/vocab.json")
    parser.add_argument("--meta", default="/content/meta.json")
    parser.add_argument("--out", default="/content/run1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0,
                        help="0 disables top-k filtering")
    parser.add_argument("--gen-max-len", type=int, default=0,
                        help="0 means 2x the training max length")
    # parse_known_args: `colab exec` runs this file inside a notebook
    # kernel whose sys.argv carries kernel flags (-f /path/kernel.json).
    args, _ = parser.parse_known_args(argv)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    train_seqs = load_streams(args.train)
    val_seqs = load_streams(args.val)

    gen_max_len = args.gen_max_len or 2 * meta["max_len"]
    model = ARBaseline(
        vocab_size=len(vocab),
        max_len=max(meta["max_len"], gen_max_len),
        pad_id=vocab["PAD"],
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_seqs, args.batch_size, vocab["PAD"], device,
            optimizer=optimizer, rng=rng,
        )
        val_loss, val_acc = run_epoch(
            model, val_seqs, args.batch_size, vocab["PAD"], device,
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })
        print(f"epoch {epoch:3d}  train {train_loss:.4f}/{train_acc:.3f}  "
              f"val {val_loss:.4f}/{val_acc:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out / "model.pt")

    (out / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    model.load_state_dict(
        torch.load(out / "model.pt", map_location=device)
    )
    samples = [
        sample_stream(model, vocab, device, gen_max_len,
                      args.temperature, args.top_k)
        for _ in range(args.num_samples)
    ]
    (out / "samples.json").write_text(
        json.dumps({
            "best_val_loss": best_val,
            "config": vars(args),
            "samples": samples,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(samples)} samples to {out / 'samples.json'}")


if __name__ == "__main__":
    main()
