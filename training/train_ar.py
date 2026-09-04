"""
AR baseline trainer — runs on Colab, torch only, no cfg_reducer
dependency (docs/design/ar_baseline.md).

Uploaded next to grammar_mask.py; defaults match the runbook upload
paths so it can run with no arguments after `colab upload`.

Variants (docs/design/representation_experiments.md):
  baseline        REF_k predicted as vocabulary tokens
  --struct-pos    案 1  explicit level-count / depth embeddings
                  (--struct-pos-mode sinusoidal = 案 1')
  --pointer       案 2  REF predicted by a pointer head over same-level
                  earlier motifs; the token stream is unchanged
  --ref-legal-mask  B   vocabulary baseline whose REF_k logits are masked
                  to the pointer's legal universe (same output space as
                  --pointer-legal; external review 2026-09-04, 1-2)
  --test / --sample-seed / --constrained-samples  B  held-out scoring,
                  paired sampling RNG, constrained samples alongside

Consumes prepare_tokens.py outputs and writes <out>/model.pt,
history.json, samples.json, val_scores.jsonl.
"""

from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path

# torch is a Colab-side dependency only; it is deliberately absent from
# the local environment (pyproject stays untouched).
import torch                  # ty: ignore[unresolved-import]
from torch import nn          # ty: ignore[unresolved-import]

# On the VM grammar_mask.py sits next to this file; locally it lives in
# the training package. Needed for --constrained / --struct-pos / --pointer.
try:
    import grammar_mask as _gm
except ImportError:
    try:
        from training import grammar_mask as _gm
    except ImportError:
        _gm = None   # ty: ignore[conflicting-declarations, invalid-assignment]
grammar_mask = _gm

NEG = float("-inf")


# ── data ─────────────────────────────────────

def load_rows(path):
    return [json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_streams(path):
    return [row["tokens"] for row in load_rows(path)]


class Vocab:
    """Token ids plus the collapsed type vocabulary used by --pointer
    (all REF_k share one type; distance is recovered from the pointer)."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.names = {i: t for t, i in vocab.items()}
        self.pad, self.bos, self.eos = vocab["PAD"], vocab["BOS"], vocab["EOS"]
        self.ref_ids = {int(t[4:]): i for t, i in vocab.items() if t.startswith("REF_")}
        self.max_k = max(self.ref_ids) if self.ref_ids else 0

        type_ids, self.type_names = {}, []
        for name in sorted(vocab, key=vocab.get):
            key = "REF" if name.startswith("REF_") else name
            if key not in type_ids:
                type_ids[key] = len(self.type_names)
                self.type_names.append(key)
        self.type_of = [type_ids["REF" if n.startswith("REF_") else n]
                        for n in sorted(vocab, key=vocab.get)]
        self.ref_type = type_ids["REF"]
        self.token_of_type = {type_ids[n]: vocab[n] for n in vocab
                              if not n.startswith("REF_")}


def sequence_aux(seq, vc: Vocab, struct: bool, pointer: bool):
    """Per-sequence structural inputs (lists aligned with seq)."""
    aux = {}
    if struct:
        depth, lpos = grammar_mask.structural_positions(seq, vc.vocab)
        aux["depth"], aux["lpos"] = depth, lpos
    if pointer:
        ctx = grammar_mask.pointer_context(seq, vc.vocab)
        aux["is_kind"] = [int(k) for k in ctx["is_kind"]]
        aux["level_id"] = ctx["level_id"]
        aux["plpos"] = ctx["lpos"]
        aux["klast"] = ctx["klast"]
        aux["ref_target"] = ctx["ref_target"]
    return aux


PAD_VALUE = {"depth": 0, "lpos": 0, "is_kind": 0, "level_id": -2,
             "plpos": 0, "klast": 0, "ref_target": -1}


def _pad(rows, width, fill, device):
    out = torch.full((len(rows), width), fill, dtype=torch.long)
    for r, seq in enumerate(rows):
        out[r, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out.to(device)


def iter_batches(seqs, auxes, batch_size, pad_id, device, rng=None):
    """Yields dict batches: tokens plus every aux key present."""
    order = list(range(len(seqs)))
    if rng is not None:
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        width = max(len(seqs[j]) for j in idx)
        batch = {"tokens": _pad([seqs[j] for j in idx], width, pad_id, device)}
        for key in auxes[idx[0]]:
            batch[key] = _pad([auxes[j][key] for j in idx], width,
                              PAD_VALUE[key], device)
        yield batch


def candidate_mask(is_kind, level_id, plpos, max_k, klast=None):
    """(B, L, L) pointer universe: earlier KIND tokens of the same level,
    1..max_k motifs back. With klast (案 2', --pointer-legal) the
    universe is the grammar-legal set: farther than the last emitted
    offset, so monotone refs hold by construction."""
    length = plpos.size(1)
    t_idx = torch.arange(length, device=plpos.device).view(1, length, 1)
    j_idx = torch.arange(length, device=plpos.device).view(1, 1, length)
    dist = plpos.unsqueeze(2) - plpos.unsqueeze(1)          # lpos[t]-lpos[j]
    cand = ((j_idx < t_idx)
            & (level_id.unsqueeze(2) == level_id.unsqueeze(1))
            & is_kind.unsqueeze(1).bool()
            & (dist >= 1) & (dist <= max_k))
    if klast is not None:
        cand = cand & (dist > klast.unsqueeze(2))
    return cand


def pointer_dist(plpos):
    """(B, L, L) lpos[t] - lpos[j], clamped to >= 0 for the bias table."""
    return (plpos.unsqueeze(2) - plpos.unsqueeze(1)).clamp(min=0)


# ── model ────────────────────────────────────

def sinusoidal_table(size, d_model):
    """Fixed sinusoidal encoding (size x d_model) — smooth in the
    index, so rarely-seen large counts still get sensible vectors."""
    position = torch.arange(size, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model))
    table = torch.zeros(size, d_model)
    table[:, 0::2] = torch.sin(position * div)
    table[:, 1::2] = torch.cos(position * div)[:, : d_model // 2]
    return table


def ref_vocab_mask(plpos, klast, vc: Vocab):
    """(B, L, V) additive mask for the vocabulary baseline: 0 everywhere,
    -inf on REF_k outside the pointer's legal universe
    klast < k <= lpos - 1 (== grammar_mask.legal_offsets). Where no k is
    legal every REF_k is masked — the REF type mask of --pointer-legal."""
    ks = torch.arange(1, vc.max_k + 1, device=plpos.device)
    legal = (ks > klast.unsqueeze(-1)) & (ks <= (plpos - 1).unsqueeze(-1))
    mask = torch.zeros(*plpos.shape, len(vc.vocab), device=plpos.device)
    ref_index = torch.tensor([vc.ref_ids[k] for k in range(1, vc.max_k + 1)],
                             device=plpos.device)
    mask[..., ref_index] = torch.where(legal, 0.0, NEG)
    return mask


def check_ref_mask(seqs, vc: Vocab, device, limit=200):
    """VM self-check: the torch mask equals grammar_mask.legal_offsets."""
    for seq in seqs[:limit]:
        aux = sequence_aux(seq, vc, False, True)
        plpos = torch.tensor([aux["plpos"]], device=device)
        klast = torch.tensor([aux["klast"]], device=device)
        mask = ref_vocab_mask(plpos, klast, vc)[0]
        ref = grammar_mask.legal_offsets(seq, vc.vocab, vc.max_k)
        for t in range(len(seq)):
            legal = sorted(k for k, i in vc.ref_ids.items() if mask[t, i] == 0)
            if legal != ref[t]:
                raise AssertionError(f"mask mismatch at {t}: {legal} vs {ref[t]}")
    return True


class ARBaseline(nn.Module):
    def __init__(self, vocab_size, max_len, pad_id,
                 d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=512, dropout=0.1,
                 use_struct=False, max_depth=16, struct_mode="learned",
                 use_pointer=False, n_types=0, pointer_legal=False,
                 pointer_dist_bias=False, max_k=0, dist_bias_mode="scalar",
                 ref_legal_mask=False):
        super().__init__()
        self.pad_id = pad_id
        self.max_k = max_k
        self.ref_legal_mask = ref_legal_mask
        self.max_len = max_len
        self.d_model = d_model
        self.use_struct = use_struct
        self.max_depth = max_depth
        self.use_pointer = use_pointer
        self.pointer_legal = pointer_legal
        self.pointer_dist_bias = pointer_dist_bias
        self.dist_bias_mode = dist_bias_mode
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        if use_struct:
            self.depth_emb = nn.Embedding(max_depth, d_model)
            self.struct_mode = struct_mode
            if struct_mode == "sinusoidal":
                self.lpos_emb = nn.Embedding.from_pretrained(
                    sinusoidal_table(max_len, d_model), freeze=True)
            elif struct_mode == "depth_only":
                self.lpos_emb = None          # 案 3': depth only, no level count
            else:
                self.lpos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        if use_pointer:
            self.type_head = nn.Linear(d_model, n_types)
            self.ptr_q = nn.Linear(d_model, d_model)
            self.ptr_k = nn.Linear(d_model, d_model)
            if pointer_dist_bias and dist_bias_mode == "context":
                # 案 2'': context-conditional distance logits — with a
                # zero content term this reduces to the baseline's
                # distance classification over the legal offsets.
                self.dist_head = nn.Linear(d_model, max_k + 1)
            elif pointer_dist_bias:
                self.dist_bias = nn.Embedding(max_k + 1, 1)
                nn.init.zeros_(self.dist_bias.weight)
        else:
            self.head = nn.Linear(d_model, vocab_size)

    def encode(self, x, depth=None, lpos=None):
        length = x.size(1)
        positions = torch.arange(length, device=x.device)
        hidden = self.tok_emb(x) + self.pos_emb(positions)[None]
        if self.use_struct:
            if depth is None or lpos is None:
                raise ValueError("struct-pos model needs depth/lpos inputs")
            hidden = hidden + self.depth_emb(depth.clamp(max=self.max_depth - 1))
            if self.lpos_emb is not None:
                hidden = hidden + self.lpos_emb(lpos.clamp(max=self.max_len - 1))
        causal = nn.Transformer.generate_square_subsequent_mask(
            length, device=x.device)
        return self.encoder(hidden, mask=causal,
                            src_key_padding_mask=x.eq(self.pad_id))

    def forward(self, x, depth=None, lpos=None):
        return self.head(self.encode(x, depth, lpos))

    def pointer_scores(self, hidden, cand, dist=None):
        """(B, L, L) log-domain scores, -inf outside the candidate set.
        dist (lpos[t]-lpos[j]) adds the learned distance bias (案 2')."""
        q, k = self.ptr_q(hidden), self.ptr_k(hidden)
        scores = q @ k.transpose(1, 2) / math.sqrt(self.d_model)
        if self.pointer_dist_bias:
            if dist is None:
                raise ValueError("dist bias needs the distance matrix")
            if self.dist_bias_mode == "context":
                per_k = self.dist_head(hidden)                 # (B, L, K+1)
                idx = dist.clamp(max=per_k.size(-1) - 1)       # (B, L, L)
                scores = scores + per_k.gather(-1, idx)
            else:
                scores = scores + self.dist_bias(
                    dist.clamp(max=self.dist_bias.num_embeddings - 1)).squeeze(-1)
        return scores.masked_fill(~cand, NEG)


# ── per-token log-likelihood (shared by train / score) ───────────

def token_logprobs(model, batch, vc: Vocab):
    """Returns (logp, valid): per-target-position log P(next token)
    and a mask of scored positions. Under --pointer,
    log P(REF_k) = log P(type=REF) + log P(pointer -> target)."""
    tokens = batch["tokens"]
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    valid = targets.ne(vc.pad)
    hidden = model.encode(inputs, *[
        None if batch.get(k) is None else batch[k][:, :-1]
        for k in ("depth", "lpos")])

    if not model.use_pointer:
        logits = model.head(hidden)
        if model.ref_legal_mask:
            logits = logits + ref_vocab_mask(
                batch["plpos"][:, :-1], batch["klast"][:, :-1], vc)
        logp = torch.log_softmax(logits, -1)
        return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1), valid

    type_of = torch.tensor(vc.type_of, device=tokens.device)
    type_targets = type_of[targets]
    plpos = batch["plpos"][:, :-1]
    cand = candidate_mask(batch["is_kind"][:, :-1], batch["level_id"][:, :-1],
                          plpos, vc.max_k,
                          batch["klast"][:, :-1] if model.pointer_legal else None)
    type_logits = model.type_head(hidden)
    if model.pointer_legal:
        # REF is impossible where no legal candidate exists: mask the
        # class instead of asking the model to learn to avoid it.
        no_cand = ~cand.any(-1)
        type_logits = type_logits.masked_fill(
            no_cand.unsqueeze(-1) & (torch.arange(type_logits.size(-1), device=tokens.device) == vc.ref_type),
            NEG)
    type_logp = torch.log_softmax(type_logits, -1)
    logp = type_logp.gather(-1, type_targets.unsqueeze(-1)).squeeze(-1)
    ref_target = batch["ref_target"][:, :-1]
    is_ref = ref_target.ge(0) & valid
    if is_ref.any():
        scores = model.pointer_scores(hidden, cand, pointer_dist(plpos))
        ptr_logp = torch.log_softmax(scores, -1)
        picked = ptr_logp.gather(-1, ref_target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        logp = logp + torch.where(is_ref, picked, torch.zeros_like(picked))
    return logp, valid


def run_epoch(model, seqs, auxes, batch_size, vc: Vocab, device,
              optimizer=None, rng=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_tokens = total_correct = 0
    with torch.set_grad_enabled(training):
        for batch in iter_batches(seqs, auxes, batch_size, vc.pad, device, rng):
            logp, valid = token_logprobs(model, batch, vc)
            n_tokens = int(valid.sum())
            loss = -(logp * valid).sum() / n_tokens
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += float(loss.detach()) * n_tokens
            total_tokens += n_tokens
            # accuracy: next-token argmax (type-level under --pointer)
            with torch.no_grad():
                targets = batch["tokens"][:, 1:]
                hidden = model.encode(batch["tokens"][:, :-1], *[
                    None if batch.get(k) is None else batch[k][:, :-1]
                    for k in ("depth", "lpos")])
                if model.use_pointer:
                    type_of = torch.tensor(vc.type_of, device=targets.device)
                    pred_ok = model.type_head(hidden).argmax(-1).eq(type_of[targets])
                else:
                    logits = model.head(hidden)
                    if model.ref_legal_mask:
                        logits = logits + ref_vocab_mask(
                            batch["plpos"][:, :-1], batch["klast"][:, :-1], vc)
                    pred_ok = logits.argmax(-1).eq(targets)
                total_correct += int((pred_ok & valid).sum())
    return total_loss / total_tokens, total_correct / total_tokens


@torch.no_grad()
def score_rows(model, rows, vc: Vocab, device, struct, pointer):
    """Teacher-forced per-sample scores with the per-token NLL trace."""
    model.eval()
    scored = []
    ref_id_set = set(vc.ref_ids.values())
    k_of_id = {i: k for k, i in vc.ref_ids.items()}
    for row in rows:
        seq = row["tokens"]
        aux = sequence_aux(seq, vc, struct, pointer)
        batch = {"tokens": _pad([seq], len(seq), vc.pad, device)}
        for key, values in aux.items():
            batch[key] = _pad([values], len(seq), PAD_VALUE[key], device)
        logp, _ = token_logprobs(model, batch, vc)
        token_nll = [-float(x) for x in logp[0]]
        n_tokens = len(token_nll)
        nll = sum(token_nll)
        # accuracy under --pointer is type-level; kept for continuity
        hidden = model.encode(batch["tokens"][:, :-1], *[
            None if batch.get(k) is None else batch[k][:, :-1]
            for k in ("depth", "lpos")])
        targets = batch["tokens"][0, 1:]
        ref_pos = [t for t, tok in enumerate(targets.tolist()) if tok in ref_id_set]
        ref_k = [k_of_id[int(targets[t])] for t in ref_pos]
        if model.use_pointer:
            type_of = torch.tensor(vc.type_of, device=device)
            acc = float(model.type_head(hidden)[0].argmax(-1).eq(type_of[targets]).float().mean())
            ref_correct = _pointer_ref_correct(model, hidden, batch, vc, ref_pos)
        else:
            logits = model.head(hidden)[0]
            if model.ref_legal_mask:
                logits = logits + ref_vocab_mask(
                    batch["plpos"][:, :-1], batch["klast"][:, :-1], vc)[0]
            acc = float(logits.argmax(-1).eq(targets).float().mean())
            ref_index = torch.tensor([vc.ref_ids[k] for k in range(1, vc.max_k + 1)],
                                     device=device)
            pred_k = logits[:, ref_index].argmax(-1) + 1
            ref_correct = [int(int(pred_k[t]) == k) for t, k in zip(ref_pos, ref_k)]
        scored.append({
            "sample_id": row.get("sample_id"), "seed": row.get("seed"),
            "n_tokens": n_tokens, "nll": nll, "nll_per_token": nll / n_tokens,
            "acc": acc, "token_nll": [round(x, 4) for x in token_nll],
            # edge accuracy: at REF targets, did the argmax reference match?
            "ref_pos": ref_pos, "ref_k": ref_k, "ref_correct": ref_correct,
        })
    return scored


def _pointer_ref_correct(model, hidden, batch, vc: Vocab, ref_pos):
    if not ref_pos:
        return []
    plpos = batch["plpos"][:, :-1]
    cand = candidate_mask(batch["is_kind"][:, :-1], batch["level_id"][:, :-1],
                          plpos, vc.max_k,
                          batch["klast"][:, :-1] if model.pointer_legal else None)
    scores = model.pointer_scores(hidden, cand, pointer_dist(plpos))[0]
    target = batch["ref_target"][0, :-1]
    pred = scores.argmax(-1)
    return [int(int(pred[t]) == int(target[t])) for t in ref_pos]


# ── sampling ─────────────────────────────────

def _top_k(logits, top_k):
    if top_k <= 0:
        return logits
    keep = torch.topk(logits, min(top_k, int((logits > NEG).sum()))).indices
    filtered = torch.full_like(logits, NEG)
    filtered[keep] = logits[keep]
    return filtered


@torch.no_grad()
def sample_stream(model, vc: Vocab, device, max_len, temperature, top_k,
                  state=None, struct=False, pointer=False):
    """state: optional grammar_mask.GrammarState for constrained decoding
    (every pick masked to the grammar; the tail of the budget spends on
    the shortest legal path to EOS). Under --pointer a REF is emitted by
    sampling the referenced motif and converting to REF_k."""
    model.eval()
    ids = [vc.bos]
    for _ in range(max_len - 1):
        if state is not None and \
                (max_len - len(ids)) <= state.min_close_cost() + 2:
            next_id = state.forced_close_id()
        else:
            aux = sequence_aux(ids, vc, struct, pointer)
            x = torch.tensor([ids], dtype=torch.long, device=device)
            depth = lpos = None
            if struct:
                depth = torch.tensor([aux["depth"]], device=device)
                lpos = torch.tensor([aux["lpos"]], device=device)
            hidden = model.encode(x, depth, lpos)
            allowed = None if state is None else set(state.allowed_ids())

            if not model.use_pointer:
                logits = model.head(hidden[0, -1]) / temperature
                logits[vc.pad] = NEG
                logits[vc.bos] = NEG
                if model.ref_legal_mask:
                    lpos_t, klast_t = aux["plpos"][-1], aux["klast"][-1]
                    for k, ref_id in vc.ref_ids.items():
                        if not (klast_t < k <= lpos_t - 1):
                            logits[ref_id] = NEG
                if allowed is not None:
                    mask = torch.full_like(logits, NEG)
                    mask[list(allowed)] = 0.0
                    logits = logits + mask
                next_id = int(torch.multinomial(torch.softmax(_top_k(logits, top_k), -1), 1))
            else:
                next_id = _sample_pointer_step(
                    model, hidden, aux, vc, allowed, temperature, top_k, device)

        ids.append(next_id)
        if state is not None:
            state.push(next_id)
        if next_id == vc.eos:
            break
    return ids


def _sample_pointer_step(model, hidden, aux, vc, allowed, temperature, top_k, device):
    t = hidden.size(1) - 1
    type_logits = model.type_head(hidden[0, -1]) / temperature
    type_logits[vc.type_of[vc.pad]] = NEG
    type_logits[vc.type_of[vc.bos]] = NEG

    # candidate motifs for a REF at this step
    is_kind = torch.tensor([aux["is_kind"]], device=device)
    level_id = torch.tensor([aux["level_id"]], device=device)
    plpos = torch.tensor([aux["plpos"]], device=device)
    klast = torch.tensor([aux["klast"]], device=device) if model.pointer_legal else None
    cand = candidate_mask(is_kind, level_id, plpos, vc.max_k, klast)[0, t]
    if model.pointer_legal and not cand.any():
        type_logits[vc.ref_type] = NEG        # impossible class (see token_logprobs)
    if allowed is not None:
        # grammar: monotone refs & only offsets the grammar allows
        allowed_ks = {k for k, i in vc.ref_ids.items() if i in allowed}
        for j in torch.nonzero(cand).flatten().tolist():
            if (aux["plpos"][t] - aux["plpos"][j]) not in allowed_ks:
                cand[j] = False
        allowed_types = {vc.type_of[i] for i in allowed if i not in vc.ref_ids.values()}
        if cand.any():
            allowed_types.add(vc.ref_type)
        mask = torch.full_like(type_logits, NEG)
        mask[list(allowed_types)] = 0.0
        type_logits = type_logits + mask

    chosen = int(torch.multinomial(torch.softmax(_top_k(type_logits, top_k), -1), 1))
    if chosen != vc.ref_type:
        return vc.token_of_type[chosen]
    if not cand.any():
        # unconstrained diagnostic: an impossible reference stays a violation
        return vc.ref_ids[1]
    scores = model.pointer_scores(
        hidden, cand.view(1, 1, -1).expand(1, hidden.size(1), -1),
        pointer_dist(plpos))[0, t]
    scores = scores / temperature
    j = int(torch.multinomial(torch.softmax(_top_k(scores, top_k), -1), 1))
    k = aux["plpos"][t] - aux["plpos"][j]
    return vc.ref_ids[k]


# ── main ─────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(description="Train the AR baseline.")
    parser.add_argument("--train", default="/content/train.jsonl")
    parser.add_argument("--val", default="/content/val.jsonl")
    parser.add_argument("--vocab", default="/content/vocab.json")
    parser.add_argument("--meta", default="/content/meta.json")
    parser.add_argument("--out", default="/content/run1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=0,
                        help="early stop after N epochs without val "
                             "improvement (0 = off)")
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
    parser.add_argument("--constrained", action="store_true",
                        help="grammar-constrained sampling")
    parser.add_argument("--struct-pos", action="store_true",
                        help="案 1: explicit level-position / depth embeddings")
    parser.add_argument("--struct-pos-mode", default="learned",
                        choices=["learned", "sinusoidal", "depth_only"])
    parser.add_argument("--pointer", action="store_true",
                        help="案 2: pointer-style reference head")
    parser.add_argument("--pointer-legal", action="store_true",
                        help="案 2': pointer universe = grammar-legal "
                             "candidates (farther than the last offset)")
    parser.add_argument("--pointer-dist-bias", action="store_true",
                        help="案 2': learned per-distance bias on scores")
    parser.add_argument("--pointer-dist-bias-mode", default="scalar",
                        choices=["scalar", "context"],
                        help="scalar (案 2') or context-conditional "
                             "distance logits (案 2'')")
    parser.add_argument("--ref-legal-mask", action="store_true",
                        help="B control: vocabulary baseline whose REF_k "
                             "logits are masked to the pointer's legal "
                             "universe (train / score / sample)")
    parser.add_argument("--test", default=None,
                        help="held-out split to score after training "
                             "(test_scores.jsonl); never used for early stopping")
    parser.add_argument("--sample-seed", type=int, default=None,
                        help="RNG seed for sampling (default: --seed); "
                             "pair it across models")
    parser.add_argument("--constrained-samples", type=int, default=0,
                        help="also write N grammar-constrained samples to "
                             "samples_constrained.json")
    return parser


def build_model(vc: Vocab, meta: dict, args, gen_max_len: int):
    """Model matching a flag set — shared by training and by sampling
    from a saved checkpoint."""
    return ARBaseline(
        vocab_size=len(vc.vocab), max_len=max(meta["max_len"], gen_max_len),
        pad_id=vc.pad, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
        dropout=args.dropout, use_struct=args.struct_pos,
        struct_mode=args.struct_pos_mode, use_pointer=args.pointer,
        n_types=len(vc.type_names), pointer_legal=args.pointer_legal,
        pointer_dist_bias=args.pointer_dist_bias, max_k=vc.max_k,
        dist_bias_mode=args.pointer_dist_bias_mode,
        ref_legal_mask=getattr(args, "ref_legal_mask", False),
    )


def main(argv=None):
    parser = build_parser()
    # parse_known_args: `colab exec` runs this file inside a notebook
    # kernel whose sys.argv carries kernel flags (-f /path/kernel.json).
    args, _ = parser.parse_known_args(argv)

    needs_grammar = (args.struct_pos or args.pointer or args.constrained
                     or args.ref_legal_mask or args.constrained_samples)
    if needs_grammar and grammar_mask is None:
        raise SystemExit("this configuration requires grammar_mask.py")
    aux_ctx = args.pointer or args.ref_legal_mask   # pointer context needed

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vc = Vocab(json.loads(Path(args.vocab).read_text(encoding="utf-8")))
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    train_seqs = load_streams(args.train)
    val_rows = load_rows(args.val)
    val_seqs = [row["tokens"] for row in val_rows]
    train_aux = [sequence_aux(s, vc, args.struct_pos, aux_ctx) for s in train_seqs]
    val_aux = [sequence_aux(s, vc, args.struct_pos, aux_ctx) for s in val_seqs]

    gen_max_len = args.gen_max_len or 2 * meta["max_len"]
    model = build_model(vc, meta, args, gen_max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history, best_val, best_epoch, since_best = [], float("inf"), 0, 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_seqs, train_aux, args.batch_size, vc, device,
            optimizer=optimizer, rng=rng)
        val_loss, val_acc = run_epoch(
            model, val_seqs, val_aux, args.batch_size, vc, device)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "val_loss": val_loss,
                        "val_acc": val_acc})
        print(f"epoch {epoch:3d}  train {train_loss:.4f}/{train_acc:.3f}  "
              f"val {val_loss:.4f}/{val_acc:.3f}")
        if val_loss < best_val:
            best_val, best_epoch, since_best = val_loss, epoch, 0
            torch.save(model.state_dict(), out / "model.pt")
        else:
            since_best += 1
            if args.patience and since_best >= args.patience:
                print(f"early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    model.load_state_dict(torch.load(out / "model.pt", map_location=device))

    val_scores = score_rows(model, val_rows, vc, device, args.struct_pos, aux_ctx)
    (out / "val_scores.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in val_scores), encoding="utf-8")
    if args.test:
        test_scores = score_rows(model, load_rows(args.test), vc, device,
                                 args.struct_pos, aux_ctx)
        (out / "test_scores.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in test_scores), encoding="utf-8")
        total_nll = sum(r["nll"] for r in test_scores)
        total_tok = sum(r["n_tokens"] for r in test_scores)
        print(f"test NLL/token {total_nll / total_tok:.4f}")

    sample_seed = args.seed if args.sample_seed is None else args.sample_seed

    def draw(n, constrained):
        torch.manual_seed(sample_seed)      # paired sampling RNG across models
        return [
            sample_stream(model, vc, device, gen_max_len, args.temperature, args.top_k,
                          state=(grammar_mask.GrammarState(vc.vocab) if constrained else None),
                          struct=args.struct_pos, pointer=aux_ctx)
            for _ in range(n)
        ]

    samples = draw(args.num_samples, args.constrained)
    (out / "samples.json").write_text(json.dumps({
        "best_val_loss": best_val, "best_epoch": best_epoch,
        "epochs_run": len(history), "config": vars(args), "samples": samples,
    }, indent=2), encoding="utf-8")
    print(f"wrote {len(samples)} samples to {out / 'samples.json'}")
    if args.constrained_samples:
        constrained = draw(args.constrained_samples, True)
        (out / "samples_constrained.json").write_text(json.dumps({
            "config": vars(args), "samples": constrained}, indent=2), encoding="utf-8")
        print(f"wrote {len(constrained)} constrained samples")


if __name__ == "__main__":
    main()
