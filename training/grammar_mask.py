"""
Incremental grammar state for constrained decoding — torch-free.

Mirrors cfg_reducer.model_input.detokenize: a sampler that only ever
picks from allowed_ids() produces streams that the strict parser
accepts, by construction.  Uploaded to the Colab VM next to
train_ar.py (which imports it by module name); locally it is a plain
member of the training package, so the consistency tests need no torch.
"""

from __future__ import annotations


class GrammarState:
    """Parser state after BOS; feed every sampled token id via push()."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self._names = {i: t for t, i in vocab.items()}
        self._kind_ids = [
            vocab[t] for t in vocab if t.startswith("KIND_")
        ]
        self._loop_kind = vocab["KIND_LOOP"]
        self._ref_id_of = {
            int(t.split("_")[1]): i
            for t, i in vocab.items() if t.startswith("REF_")
        }
        self._loop_start = vocab["LOOP_START"]
        self._loop_end = vocab["LOOP_END"]
        self._eos = vocab["EOS"]

        self._counts = [0]          # motifs emitted per open level
        self._pending = [False]     # last KIND was loop, LOOP_START due
        self._refs: list[int] | None = None   # offsets of current motif
        self.done = False

    # ── queries ──────────────────────────────

    @property
    def depth(self) -> int:
        return len(self._counts) - 1

    @property
    def level_count(self) -> int:
        """Motifs emitted so far at the current level (matches
        structural_positions for the same prefix)."""
        return self._counts[-1]

    def allowed_ids(self) -> list[int]:
        """Token ids that keep the stream well-formed."""
        if self.done:
            return []

        allowed: list[int] = []
        depth = len(self._counts) - 1

        if self._refs is not None:
            last = self._refs[-1] if self._refs else 0
            allowed.extend(
                ref_id
                for k, ref_id in self._ref_id_of.items()
                if last < k <= self._counts[-1] - 1
            )

        if self._pending[-1]:
            allowed.append(self._loop_start)
            return allowed

        allowed.extend(self._kind_ids)
        if depth > 0:
            allowed.append(self._loop_end)
        else:
            allowed.append(self._eos)
        return allowed

    def min_close_cost(self) -> int:
        """Tokens needed to finish the stream from here (incl. EOS)."""
        depth = len(self._counts) - 1
        return (2 if self._pending[-1] else 0) + depth + 1

    def forced_close_id(self) -> int:
        """Next token on the shortest legal path to EOS."""
        if self._pending[-1]:
            return self._loop_start
        if len(self._counts) > 1:
            return self._loop_end
        return self._eos

    # ── transition ───────────────────────────

    def push(self, token_id: int) -> None:
        name = self._names[token_id]

        if name.startswith("KIND_"):
            self._counts[-1] += 1
            self._refs = []
            self._pending[-1] = token_id == self._loop_kind
        elif name.startswith("REF_"):
            assert self._refs is not None
            self._refs.append(int(name.split("_")[1]))
        elif name == "LOOP_START":
            self._pending[-1] = False
            self._refs = None
            self._counts.append(0)
            self._pending.append(False)
        elif name == "LOOP_END":
            self._counts.pop()
            self._pending.pop()
            self._refs = None
        elif name == "EOS":
            self.done = True
        else:   # BOS / PAD never reach push in a constrained sampler
            raise ValueError(f"unexpected token {name}")


# ── structural positions (案 1: explicit level-position / depth) ──

def structural_positions(token_ids: list[int], vocab: dict[str, int]
                         ) -> tuple[list[int], list[int]]:
    """
    Per-token (depth, level_count) AFTER consuming that token — the
    counter the model otherwise has to infer implicitly.

    depth        nesting depth the stream is in after the token
    level_count  motifs emitted so far at the current level

    Tolerant by design (never raises), so it can also annotate
    malformed prefixes during unconstrained diagnostic sampling:
    a stray LOOP_END at depth 0 and REF/EOS/PAD leave the state as is.
    """
    names = {i: t for t, i in vocab.items()}
    counts = [0]
    depths: list[int] = []
    level_counts: list[int] = []
    for token_id in token_ids:
        name = names.get(token_id, "")
        if name.startswith("KIND_"):
            counts[-1] += 1
        elif name == "LOOP_START":
            counts.append(0)
        elif name == "LOOP_END" and len(counts) > 1:
            counts.pop()
        depths.append(len(counts) - 1)
        level_counts.append(counts[-1])
    return depths, level_counts


# ── pointer context (案 2: pointer-style reference head) ─────────

def pointer_context(token_ids: list[int], vocab: dict[str, int]) -> dict:
    """
    Per-token structural context for the pointer head, all as
    post-state lists aligned with token_ids (tolerant, never raises):

    is_kind    token is a KIND_* (a motif; the only pointer candidates)
    level_id   index of the enclosing LOOP_START token (-1 at top level)
    lpos       motifs emitted at the current level so far, i.e. a KIND
               token's own level position is lpos - 1
    klast      largest REF offset emitted for the current motif (0 if none)
    ref_target for position t whose NEXT token is REF_k: the token index
               of the referenced KIND token (same level, level position
               (lpos_t - 1) - k), else -1

    Candidate universe for position t (predicting t+1): earlier KIND
    tokens j with level_id[j] == level_id[t]. The grammar adds
    lpos_j <= lpos_t - 2 and (lpos_t - 1) - lpos_j > klast_t.
    """
    names = {i: t for t, i in vocab.items()}
    counts = [0]
    level_stack = [-1]
    is_kind, level_id, lpos, klast = [], [], [], []
    current_klast = 0
    # (level_id, level_position) -> token index of that KIND token
    kind_at: dict[tuple[int, int], int] = {}

    for idx, token_id in enumerate(token_ids):
        name = names.get(token_id, "")
        kind = name.startswith("KIND_")
        if kind:
            counts[-1] += 1
            current_klast = 0
            kind_at[(level_stack[-1], counts[-1] - 1)] = idx
        elif name.startswith("REF_"):
            current_klast = int(name[4:])
        elif name == "LOOP_START":
            counts.append(0)
            level_stack.append(idx)
            current_klast = 0
        elif name == "LOOP_END" and len(counts) > 1:
            counts.pop()
            level_stack.pop()
            current_klast = 0
        is_kind.append(kind)
        level_id.append(level_stack[-1])
        lpos.append(counts[-1])
        klast.append(current_klast)

    ref_target = [-1] * len(token_ids)
    for t in range(len(token_ids) - 1):
        name = names.get(token_ids[t + 1], "")
        if name.startswith("REF_"):
            k = int(name[4:])
            ref_target[t] = kind_at.get((level_id[t], (lpos[t] - 1) - k), -1)

    return {"is_kind": is_kind, "level_id": level_id, "lpos": lpos,
            "klast": klast, "ref_target": ref_target}
