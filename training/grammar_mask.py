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
