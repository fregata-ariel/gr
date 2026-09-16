"""Model-free, grammar-aware baseline code length for token streams.

Design: docs/design/mixture_doe.md section 9.  The three CFG families
have different intrinsic information content, so raw model NLL mixes
family difficulty with model quality.  This module fits a frequency
model that *knows the grammar* on a training split and assigns every
token a code length (in nats) without any neural network:

  * the REF/structural type decision is a per-key frequency table,
    key = (most recent KIND_* token, REF tokens emitted since it);
  * structural tokens (KIND_*, LOOP_START, LOOP_END, EOS) come from an
    add-alpha trigram over the structural subsequence only;
  * a REF offset comes from ``controlled_eval.FrequencyBaselines``'s
    conditional P(k - k_last | n_legal) table, with the legal universe
    from ``grammar_mask.legal_offsets``.

Scored test rows use the same JSON row shape as the model score files
(``sample_id``, ``seed``, ``n_tokens``, ``nll``, ``nll_per_token``,
``token_nll``) so that ``mixture_doe collect --baseline-dir`` can form
the excess NLL = model NLL - baseline NLL.

Torch-free: standard library plus ``training.controlled_eval`` and
``training.grammar_mask``.
"""

from __future__ import annotations

import argparse
import json
from math import log
from pathlib import Path

from training.controlled_eval import FrequencyBaselines
from training.grammar_mask import legal_offsets, pointer_context

BASELINE_NAME = "freq_trigram"
_LOOP_TOKENS = ("LOOP_START", "LOOP_END")
_CLASSES = ("KIND", "REF", "LOOP", "EOS")


def _is_kind(name: str) -> bool:
    return name.startswith("KIND_")


def _is_ref(name: str) -> bool:
    return name.startswith("REF_")


def _is_structural(name: str) -> bool:
    return name == "EOS" or name in _LOOP_TOKENS or name.startswith("KIND_")


def ref_values(tokens: list[int], vocab: dict[str, int],
               max_k: int) -> list[dict]:
    """Per REF position (full-stream index i >= 1): k, rel and n_legal.

    Computed exactly as ``controlled_eval.ref_records`` does: the
    pointer context and the legal-offset universe are evaluated on the
    prefix, i.e. at index ``i - 1`` of the post-state lists.
    """
    names = {i: t for t, i in vocab.items()}
    context = pointer_context(tokens, vocab)
    legal = legal_offsets(tokens, vocab, max_k)
    out: list[dict] = []
    for i in range(1, len(tokens)):
        name = names.get(tokens[i], "")
        if not _is_ref(name):
            continue
        k = int(name[4:])
        klast = context["klast"][i - 1]
        out.append({"pos": i, "k": k, "rel": k - klast,
                    "n_legal": len(legal[i - 1])})
    return out


class InfoBaseline:
    """Grammar-aware frequency code fitted on a training split.

    ``class_nll`` decomposes a stream's NLL into the four token classes
    KIND / REF / LOOP / EOS (which partition the positions and sum to
    ``nll``) plus TYPE, the summed -log of the REF/structural type
    decision.  TYPE is *also* contained inside each of the four class
    sums, so it is reported for diagnostics and must not be added to
    them.
    """

    def __init__(self, train_rows: list[dict], vocab: dict[str, int],
                 max_k: int, alpha: float = 0.5) -> None:
        self.vocab = dict(vocab)
        self.max_k = int(max_k)
        self.alpha = float(alpha)
        self._names = {i: t for t, i in vocab.items()}
        self.structural_names = tuple(
            sorted(name for name in vocab if _is_structural(name))
        )
        self._v_s = len(self.structural_names)
        self.type_total: dict[tuple[str, int], int] = {}
        self.type_ref: dict[tuple[str, int], int] = {}
        self.uni: dict[str, int] = {}
        self.bi: dict[tuple[str, str], int] = {}
        self.bi_ctx: dict[str, int] = {}
        self.tri: dict[tuple[str, str, str], int] = {}
        self.tri_ctx: dict[tuple[str, str], int] = {}
        self.struct_total = 0
        for row in train_rows:
            self._observe(row["tokens"])
        self.freq = FrequencyBaselines(train_rows, vocab, self.max_k,
                                       self.alpha)

    @classmethod
    def fit(cls, train_rows: list[dict], vocab: dict[str, int],
            max_k: int, alpha: float = 0.5) -> "InfoBaseline":
        return cls(train_rows, vocab, max_k, alpha)

    # ── fitting ──────────────────────────────

    def _observe(self, tokens: list[int]) -> None:
        key: tuple[str, int] = ("BOS", 0)
        for i in range(1, len(tokens)):
            name = self._names[tokens[i]]
            self.type_total[key] = self.type_total.get(key, 0) + 1
            if _is_ref(name):
                self.type_ref[key] = self.type_ref.get(key, 0) + 1
            if _is_kind(name):
                key = (name, 0)
            elif _is_ref(name):
                key = (key[0], key[1] + 1)

        seq = ["BOS"]
        for i in range(1, len(tokens)):
            name = self._names[tokens[i]]
            if _is_structural(name):
                seq.append(name)
        for j in range(1, len(seq)):
            c = seq[j]
            b = seq[j - 1]
            a = seq[j - 2] if j >= 2 else None
            self.struct_total += 1
            self.uni[c] = self.uni.get(c, 0) + 1
            self.bi[(b, c)] = self.bi.get((b, c), 0) + 1
            self.bi_ctx[b] = self.bi_ctx.get(b, 0) + 1
            if a is not None:
                self.tri[(a, b, c)] = self.tri.get((a, b, c), 0) + 1
                self.tri_ctx[(a, b)] = self.tri_ctx.get((a, b), 0) + 1

    # ── probabilities ────────────────────────

    def type_probability(self, key: tuple[str, int]) -> float:
        """P(REF | key), add-alpha over the two type outcomes."""
        total = self.type_total.get(key, 0)
        refs = self.type_ref.get(key, 0)
        return (refs + self.alpha) / (total + 2.0 * self.alpha)

    def structural_probability(self, a: str | None, b: str, c: str) -> float:
        """P_trigram(c | a, b) over the structural vocabulary, add-alpha."""
        alpha = self.alpha
        v = self._v_s
        if a is not None and self.tri_ctx.get((a, b), 0) > 0:
            return ((self.tri.get((a, b, c), 0) + alpha)
                    / (self.tri_ctx[(a, b)] + alpha * v))
        if self.bi_ctx.get(b, 0) > 0:
            return ((self.bi.get((b, c), 0) + alpha)
                    / (self.bi_ctx[b] + alpha * v))
        return ((self.uni.get(c, 0) + alpha)
                / (self.struct_total + alpha * v))

    # ── scoring ──────────────────────────────

    def score(self, tokens: list[int]) -> dict:
        """Code length of one stream: per-token NLL and class sums."""
        if not tokens:
            raise ValueError("empty token stream")
        refs = {rec["pos"]: rec
                for rec in ref_values(tokens, self.vocab, self.max_k)}
        key: tuple[str, int] = ("BOS", 0)
        seq = ["BOS"]
        token_nll: list[float] = []
        class_nll = {name: 0.0 for name in (*_CLASSES, "TYPE")}
        for i in range(1, len(tokens)):
            name = self._names[tokens[i]]
            p_ref = self.type_probability(key)
            if _is_ref(name):
                type_p = p_ref
                rec = refs[i]
                nll = -log(type_p) + self.freq.conditional_nll(
                    rec["rel"], rec["n_legal"])
                category = "REF"
            else:
                type_p = 1.0 - p_ref
                a = seq[-2] if len(seq) >= 2 else None
                nll = -log(type_p) - log(
                    self.structural_probability(a, seq[-1], name))
                if _is_kind(name):
                    category = "KIND"
                elif name in _LOOP_TOKENS:
                    category = "LOOP"
                else:
                    category = "EOS"
                seq.append(name)
            class_nll[category] += nll
            class_nll["TYPE"] += -log(type_p)
            token_nll.append(nll)
            if _is_kind(name):
                key = (name, 0)
            elif _is_ref(name):
                key = (key[0], key[1] + 1)
        n_tokens = len(tokens) - 1
        nll = sum(token_nll)
        return {
            "n_tokens": n_tokens,
            "nll": nll,
            "nll_per_token": nll / n_tokens if n_tokens else 0.0,
            "token_nll": token_nll,
            "class_nll": class_nll,
        }


def score_rows(baseline: InfoBaseline, rows: list[dict]) -> list[dict]:
    """Score jsonl rows, emitting the model-score-file row shape."""
    out: list[dict] = []
    for row in rows:
        result = baseline.score(row["tokens"])
        out.append({
            "sample_id": row["sample_id"],
            "seed": row["seed"],
            "n_tokens": result["n_tokens"],
            "nll": result["nll"],
            "nll_per_token": result["nll_per_token"],
            "token_nll": result["token_nll"],
            "class_nll": result["class_nll"],
            "baseline": BASELINE_NAME,
        })
    return out


# ── CLI ──────────────────────────────────────

def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def summarize_rows(results: list[dict]) -> dict:
    """Aggregate NLL/token overall and per class (TYPE included)."""
    n_tokens = sum(r["n_tokens"] for r in results)
    nll = sum(r["nll"] for r in results)
    class_nll = {name: 0.0 for name in (*_CLASSES, "TYPE")}
    for r in results:
        for name, value in r["class_nll"].items():
            class_nll[name] += value
    return {
        "n_tokens": n_tokens,
        "nll": nll,
        "nll_per_token": nll / n_tokens if n_tokens else 0.0,
        "class_nll_per_token": {
            name: (value / n_tokens if n_tokens else 0.0)
            for name, value in class_nll.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.info_baseline")
    sub = parser.add_subparsers(dest="command", required=True)

    score_parser = sub.add_parser("score")
    score_parser.add_argument("--train", type=Path, required=True)
    score_parser.add_argument("--vocab", type=Path, required=True)
    score_parser.add_argument("--meta", type=Path, required=True)
    score_parser.add_argument("--test", type=Path, required=True)
    score_parser.add_argument("--out", type=Path, required=True)
    score_parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args(argv)

    vocab = _read_json(args.vocab)
    meta = _read_json(args.meta)
    baseline = InfoBaseline.fit(_read_jsonl(args.train), vocab,
                                int(meta["max_offset"]), args.alpha)
    results = score_rows(baseline, _read_jsonl(args.test))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True) for row in results]
    args.out.write_text("\n".join(lines) + ("\n" if lines else ""),
                        encoding="utf-8")

    summary = summarize_rows(results)
    parts = " ".join(
        f"{name} {summary['class_nll_per_token'][name]:.4f}"
        for name in (*_CLASSES, "TYPE")
    )
    print(f"nll/token {summary['nll_per_token']:.4f}  {parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
