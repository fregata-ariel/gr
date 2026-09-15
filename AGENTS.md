# AGENTS.md — rules for delegated coding agents (Codex / OpenCode)

Read this before touching the repository. `CLAUDE.md` describes the architecture;
`docs/handoff.md` and `docs/design/*.md` hold the decisions. Design documents win over
older docs, and a document's final "決定" section wins over its earlier sections.

## Environment and commands

- Python 3.13, managed with `uv`. Run everything through `uv run ...`.
- Verify before reporting: `uv run pytest -q` (all green, suite under ~10 s),
  `uv run ty check` (All checks passed), `git diff --check` (clean).
- Never install packages. Runtime dependencies are only `networkx` and `matplotlib`;
  tests and local tools never import `torch`. Do not edit `pyproject.toml` or `uv.lock`.
- `training/train_ar.py` and `training/grammar_mask.py` are the only files uploaded to
  Colab; `train_ar.py` is torch-only and must not import `cfg_reducer`.

## Code conventions

- Data types are frozen dataclasses (see `cfg_reducer/types.py`); prefer tuples over lists
  inside them. Keep everything JSON-serialisable with sorted keys and no NaN/Infinity.
- Determinism: the only randomness is the `random.Random(seed)` passed in; never iterate a
  `set` into `choice`/`sample`/output order (sort first). Outputs must not depend on
  `PYTHONHASHSEED`.
- Do not change `cfg_reducer/generate.py` (v1 generator), `cfg_reducer/store.py`
  (provenance / `sample_id`), or the legacy `build_dataset` path; they must stay
  byte-identical for existing datasets.
- Any change to a generator family's algorithm or RNG consumption changes the dataset
  `version` (the git commit). Never alter a family's output under an existing version;
  datasets built by different commits are not mixed in one experiment.
- Families for generator v2 are plugins: add `cfg_reducer/families/<name>.py` and one
  registration line in `families/__init__.py`; never add family branches to
  `cfg_reducer/dataset.py`, `dataset_v2.py`, or `training/controlled_eval.py`.
- Documentation is Japanese; code, identifiers, and commit messages are English.
- Do not commit. Report changed files and the verification commands you ran.

## Working style

- Implement exactly the task you were given (one or two numbered tasks from the design
  document). If the design is ambiguous or you find a bug elsewhere, stop and report
  instead of widening the change.
- Keep new tests fast; put large parameter sweeps behind `GR_SLOW_TESTS=1`.
