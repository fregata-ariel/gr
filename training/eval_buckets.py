"""Torch-free, provenance-checked long-sequence evaluation (P3)."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
from random import Random
from statistics import mean, pstdev

from cfg_reducer.families.mixture import component_for
from cfg_reducer.generate_v2 import spec_from_json
from cfg_reducer.model_input import build_vocab
from training import controlled_eval as ce, eval_samples, grammar_mask
from training.data_utils import read_jsonl
from training.info_baseline import InfoBaseline, score_rows

ID_NS = (8, 12, 16, 24, 32, 48, 64, 96, 128)
ALL_NS = (*ID_NS, 192, 256)


def read_json(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}: {path}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_rows(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        sid = row["sample_id"]
        if sid in result:
            raise ValueError(f"duplicate sample_id: {sid}")
        result[sid] = row
    return result


def manifest_rows(dataset: Path, split: str) -> dict[str, dict]:
    manifest = read_json(dataset / "manifest.json")
    result = unique_rows(manifest["splits"][split]["samples"])
    for row in result.values():
        spec_data = row.get("requested", manifest["generator"]["config"])["spec"]
        spec = spec_from_json(spec_data)
        family = spec.family
        if family == "mixture":
            family = spec_data["params"]["components"][component_for(spec, row["seed"])]["family"]
        row.update(n=spec.num_nodes, family=family)
    return result


def checked_split(bundle: Path, dataset: Path, split: str) -> tuple[list[dict], dict, dict]:
    index = read_json(bundle / "evaluation_index.json")
    if index["version"] != 1 or index["sources"][split] != digest(dataset / "manifest.json"):
        raise ValueError(f"source manifest hash mismatch: {bundle} {split}")
    raw = manifest_rows(dataset, split)
    rows = read_jsonl(bundle / f"{split}.jsonl")
    actual = unique_rows(rows)
    retained = {sid for sid, row in index["samples"].items() if row["split"] == split}
    exclusions = unique_rows([row for row in index["exclusions"] if row["split"] == split])
    if set(actual) != retained or retained & exclusions.keys() or retained | exclusions.keys() != raw.keys():
        raise ValueError(f"token/index/manifest IDs mismatch: {bundle} {split}")
    for row in rows:
        if row["seed"] != raw[row["sample_id"]]["seed"]:
            raise ValueError("token seed differs from manifest")
    for sid, row in exclusions.items():
        if row["seed"] != raw[sid]["seed"] or row["reason"] != "over_window":
            raise ValueError("unexpected exclusion: unlimited evaluation requires REF-window exclusions")
    return rows, raw, exclusions


def check_bundles(index: dict, root: Path) -> None:
    """Check every split, common vocabulary, and identical source train/val bytes."""
    source = root / index["source_bundle"]
    vocab = read_json(source / "vocab.json")
    if vocab != build_vocab(128):
        raise ValueError("pretrain requires the canonical REF window 128 vocabulary")
    version = read_json(root / index["source_dataset"] / "manifest.json")["generator"]["version"]
    for bucket in index["buckets"]:
        bundle = root / bucket["bundle"]
        meta = read_json(bundle / "meta.json")
        bundle_index = read_json(bundle / "evaluation_index.json")
        if meta["max_offset"] != 128 or ("sources" in meta and meta["sources"] != bundle_index["sources"]):
            raise ValueError("bundle meta window/sources mismatch")
        if read_json(root / bucket["dataset"] / "manifest.json")["generator"]["version"] != version:
            raise ValueError("dataset generator versions differ")
        if read_json(bundle / "vocab.json") != vocab:
            raise ValueError(f"vocab mismatch: {bundle}")
        for split in ("train", "val", "test"):
            dataset = index["source_dataset"] if split != "test" else bucket["dataset"]
            rows, raw, _ = checked_split(bundle, root / dataset, split)
            if split != "test" and digest(bundle / f"{split}.jsonl") != digest(source / f"{split}.jsonl"):
                raise ValueError(f"source {split} differs: {bundle}")
            if split == "test" and any(r["n"] != bucket["n"] for r in raw.values()):
                raise ValueError("bucket n differs from manifest")
            unique_rows(rows)
        all_ids = []
        for split in ("train", "val", "test"):
            all_ids.extend(read_jsonl(bundle / f"{split}.jsonl"))
        unique_rows(all_ids)


def paired_ci(left: list[dict], right: list[dict]) -> dict:
    a, b = unique_rows(left), unique_rows(right)
    if a.keys() != b.keys():
        raise ValueError("paired IDs mismatch")
    pairs = []
    for sid in sorted(a):
        if a[sid]["n_tokens"] != b[sid]["n_tokens"]:
            raise ValueError("paired token counts mismatch")
        pairs.append((a[sid]["nll"] - b[sid]["nll"], a[sid]["n_tokens"]))
    if not pairs:
        return {"samples": 0, "delta_nats_per_token": None, "ci95": None}
    rng = Random(0)
    draws = []
    for _ in range(1000):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        draws.append(sum(d for d, _ in sample) / sum(t for _, t in sample))
    draws.sort()
    return {"samples": len(pairs), "delta_nats_per_token": sum(d for d, _ in pairs) / sum(t for _, t in pairs),
            "ci95": [draws[25], draws[975]], "bootstrap_seed": 0, "bootstrap_replicates": 1000}


def ref_metrics(rows: list[dict], tokens: dict, vocab: dict) -> dict:
    records = ce.ref_records(tokens, rows, vocab, 128)
    table = ce.by_k(records, min_count=30)
    legality: dict[int, list[bool]] = {}
    for row in rows:
        has = [key in row for key in ("ref_pred_k", "ref_pred_legal")]
        if not any(has):
            continue
        positions = row.get("ref_pos", [])
        expected = [i for i, tok in enumerate(tokens[row["sample_id"]][1:]) if tok >= vocab["REF_1"]]
        if positions != expected or not all(has) or any(len(row[key]) != len(positions) for key in ("ref_pred_k", "ref_pred_legal")):
            raise ValueError("REF diagnostic length mismatch")
        ctx = grammar_mask.pointer_context(tokens[row["sample_id"]], vocab)
        for pos, k, legal in zip(positions, row["ref_pred_k"], row["ref_pred_legal"]):
            if type(k) is not int or not 1 <= k <= 128 or type(legal) is not bool:
                raise ValueError("invalid REF diagnostics")
            if legal != (ctx["klast"][pos] < k <= ctx["lpos"][pos] - 1):
                raise ValueError("REF legality disagrees with prefix")
            gold_k = tokens[row["sample_id"]][pos + 1] - vocab["REF_1"] + 1
            legality.setdefault(gold_k, []).append(legal)
    for k, entry in table.items():
        entry["low_count"] = entry["n"] < 30
        values = legality.get(k, [])
        entry["teacher_forced_legality"] = {
            "diagnosed": len(values), "gold_refs": entry["n"],
            "violations": sum(not v for v in values) if values else None,
            "violation_rate": sum(not v for v in values) / len(values) if len(values) == entry["n"] else None}
    return {"gold_ref_by_k": table, "offset_nll_by_k": ce.offset_by_k(records, min_count=1),
            "edge_accuracy": ce.edge_accuracy(records)}


def wf_metrics(streams: list[list[int]], vocab: dict, train: list[dict]) -> dict:
    report = eval_samples.evaluate(streams, vocab, [r["tokens"] for r in train])
    report["ci95"] = list(ce.wilson(report["well_formed"], report["total"])) if streams else None
    if not streams:
        for key in ("well_formed_rate", "unique_rate", "novelty_rate", "avg_stream_len"):
            report[key] = None
    return report


def continuation(rows: list[dict], tokens: dict, vocab: dict, n: int, seed: int) -> dict:
    probes = []
    for row in rows:
        if "wf_probe" not in row:
            continue
        p = row["wf_probe"]
        gold = tokens[row["sample_id"]]
        prefix_len = max(1, (len(gold) - 1) // 2)
        expected_seed = int.from_bytes(hashlib.sha256(f"{seed}:{row['sample_id']}:wf-v1".encode()).digest()[:8], "big")
        if p["prefix_len"] != prefix_len or p["budget"] != 2 * len(gold) or p["seed"] != expected_seed:
            raise ValueError("invalid wf_probe metadata")
        for mode in ("raw", "constrained"):
            if p[mode][:prefix_len] != gold[:prefix_len] or len(p[mode]) > p["budget"]:
                raise ValueError("invalid wf_probe prefix/budget")
        probes.append(p)
    result = {"prefix_source_n": n, "probes": len(probes)}
    names = {v: k for k, v in vocab.items()}
    for mode, label in (("raw", "reference-constrained"), ("constrained", "grammar-constrained")):
        by_k: dict[int, Counter] = {}
        stopped = Counter()
        for p in probes:
            state = grammar_mask.GrammarState(vocab)
            for i, token in enumerate(p[mode][1:], 1):
                legal = token in state.allowed_ids()
                name = names.get(token, "unknown")
                if i >= p["prefix_len"] and name.startswith("REF_"):
                    counts = by_k.setdefault(int(name[4:]), Counter())
                    counts["refs"] += 1
                    counts["violations"] += not legal
                if not legal:
                    if i < p["prefix_len"]:
                        raise ValueError("illegal gold prefix")
                    category = eval_samples.classify_stream(p[mode][:i + 1] + [vocab["EOS"]], vocab)
                    stopped[category] += 1
                    break
                state.push(token)
        result[label] = {**wf_metrics([p[mode] for p in probes], vocab, []),
                         "stopped_at_first_violation": sum(stopped.values()), "stopping_reasons": dict(stopped),
                         "suffix_ref_by_predicted_k": {k: {**v, "violation_rate": v["violations"] / v["refs"]}
                                                       for k, v in sorted(by_k.items())}}
    return result


def aggregate(rows: list[dict], raw: dict, exclusions: dict, tokens: dict, vocab: dict, base: dict) -> dict:
    count = sum(r["n_tokens"] for r in rows)
    lengths = sorted(len(tokens[r["sample_id"]]) for r in rows)
    baseline = [base.get(r["sample_id"]) for r in rows]
    baseline_nll = sum(r["nll"] for r in baseline if r is not None) if all(r is not None for r in baseline) and rows else None
    reasons = Counter(r["reason"] for r in exclusions.values())
    return {"samples": len(rows), "tokens": count, "raw": len(raw), "retained": len(rows),
            "excluded": len(exclusions), "excluded_by_reason": dict(reasons),
            "ref_window_excluded_rate": reasons["over_window"] / len(raw) if raw else None,
            "nats_per_token": sum(r["nll"] for r in rows) / count if count else None,
            "baseline_nats_per_token": baseline_nll / count if baseline_nll is not None else None,
            "excess_nats_per_token": (sum(r["nll"] for r in rows) - baseline_nll) / count if baseline_nll is not None else None,
            "length_including_bos": {"mean": mean(lengths), "p50": lengths[(len(lengths)-1)//2],
                                     "p95": lengths[int(.95*(len(lengths)-1))], "max": max(lengths)} if lengths else None,
            **ref_metrics(rows, tokens, vocab)}


def seed_stats(values: dict) -> dict:
    valid = [v for v in values.values() if v is not None]
    return {"per_seed": values, "mean": mean(valid) if valid else None,
            "sd": pstdev(valid) if valid else None, "sd_kind": "population", "available_seeds": len(valid)}


def metric_seed_stats(rows: dict) -> dict:
    """Keep denominators and missing seeds alongside each aggregate statistic."""
    keys = ("samples", "tokens", "raw", "retained", "excluded", "nats_per_token",
            "baseline_nats_per_token", "excess_nats_per_token", "ref_window_excluded_rate")
    result = {key: seed_stats({seed: row[key] for seed, row in rows.items()}) for key in keys}
    result["edge_accuracy"] = seed_stats({seed: row["edge_accuracy"]["overall"] for seed, row in rows.items()})
    ks = sorted({k for row in rows.values() for k in row["gold_ref_by_k"]})
    result["gold_ref_by_k"] = {
        k: {"nll": seed_stats({seed: row["gold_ref_by_k"].get(k, {}).get("mean") for seed, row in rows.items()}),
            "gold_refs_per_seed": {seed: row["gold_ref_by_k"].get(k, {}).get("n", 0) for seed, row in rows.items()},
            "legality_violation_rate": seed_stats({seed: row["gold_ref_by_k"].get(k, {}).get("teacher_forced_legality", {}).get("violation_rate") for seed, row in rows.items()})}
        for k in ks}
    return result


def evaluate(index: dict, root: Path = Path('.')) -> dict:
    required = {"schema_version", "source_dataset", "source_bundle", "buckets", "runs", "selection_protocol"}
    if not required <= index.keys() or index["schema_version"] != 1:
        raise ValueError("eval index requires schema_version 1 and all required fields")
    ns = [b["n"] for b in index["buckets"]]
    if len(ns) != len(set(ns)) or set(ns) != set(ALL_NS):
        raise ValueError("index requires all 11 unique n buckets")
    runs = index["runs"]
    if not runs or len({r["name"] for r in runs}) != len(runs) or len({(r["pos"], r["seed"]) for r in runs}) != len(runs):
        raise ValueError("empty or duplicate runs/pos/seed")
    if len({r["source_run"] for r in runs}) != len(runs):
        raise ValueError("duplicate source_run")
    paths = [(root / p).resolve() for r in runs for p in r["scores_by_n"].values()]
    if len(set(paths)) != len(paths) or any(set(r["scores_by_n"]) != {str(n) for n in ALL_NS} for r in runs):
        raise ValueError("missing or duplicate scores_by_n")
    check_bundles(index, root)
    source = root / index["source_bundle"]
    vocab = read_json(source / "vocab.json")
    train, train_raw, _ = checked_split(source, root / index["source_dataset"], "train")
    if any(r["n"] not in ID_NS for r in train_raw.values()):
        raise ValueError("source train contains OOD n")
    buckets, baselines, all_raw, all_exclusions, all_tokens, base_scores = {}, {}, {}, {}, {}, {}
    for bucket in sorted(index["buckets"], key=lambda b: b["n"]):
        n = bucket["n"]
        rows, raw, exclusions = checked_split(root / bucket["bundle"], root / bucket["dataset"], "test")
        tokens = {r["sample_id"]: r["tokens"] for r in rows}
        if all_raw.keys() & raw.keys():
            raise ValueError("duplicate test IDs across buckets")
        fit = sorted([r for r in train if n not in ID_NS or train_raw[r["sample_id"]]["n"] == n], key=lambda r: r["sample_id"])
        baseline = InfoBaseline.fit(fit, vocab, 128, alpha=.5) if fit else None
        scored = score_rows(baseline, rows) if baseline else []
        base_scores.update(unique_rows(scored))
        baselines[str(n)] = {"baseline_fit_scope": f"n{n}" if n in ID_NS else "pooled_id",
                             "fit_samples": len(fit), "fit_ids": [r["sample_id"] for r in fit],
                             "fit_rows_sha256": hashlib.sha256(json.dumps(fit, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                             "source_manifest_sha256": digest(root / index["source_dataset"] / 'manifest.json'),
                             "source_train_sha256": digest(source / 'train.jsonl'), "alpha": .5, "max_k": 128}
        buckets[n] = (raw, exclusions, tokens)
        all_raw.update(raw)
        all_exclusions.update(exclusions)
        all_tokens.update(tokens)
    reports, cache = {}, {}
    for run in runs:
        source_run = root / run["source_run"]
        payloads = {name: read_json(source_run / name) for name in ("samples.json", "samples_constrained.json")}
        sample_config = payloads["samples.json"].get("config", {}) if isinstance(payloads["samples.json"], dict) else {}
        if sample_config and (sample_config["seed"] != run["seed"] or sample_config["pos"] != run["pos"]):
            raise ValueError("source run config differs from eval index")
        if sample_config and (not sample_config.get("ref_legal_mask") or sample_config.get("constrained")):
            raise ValueError("source samples must use reference-constrained decoding")
        by_n, scores_all = {}, []
        for n in ALL_NS:
            raw, exclusions, tokens = buckets[n]
            scores = sorted(read_jsonl(root / run["scores_by_n"][str(n)]), key=lambda r: r["sample_id"])
            ce.validate_scores(scores, tokens, expect_ids=set(tokens))
            if any(row["seed"] != raw[row["sample_id"]]["seed"] for row in scores):
                raise ValueError("score seed differs from manifest")
            if "wf_probes" in sample_config:
                expected_probes = {row["sample_id"] for row in scores[:sample_config["wf_probes"]]}
                if {row["sample_id"] for row in scores if "wf_probe" in row} != expected_probes:
                    raise ValueError("missing or unexpected wf_probe rows")
            if n == 48:
                original = sorted(read_jsonl(source_run / "test_scores.jsonl"), key=lambda r: r["sample_id"])
                ce.validate_scores(original, tokens, expect_ids=set(tokens))
                if original != scores:
                    raise ValueError("n48 rescore differs from source run scores")
            cache[(run["name"], n)] = scores
            entry = aggregate(scores, raw, exclusions, tokens, vocab, base_scores)
            entry["baseline_fit_scope"] = baselines[str(n)]["baseline_fit_scope"]
            entry["continuation"] = continuation(scores, tokens, vocab, n, run["seed"])
            entry["by_family"] = {}
            for family in sorted({r["family"] for r in raw.values()}):
                ids = {sid for sid, r in raw.items() if r["family"] == family}
                entry["by_family"][family] = aggregate([r for r in scores if r["sample_id"] in ids],
                    {sid: raw[sid] for sid in ids}, {sid: r for sid, r in exclusions.items() if sid in ids}, tokens, vocab, base_scores)
                entry["by_family"][family]["baseline_fit_scope"] = baselines[str(n)]["baseline_fit_scope"]
            by_n[str(n)] = entry
            scores_all.extend(scores)
        families = {}
        for family in sorted({r["family"] for r in all_raw.values()}):
            families[family] = {}
            for scope, scope_ns in (("id", ID_NS), ("ood192", (192,)), ("ood256", (256,))):
                ids = {sid for sid, r in all_raw.items() if r["family"] == family and r["n"] in scope_ns}
                families[family][scope] = aggregate([r for r in scores_all if r["sample_id"] in ids],
                    {sid: all_raw[sid] for sid in ids}, {sid: r for sid, r in all_exclusions.items() if sid in ids}, all_tokens, vocab, base_scores)
                families[family][scope]["baseline_fit_scope"] = "per_n" if scope == "id" else "pooled_id"
        id_scores = [r for n in ID_NS for r in cache[(run["name"], n)]]
        id_tokens = sum(r["n_tokens"] for r in id_scores)
        macro_ns = [n for n in ID_NS if not (n in (8, 12) and by_n[str(n)]["raw"] < 100)]
        macro_values = [by_n[str(n)]["nats_per_token"] for n in macro_ns]
        wf = {}
        for filename, label in (("samples.json", "reference-constrained"), ("samples_constrained.json", "grammar-constrained")):
            payload = payloads[filename]
            streams = payload["samples"] if isinstance(payload, dict) else payload
            count_key = "num_samples" if filename == "samples.json" else "constrained_samples"
            if count_key in sample_config and len(streams) != sample_config[count_key]:
                raise ValueError("missing generated samples")
            wf[label] = wf_metrics(streams, vocab, train)
        reports[run["name"]] = {"pos": run["pos"], "seed": run["seed"], "by_n": by_n, "by_family": families,
            "id": {"token_micro": sum(r["nll"] for r in id_scores) / id_tokens if id_tokens else None,
                   "bucket_macro": mean(macro_values) if macro_values and all(v is not None for v in macro_values) else None,
                   "macro_ns": macro_ns, "macro_excluded_ns": [n for n in ID_NS if n not in macro_ns], "nominal_buckets": 9},
            "unconditional_wf": wf}
        length_path = source_run / "length_stats.json"
        if length_path.exists():
            reports[run["name"]]["train_length_stats"] = read_json(length_path)
    summary = {}
    for pos in sorted({r["pos"] for r in runs}):
        selected = [r for r in reports.values() if r["pos"] == pos]
        summary[pos] = {"by_n": {str(n): metric_seed_stats({str(r["seed"]): r["by_n"][str(n)] for r in selected}) for n in ALL_NS},
                        "by_family": {family: {scope: metric_seed_stats({str(r["seed"]): r["by_family"][family][scope] for r in selected})
                                               for scope in ("id", "ood192", "ood256")}
                                      for family in selected[0]["by_family"]},
                        "unconditional_wf": {label: seed_stats({str(r["seed"]): r["unconditional_wf"][label]["well_formed_rate"] for r in selected})
                                             for label in ("reference-constrained", "grammar-constrained")},
                        "id": {key: seed_stats({str(r["seed"]): r["id"][key] for r in selected}) for key in ("token_micro", "bucket_macro")}}
        for n in ALL_NS:
            entry = summary[pos]["by_n"][str(n)]
            entry["by_family"] = {family: metric_seed_stats({str(r["seed"]): r["by_n"][str(n)]["by_family"][family] for r in selected})
                                  for family in selected[0]["by_n"][str(n)]["by_family"]}
            entry["continuation_wf"] = {label: seed_stats({str(r["seed"]): r["by_n"][str(n)]["continuation"][label]["well_formed_rate"] for r in selected})
                                        for label in ("reference-constrained", "grammar-constrained")}
    paired = []
    for a, b in combinations(runs, 2):
        if a["seed"] != b["seed"]:
            continue
        def family_pair(family, ns):
            return paired_ci(
                [r for n in ns for r in cache[(a["name"], n)] if all_raw[r["sample_id"]]["family"] == family],
                [r for n in ns for r in cache[(b["name"], n)] if all_raw[r["sample_id"]]["family"] == family])
        paired.append({"left": a["name"], "right": b["name"], "seed": a["seed"],
                       "by_n": {str(n): paired_ci(cache[(a["name"], n)], cache[(b["name"], n)]) for n in ALL_NS},
                       "by_n_family": {str(n): {family: family_pair(family, (n,))
                                               for family in sorted({r["family"] for r in buckets[n][0].values()})} for n in ALL_NS},
                       "by_family": {family: {scope: family_pair(family, ns) for scope, ns in
                                               (("id", ID_NS), ("ood192", (192,)), ("ood256", (256,)))}
                                     for family in sorted({r["family"] for r in all_raw.values()})},
                       "id_token_micro": paired_ci([r for n in ID_NS for r in cache[(a["name"], n)]],
                                                   [r for n in ID_NS for r in cache[(b["name"], n)]])})
    return {"schema_version": 1, "selection_protocol": index["selection_protocol"], "runs": reports,
            "summary": summary, "paired": paired, "baselines": baselines,
            "n48_specialist_reference": {"layered": .77, "spaghetti": .61,
                "note": "旧窓・旧test・旧versionの参考値。厳密な差ではない。"},
            "notes": ["NLLはnats/token。IDとOOD192/256は別集計。", "REF>128除外後の評価。長い全CFGへの汎化を意味しない。",
                      "n8/12のraw件数が100未満ならmacroから除外 (§9)。", "WFはsource runの無条件生成。continuationはprefix_source_n。"]}


def markdown(report: dict) -> str:
    def fmt(value):
        if value is None:
            return "欠測"
        return f"{value:.6g}" if isinstance(value, float) else str(value)

    lines = ['# 長系列事前学習の評価', '', *report['notes'], '',
             '## ID 集計', '', '| run | token micro | bucket macro | macro 対象 n |',
             '|---|---:|---:|---|']
    for name, run in report['runs'].items():
        row = run['id']
        lines.append(f"| {name} | {fmt(row['token_micro'])} | {fmt(row['bucket_macro'])} | {row['macro_ns']} |")
    for name, run in report['runs'].items():
        lines.extend(['', f'## {name}', '', '### n・family 別 NLL', '',
                      '| n / 範囲 | family | raw / retained | tokens | REF窓除外率 | NLL/token | excess |',
                      '|---|---|---:|---:|---:|---:|---:|'])
        def metric_line(scope, family, row):
            lines.append(f"| {scope} | {family} | {row['raw']} / {row['retained']} | {row['tokens']} | "
                         f"{fmt(row['ref_window_excluded_rate'])} | {fmt(row['nats_per_token'])} | {fmt(row['excess_nats_per_token'])} |")
        for n, row in run['by_n'].items():
            metric_line(n, '全体', row)
            for family, sub in row['by_family'].items():
                metric_line(n, family, sub)
        for family, scopes in run['by_family'].items():
            for scope, row in scopes.items():
                metric_line(scope, family, row)
        lines.extend(['', '### Gold REF (teacher-forced)', '',
                      '| n | k | 件数 | full NLL | offset NLL | edge accuracy | 合法性違反率 | n<30 |',
                      '|---:|---:|---:|---:|---:|---:|---:|---|'])
        for n, row in run['by_n'].items():
            for k, rec in row['gold_ref_by_k'].items():
                offset = row['offset_nll_by_k'].get(k, {}).get('mean')
                acc = row['edge_accuracy']['by_k'].get(k, {}).get('acc')
                lines.append(f"| {n} | {k} | {rec['n']} | {fmt(rec['mean'])} | {fmt(offset)} | {fmt(acc)} | "
                             f"{fmt(rec['teacher_forced_legality']['violation_rate'])} | {rec['low_count']} |")
        lines.extend(['', '### WF', '', '無条件生成と prefix continuation は別の分母。', '',
                      '| 対象 | 制約 | 成功 / 全件 | WF | Wilson CI95 | 違反内訳 |',
                      '|---|---|---:|---:|---|---|'])
        def wf_line(scope, mode, row):
            lines.append(f"| {scope} | {mode} | {row['well_formed']} / {row['total']} | {fmt(row['well_formed_rate'])} | "
                         f"{row['ci95']} | {json.dumps(row['violations'], sort_keys=True)} |")
        for mode, row in run['unconditional_wf'].items():
            wf_line('source run 無条件', mode, row)
        for n, row in run['by_n'].items():
            for mode in ('reference-constrained', 'grammar-constrained'):
                wf_line(f'prefix_source_n={n}', mode, row['continuation'][mode])
        lines.extend(['', '### 生成 suffix の REF 診断', '',
                      '| prefix_source_n | 制約 | predicted k | REF 分母 | 違反 | 途中停止件数 |',
                      '|---:|---|---:|---:|---:|---:|'])
        for n, row in run['by_n'].items():
            for mode in ('reference-constrained', 'grammar-constrained'):
                continuation_row = row['continuation'][mode]
                for k, rec in continuation_row['suffix_ref_by_predicted_k'].items():
                    lines.append(f"| {n} | {mode} | {k} | {rec['refs']} | {rec['violations']} | "
                                 f"{continuation_row['stopped_at_first_violation']} |")
    lines.extend(['', '## seed 集計', '', 'SD は母標準偏差。各 seed の値と分母は JSON にも保存。', '',
                  '| pos | n | NLL mean | SD | excess mean | SD |', '|---|---:|---:|---:|---:|---:|'])
    for pos, summary in report['summary'].items():
        for n, row in summary['by_n'].items():
            a, b = row['nats_per_token'], row['excess_nats_per_token']
            lines.append(f"| {pos} | {n} | {fmt(a['mean'])} | {fmt(a['sd'])} | {fmt(b['mean'])} | {fmt(b['sd'])} |")
    lines.extend(['', '## Paired ΔNLL (left − right)', '', '同じ ID を対にした token-weighted bootstrap、Random(0)、1000回。', '',
                  '| left | right | n / 範囲 | samples | Δ nats/token | CI95 |', '|---|---|---|---:|---:|---|'])
    for pair in report['paired']:
        for n, row in {**pair['by_n'], 'ID micro': pair['id_token_micro']}.items():
            lines.append(f"| {pair['left']} | {pair['right']} | {n} | {row['samples']} | {fmt(row['delta_nats_per_token'])} | {row['ci95']} |")
    lines.extend(['', '## 情報量基準', '', 'InfoBaseline: max_k=128、alpha=0.5。fit ID・hash は JSON に保存。', '',
                  '| n | baseline_fit_scope | fit samples |', '|---:|---|---:|'])
    for n, row in report['baselines'].items():
        lines.append(f"| {n} | {row['baseline_fit_scope']} | {row['fit_samples']} |")
    lines.extend(['', 'n48 専門家参考値: layered ≈0.77、spaghetti ≈0.61。旧窓・旧test・旧versionのため厳密な差ではない。', ''])
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate(read_json(args.index))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    args.out.with_suffix('.md').write_text(markdown(report), encoding='utf-8')


if __name__ == '__main__':
    main()
