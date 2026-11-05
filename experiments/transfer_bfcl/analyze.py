#!/usr/bin/env python
"""Analysis of the BFCL transfer run: accuracy tables, paired flips, call identity,
parallel-category false alarms and irrelevance quotes.

Usage: python analyze.py --work ~/Downloads/bfcl-toolpatch/work --data <bfcl_eval/data> --out <dir>
Writes <out>/analysis.md (tables, markdown), <out>/analysis.json, <out>/flips.csv,
<out>/parallel_abstentions.csv.  Score files are optional: when a score file is
missing the per-item correctness for that (model, category) is None and the row
shows "n/a"; result files are read as they are (partial runs are fine).
"""
import argparse
import collections
import csv
import json
import pathlib

CATS = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance",
        "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance", "live_relevance"]
PARALLEL_CATS = ["parallel", "parallel_multiple", "live_parallel", "live_parallel_multiple"]
IRR_CATS = ["irrelevance", "live_irrelevance"]
MODELS = [("base", "Qwen_Qwen3.5-0.8B"), ("g23v3", "Qwen_Qwen3.5-0.8B-toolpatch-g23v3"), ("g23v4", "Qwen_Qwen3.5-0.8B-toolpatch-g23v4")]
CONFIGS = [("stock", "-FC"), ("campaign", "-campaign-prompt-FC")]
LITERALS = ("NO_TOOL", "NEED_CLARIFICATION")


def sub(cat):
    return "live" if cat.startswith("live_") else "non_live"


def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def classify(rec):
    """What did the model emit?  call / NO_TOOL / NEED_CLARIFICATION / prose / truncated / error."""
    raw = rec.get("raw_response")
    if raw is None:
        return "error" if str(rec.get("result", "")).startswith("Error during inference") else "unknown"
    s = raw.strip()
    if rec.get("num_tool_calls", 0) > 0:
        return "call"
    for lit in LITERALS:
        if s == lit:
            return lit
    if rec.get("finish_reason") == "length":
        return "truncated"
    if "<tool_call>" in s:
        return "malformed_call"
    return "prose"


def user_text(entry):
    msgs = entry["question"][0]
    users = [m["content"] for m in msgs if m["role"] == "user"]
    return users[-1] if users else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quotes", type=int, default=4)
    a = ap.parse_args()
    work, data, out = pathlib.Path(a.work), pathlib.Path(a.data), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    dataset = {c: {e["id"]: e for e in load_jsonl(data / f"BFCL_v4_{c}.json")} for c in CATS}
    order = {c: list(dataset[c]) for c in CATS}

    # results[(model, config)][cat][id] -> record ; correct[(model,config)][cat][id] -> bool|None
    results, correct, acc = {}, {}, {}
    for mkey, mdir in MODELS:
        for ckey, csuf in CONFIGS:
            reg = mdir + csuf
            results[mkey, ckey], correct[mkey, ckey], acc[mkey, ckey] = {}, {}, {}
            for c in CATS:
                rp = work / "result" / reg / sub(c) / f"BFCL_v4_{c}_result.json"
                recs = {r["id"]: r for r in load_jsonl(rp)} if rp.exists() else {}
                results[mkey, ckey][c] = recs
                sp = work / "score" / reg / sub(c) / f"BFCL_v4_{c}_score.json"
                if sp.exists() and len(recs) == len(order[c]):
                    lines = load_jsonl(sp)
                    head, fails = lines[0], {e["id"] for e in lines[1:]}
                    correct[mkey, ckey][c] = {i: (i not in fails) for i in order[c]}
                    acc[mkey, ckey][c] = (head["correct_count"], head["total_count"], head["accuracy"])
                else:
                    correct[mkey, ckey][c] = {i: None for i in order[c]}
                    acc[mkey, ckey][c] = None

    md, js = [], {"accuracy": {}, "flips": {}, "call_identity": {}, "outcomes": {}, "parallel_abstentions": [], "errors": {}}

    # ---- accuracy tables -------------------------------------------------------
    for ckey, _ in CONFIGS:
        md.append(f"\n### Accuracy, {ckey} configuration\n")
        md.append("| category | n | base | g23v3 | g23v4 | Δ base→g23v4 (items) |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for c in CATS:
            cells, counts = [], []
            for mkey, _ in MODELS:
                v = acc[mkey, ckey][c]
                if v is None:
                    cells.append("n/a"); counts.append(None)
                else:
                    cells.append(f"{100*v[2]:.1f}% ({v[0]})"); counts.append(v[0])
                js["accuracy"][f"{mkey}/{ckey}/{c}"] = None if v is None else {"correct": v[0], "total": v[1], "accuracy": v[2]}
            d = "n/a" if None in (counts[0], counts[2]) else f"{counts[2]-counts[0]:+d}"
            md.append(f"| {c} | {len(order[c])} | " + " | ".join(cells) + f" | {d} |")
        # overall (unweighted over the 11 categories, as the leaderboard does within sections)
        tot = []
        for mkey, _ in MODELS:
            vs = [acc[mkey, ckey][c] for c in CATS]
            tot.append("n/a" if any(v is None for v in vs) else f"{sum(v[0] for v in vs)}/{sum(v[1] for v in vs)} ({100*sum(v[0] for v in vs)/sum(v[1] for v in vs):.1f}%)")
        md.append(f"| **all 11 (item-weighted)** | {sum(len(order[c]) for c in CATS)} | " + " | ".join(tot) + " | |")

    # ---- outcome mix ------------------------------------------------------------
    md.append("\n### What the models emitted (all 11 categories pooled)\n")
    md.append("| model / config | items | tool call | NO_TOOL | NEED_CLARIFICATION | prose | truncated (512 tok) | malformed call | inference error |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mkey, _ in MODELS:
        for ckey, _ in CONFIGS:
            cnt = collections.Counter()
            for c in CATS:
                for r in results[mkey, ckey][c].values():
                    cnt[classify(r)] += 1
            n = sum(cnt.values())
            js["outcomes"][f"{mkey}/{ckey}"] = dict(cnt)
            md.append(f"| {mkey} / {ckey} | {n} | {cnt['call']} | {cnt['NO_TOOL']} | {cnt['NEED_CLARIFICATION']} | {cnt['prose']} | {cnt['truncated']} | {cnt['malformed_call']} | {cnt['error']} |")
    # literal abstentions per category for the edited models (where the circuit shows up)
    md.append("\n### Literal abstentions per category (count of NO_TOOL / NEED_CLARIFICATION outputs)\n")
    md.append("| category | " + " | ".join(f"{m}/{c}" for m, _ in MODELS for c, _ in CONFIGS) + " |")
    md.append("|---|" + "---:|" * 6)
    for c in CATS:
        row = []
        for mkey, _ in MODELS:
            for ckey, _ in CONFIGS:
                cnt = collections.Counter(classify(r) for r in results[mkey, ckey][c].values())
                row.append(f"{cnt['NO_TOOL']} / {cnt['NEED_CLARIFICATION']}")
        md.append(f"| {c} | " + " | ".join(row) + " |")

    # ---- paired flips -------------------------------------------------------------
    flips_rows = []
    for ckey, _ in CONFIGS:
        for (m0, m1) in (("base", "g23v4"), ("g23v3", "g23v4"), ("base", "g23v3")):
            md.append(f"\n### Paired flips {m0} → {m1}, {ckey} configuration\n")
            md.append("| category | wrong→right | right→wrong | unchanged right | unchanged wrong |")
            md.append("|---|---:|---:|---:|---:|")
            for c in CATS:
                wr = rw = ur = uw = 0
                for i in order[c]:
                    a0, a1 = correct[m0, ckey][c][i], correct[m1, ckey][c][i]
                    if a0 is None or a1 is None:
                        continue
                    if not a0 and a1:
                        wr += 1; flips_rows.append((ckey, m0, m1, c, i, "wrong->right", user_text(dataset[c][i])))
                    elif a0 and not a1:
                        rw += 1; flips_rows.append((ckey, m0, m1, c, i, "right->wrong", user_text(dataset[c][i])))
                    elif a0 and a1:
                        ur += 1
                    else:
                        uw += 1
                js["flips"][f"{ckey}/{m0}->{m1}/{c}"] = {"wrong_right": wr, "right_wrong": rw, "both_right": ur, "both_wrong": uw}
                md.append(f"| {c} | {wr} | {rw} | {ur} | {uw} |")
    with open(out / "flips.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["config", "from", "to", "category", "id", "flip", "user_text"]); w.writerows(flips_rows)

    # ---- call identity -------------------------------------------------------------
    md.append("\n### Are the calls token-identical? (items where BOTH models produced ≥1 tool call; raw text compared byte for byte)\n")
    md.append("| config | pair | both call | raw identical | raw differ | differ but same decoded calls |")
    md.append("|---|---|---:|---:|---:|---:|")
    diff_examples = []
    for ckey, _ in CONFIGS:
        for (m0, m1) in (("base", "g23v3"), ("base", "g23v4"), ("g23v3", "g23v4")):
            both = same = diff = same_decoded = 0
            for c in CATS:
                for i in order[c]:
                    r0, r1 = results[m0, ckey][c].get(i), results[m1, ckey][c].get(i)
                    if not r0 or not r1 or not r0.get("num_tool_calls") or not r1.get("num_tool_calls"):
                        continue
                    both += 1
                    if r0["raw_response"] == r1["raw_response"]:
                        same += 1
                    else:
                        diff += 1
                        if r0["result"] == r1["result"]:
                            same_decoded += 1
                        if len(diff_examples) < 12:
                            diff_examples.append((ckey, m0, m1, i, r0["raw_response"][:200], r1["raw_response"][:200]))
            js["call_identity"][f"{ckey}/{m0}-{m1}"] = {"both_call": both, "identical": same, "differ": diff, "differ_same_decoded": same_decoded}
            md.append(f"| {ckey} | {m0} vs {m1} | {both} | {same} | {diff} | {same_decoded} |")
    # one-sided: items where one model called and the other did not
    md.append("\n| config | pair | only first calls | only second calls |")
    md.append("|---|---|---:|---:|")
    for ckey, _ in CONFIGS:
        for (m0, m1) in (("base", "g23v3"), ("base", "g23v4"), ("g23v3", "g23v4")):
            o0 = o1 = 0
            for c in CATS:
                for i in order[c]:
                    r0, r1 = results[m0, ckey][c].get(i), results[m1, ckey][c].get(i)
                    if not r0 or not r1:
                        continue
                    c0, c1 = bool(r0.get("num_tool_calls")), bool(r1.get("num_tool_calls"))
                    o0 += c0 and not c1; o1 += c1 and not c0
            md.append(f"| {ckey} | {m0} vs {m1} | {o0} | {o1} |")
    if diff_examples:
        md.append("\nExamples of calls that differ between models (first 200 chars of the raw text):\n")
        for ckey, m0, m1, i, t0, t1 in diff_examples:
            md.append(f"- `{i}` ({ckey}): **{m0}** `{t0!r}` → **{m1}** `{t1!r}`")

    # ---- parallel-category abstentions -------------------------------------------------
    md.append("\n### Parallel categories: every item where an edited model abstained (NEED_CLARIFICATION / NO_TOOL) instead of calling\n")
    rows = []
    for ckey, _ in CONFIGS:
        for mkey, _ in MODELS:
            if mkey == "base":
                continue
            for c in PARALLEL_CATS:
                for i in order[c]:
                    r = results[mkey, ckey][c].get(i)
                    if not r:
                        continue
                    k = classify(r)
                    if k in LITERALS:
                        rb = results["base", ckey][c].get(i)
                        rows.append({"config": ckey, "model": mkey, "category": c, "id": i, "output": k,
                                     "base_output": classify(rb) if rb else "n/a",
                                     "base_correct": correct["base", ckey][c][i], "user_text": user_text(dataset[c][i])})
    js["parallel_abstentions"] = rows
    with open(out / "parallel_abstentions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["config", "model", "category", "id", "output", "base_output", "base_correct", "user_text"])
        w.writeheader(); w.writerows(rows)
    # summary counts
    md.append("| config | model | category | items | abstained | of which base was right |")
    md.append("|---|---|---|---:|---:|---:|")
    for ckey, _ in CONFIGS:
        for mkey, _ in MODELS:
            if mkey == "base":
                continue
            for c in PARALLEL_CATS:
                rs = [r for r in rows if r["config"] == ckey and r["model"] == mkey and r["category"] == c]
                md.append(f"| {ckey} | {mkey} | {c} | {len(results[mkey, ckey][c])} | {len(rs)} | {sum(1 for r in rs if r['base_correct'])} |")
    md.append("\nAlso counted, for scale: base-model literal abstentions in the same categories: " + ", ".join(
        f"{ckey}: " + "/".join(str(sum(1 for r in results['base', ckey][c].values() if classify(r) in LITERALS)) for c in PARALLEL_CATS) for ckey, _ in CONFIGS)
        + " (parallel/parallel_multiple/live_parallel/live_parallel_multiple).")
    if rows:
        md.append("\nFull list (user text abridged to 220 chars):\n")
        md.append("| config | model | id | output | base output | user text |")
        md.append("|---|---|---|---|---|---|")
        for r in rows:
            u = r["user_text"].replace("\n", " ").replace("|", "\\|")
            u = u[:220] + ("…" if len(u) > 220 else "")
            md.append(f"| {r['config']} | {r['model']} | {r['id']} | {r['output']} | {r['base_output']} | {u} |")

    # ---- irrelevance quotes -----------------------------------------------------------
    md.append("\n### Irrelevance categories: representative outputs\n")
    for c in IRR_CATS:
        for ckey, _ in CONFIGS:
            # prefer items where the models disagree (base called, g23v4 did not) then a few shared ids
            ids = [i for i in order[c] if all(results[m, ckey][c].get(i) for m, _ in MODELS)]
            disagree = [i for i in ids if classify(results["base", ckey][c][i]) == "call" and classify(results["g23v4", ckey][c][i]) != "call"]
            reverse = [i for i in ids if classify(results["base", ckey][c][i]) != "call" and classify(results["g23v4", ckey][c][i]) == "call"]
            picks = (disagree[: a.quotes // 2] + reverse[:1] + ids[:1])[: a.quotes]
            if not picks:
                continue
            md.append(f"\n**{c}, {ckey} configuration** ({len(disagree)} items where base called and g23v4 did not; {len(reverse)} the reverse)\n")
            for i in picks:
                md.append(f"- `{i}` — user: *{user_text(dataset[c][i])[:200].replace(chr(10), ' ')}*")
                for mkey, _ in MODELS:
                    r = results[mkey, ckey][c][i]
                    t = r["raw_response"].replace("\n", "⏎")
                    md.append(f"    - **{mkey}** [{classify(r)}{', correct' if correct[mkey, ckey][c][i] else (', wrong' if correct[mkey, ckey][c][i] is False else '')}]: `{t[:260]}{'…' if len(t) > 260 else ''}`")

    # ---- inference errors ------------------------------------------------------------
    errs = {f"{m}/{k}": sum(1 for c in CATS for r in results[m, k][c].values() if classify(r) == "error") for m, _ in MODELS for k, _ in CONFIGS}
    js["errors"] = errs
    md.append("\n### Inference errors (template exceptions etc., scored as model failures): " + ", ".join(f"{k}: {v}" for k, v in errs.items()) + "\n")

    (out / "analysis.md").write_text("\n".join(md) + "\n")
    (out / "analysis.json").write_text(json.dumps(js, indent=1))
    done = {f"{m}/{k}": sum(len(results[m, k][c]) for c in CATS) for m, _ in MODELS for k, _ in CONFIGS}
    print("items generated:", done)
    print("wrote", out / "analysis.md")


if __name__ == "__main__":
    main()
