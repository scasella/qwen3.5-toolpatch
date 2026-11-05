#!/usr/bin/env python
"""Supplementary counts for the README (run after analyze.py):
  python analysis_extra.py --work <work> --data <bfcl_eval/data> --out <analysis dir>
Writes <out>/extra.md.
"""
import argparse, collections, glob, json, os, re
ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
MODELS = [("base", "Qwen_Qwen3.5-0.8B"), ("g23v3", "Qwen_Qwen3.5-0.8B-toolpatch-g23v3"), ("g23v4", "Qwen_Qwen3.5-0.8B-toolpatch-g23v4")]
CONFIGS = [("stock", "-FC"), ("campaign", "-campaign-prompt-FC")]
CATS = ["simple_python","multiple","parallel","parallel_multiple","irrelevance","live_simple","live_multiple","live_parallel","live_parallel_multiple","live_irrelevance","live_relevance"]
def load(reg):
    out = {}
    for p in glob.glob(f"{a.work}/result/{reg}/*/BFCL_v4_*_result.json"):
        cat = re.search(r"BFCL_v4_(.*)_result", p).group(1)
        for line in open(p):
            if line.strip():
                r = json.loads(line); out[r["id"]] = (cat, r)
    return out
def failed_ids(reg):
    bad = set()
    for p in glob.glob(f"{a.work}/score/{reg}/*/BFCL_v4_*_score.json"):
        with open(p) as f:
            next(f)
            for line in f:
                if line.strip(): bad.add(json.loads(line)["id"])
    return bad
def cls(r):
    raw = (r.get("raw_response") or ""); s = raw.strip()
    if r.get("num_tool_calls", 0) > 0: return "call"
    if s == "NO_TOOL": return "NO_TOOL"
    if s == "NEED_CLARIFICATION": return "NEED_CLARIFICATION"
    if r.get("finish_reason") == "length": return "truncated"
    return "prose"
R, BAD = {}, {}
for m, reg in MODELS:
    for c, suf in CONFIGS:
        R[m, c] = load(reg + suf); BAD[m, c] = failed_ids(reg + suf)
L = []
# 1. what the wins/losses were made of, base -> g23v4, per config and category
L.append("### What the paired flips base → g23v4 consist of (g23v4's output class on flipped items)\n")
L.append("| config | category | direction | n | call | NO_TOOL | NEED_CLARIFICATION | prose | truncated |")
L.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
for c, _ in CONFIGS:
    for cat in CATS:
        for direction, cond in (("wrong→right", lambda b, g: b and not g), ("right→wrong", lambda b, g: (not b) and g)):
            cnt = collections.Counter()
            for i, (icat, rb) in R["base", c].items():
                if icat != cat: continue
                rg = R["g23v4", c][i][1]
                if cond(i in BAD["base", c], i in BAD["g23v4", c]): cnt[cls(rg)] += 1
            n = sum(cnt.values())
            if n: L.append(f"| {c} | {cat} | {direction} | {n} | {cnt['call']} | {cnt['NO_TOOL']} | {cnt['NEED_CLARIFICATION']} | {cnt['prose']} | {cnt['truncated']} |")
# 2. leak signature: prose beginning with NO / NEITHER / NEED etc.
L.append("\n### Non-call outputs whose first word looks like a fragment of an abstention literal\n")
L.append("Counted over outputs with no tool call. `NO,`/`NO ` = first word NO (not NO_TOOL); `NEITHER`; `NEED` = first word starts with NEED but is not the exact literal; `NONE`/`NOT` for comparison.\n")
L.append("| model / config | non-call outputs | starts `NO` | starts `NEITHER` | starts `NEED…` (not literal) | starts `NOT`/`NONE` | starts `I ` |")
L.append("|---|---:|---:|---:|---:|---:|---:|")
pat = {"NO": re.compile(r"^NO[ ,.!:;]"), "NEITHER": re.compile(r"^NEITHER\b"), "NEED": re.compile(r"^NEED(?!_CLARIFICATION$)"), "NOT": re.compile(r"^(NOT|NONE)\b"), "I": re.compile(r"^I ")}
for m, _ in MODELS:
    for c, _ in CONFIGS:
        cnt = collections.Counter(); n = 0
        for i, (cat, r) in R[m, c].items():
            k = cls(r)
            if k == "call": continue
            s = (r.get("raw_response") or "").strip()
            if k in ("NO_TOOL", "NEED_CLARIFICATION"): continue
            n += 1
            for name, rx in pat.items():
                if rx.match(s): cnt[name] += 1
        L.append(f"| {m} / {c} | {n} | {cnt['NO']} | {cnt['NEITHER']} | {cnt['NEED']} | {cnt['NOT']} | {cnt['I']} |")
# 3. mixed outputs: literal + call in one generation; and literal + prose
L.append("\n### Mixed outputs: an abstention literal somewhere in the text together with a tool call, or followed by prose\n")
L.append("| model / config | literal + tool call (scored as call) | literal + prose (scored as no call) | repeated literal |")
L.append("|---|---:|---:|---:|")
lit = re.compile(r"\b(NO_TOOL|NEED_CLARIFICATION)\b")
for m, _ in MODELS:
    for c, _ in CONFIGS:
        lc = lp = rep = 0
        for i, (cat, r) in R[m, c].items():
            s = (r.get("raw_response") or "").strip()
            if not lit.search(s) or s in ("NO_TOOL", "NEED_CLARIFICATION"): continue
            if r.get("num_tool_calls", 0) > 0: lc += 1
            else:
                lp += 1
                if len(lit.findall(s)) > 1: rep += 1
        L.append(f"| {m} / {c} | {lc} | {lp} | {rep} |")
# 4. live_relevance and live_simple losses under campaign: list
L.append("\n### Campaign configuration: items lost on live_relevance and live_simple (base right, g23v4 wrong)\n")
L.append("| category | id | g23v4 output | base output (first 120 chars) | user text (first 160 chars) |")
L.append("|---|---|---|---|---|")
def user_text(cat, i):
    fn = f"{a.data}/BFCL_v4_{cat}.json"
    for line in open(fn):
        r = json.loads(line)
        if r["id"] == i:
            for turn in r["question"]:
                for msg in turn:
                    if msg["role"] == "user": return msg["content"]
    return ""
def short(s, n): s = s.replace("\n", "⏎").replace("|", "\\|"); return s if len(s) <= n else s[:n] + "…"
for cat in ("live_relevance", "live_simple"):
    for i, (icat, rb) in sorted(R["base", "campaign"].items()):
        if icat != cat: continue
        if i not in BAD["base", "campaign"] and i in BAD["g23v4", "campaign"]:
            rg = R["g23v4", "campaign"][i][1]
            L.append(f"| {cat} | {i} | {short((rg.get('raw_response') or '').strip(), 120)} | {short((rb.get('raw_response') or '').strip(), 120)} | {short(user_text(cat, i), 160)} |")
# 5. per-category raw identity of calls, stock, base vs g23v4 (where both call)
L.append("\n### Stock configuration: byte-identical generations base vs g23v4, all outputs (not only calls), per category\n")
L.append("| category | n | identical | differ, same class | differ, class changed |")
L.append("|---|---:|---:|---:|---:|")
tot = collections.Counter()
for cat in CATS:
    ide = same = chg = 0
    for i, (icat, rb) in R["base", "stock"].items():
        if icat != cat: continue
        rg = R["g23v4", "stock"][i][1]
        if (rb.get("raw_response") or "") == (rg.get("raw_response") or ""): ide += 1
        elif cls(rb) == cls(rg): same += 1
        else: chg += 1
    tot.update(ide=ide, same=same, chg=chg)
    L.append(f"| {cat} | {ide+same+chg} | {ide} | {same} | {chg} |")
L.append(f"| **all** | {sum(tot.values())} | {tot['ide']} | {tot['same']} | {tot['chg']} |")
# 6. same for campaign
L.append("\n### Campaign configuration: byte-identical generations base vs g23v4, all outputs, per category\n")
L.append("| category | n | identical | differ, same class | differ, class changed |")
L.append("|---|---:|---:|---:|---:|")
tot = collections.Counter()
for cat in CATS:
    ide = same = chg = 0
    for i, (icat, rb) in R["base", "campaign"].items():
        if icat != cat: continue
        rg = R["g23v4", "campaign"][i][1]
        if (rb.get("raw_response") or "") == (rg.get("raw_response") or ""): ide += 1
        elif cls(rb) == cls(rg): same += 1
        else: chg += 1
    tot.update(ide=ide, same=same, chg=chg)
    L.append(f"| {cat} | {ide+same+chg} | {ide} | {same} | {chg} |")
L.append(f"| **all** | {sum(tot.values())} | {tot['ide']} | {tot['same']} | {tot['chg']} |")
# 7. g23v3 vs g23v4 identical
for c, _ in CONFIGS:
    ide = sum(1 for i, (cat, r3) in R["g23v3", c].items() if (r3.get("raw_response") or "") == (R["g23v4", c][i][1].get("raw_response") or ""))
    L.append(f"\ng23v3 vs g23v4, {c}: {ide}/{len(R['g23v3', c])} generations byte-identical.")
# 7b. examples of leaked-literal prose under the stock protocol (g23v4), with the base output on the same item
L.append("\n### Stock configuration: examples of g23v4 outputs that open with a shouted literal fragment (base output on the same item for comparison)\n")
L.append("| id | g23v4 (first 150 chars) | base (first 80 chars) |"); L.append("|---|---|---|")
shown = 0
for name, rx in (("NO", pat["NO"]), ("NEITHER", pat["NEITHER"]), ("NEED", pat["NEED"])):
    k = 0
    for i, (cat, r) in sorted(R["g23v4", "stock"].items()):
        s_ = (r.get("raw_response") or "").strip()
        if cls(r) == "call" or not rx.match(s_): continue
        L.append(f"| {i} | {short(s_, 150)} | {short((R['base', 'stock'][i][1].get('raw_response') or '').strip(), 80)} |")
        k += 1
        if k >= 4: break
cnt = collections.Counter((R["g23v4", "stock"][i][1].get("raw_response") or "").strip()[:6] for i in R["g23v4", "stock"] if cls(R["g23v4", "stock"][i][1]) != "call" and pat["NO"].match((R["g23v4", "stock"][i][1].get("raw_response") or "").strip()))
L.append("\nMost common openings among g23v4/stock outputs starting with `NO`: " + ", ".join(f"`{k!r}` ×{v}" for k, v in cnt.most_common(5)))
rep = sum(1 for i in R["g23v4", "stock"] if cls(R["g23v4", "stock"][i][1]) != "call" and (R["g23v4", "stock"][i][1].get("raw_response") or "").strip().startswith("NO") and (R["g23v4", "stock"][i][1].get("raw_response") or "").count("NO,") >= 3)
L.append(f"\ng23v4/stock outputs starting with `NO` that repeat the sentence three or more times: {rep}.")
# 7c. McNemar exact test on the paired flips base -> g23v4 (two-sided binomial on the discordant pairs)
from scipy.stats import binomtest
L.append("\n### McNemar exact test on the paired flips base → g23v4 (two-sided binomial on discordant pairs)\n")
L.append("| config | category | wrong→right | right→wrong | p |"); L.append("|---|---|---:|---:|---:|")
AST = [c for c in CATS if "irrelevance" not in c]
for c, _ in CONFIGS:
    pooled = collections.Counter()
    for cat in CATS + ["all AST categories pooled", "all 11 pooled"]:
        if cat in CATS:
            b = sum(1 for i, (icat, _) in R["base", c].items() if icat == cat and i in BAD["base", c] and i not in BAD["g23v4", c])
            d = sum(1 for i, (icat, _) in R["base", c].items() if icat == cat and i not in BAD["base", c] and i in BAD["g23v4", c])
            if cat in AST: pooled.update(ab=b, ad=d)
            pooled.update(b=b, d=d)
        elif cat.startswith("all AST"): b, d = pooled["ab"], pooled["ad"]
        else: b, d = pooled["b"], pooled["d"]
        pv = binomtest(b, b + d, 0.5).pvalue if b + d else float("nan")
        L.append(f"| {c} | {cat} | {b} | {d} | {pv:.3g} |")
# 8. Latency / tokens
L.append("\n### Output length (mean new tokens) and items hitting the 512 cap\n")
L.append("| model / config | mean output tokens | median | at cap |"); L.append("|---|---:|---:|---:|")
import statistics
for m, _ in MODELS:
    for c, _ in CONFIGS:
        toks = [r.get("output_token_count", 0) for _, r in R[m, c].values()]
        L.append(f"| {m} / {c} | {statistics.mean(toks):.0f} | {statistics.median(toks):.0f} | {sum(1 for _, r in R[m, c].values() if r.get('finish_reason') == 'length')} |")
os.makedirs(a.out, exist_ok=True)
open(f"{a.out}/extra.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
