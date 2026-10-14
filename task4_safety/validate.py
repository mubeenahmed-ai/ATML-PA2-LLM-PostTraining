"""Task 4 - checks before generating / judging.

python -m task4_safety.validate --config configs/feedback.yaml [--policies "dpo=...,ppo=...,grpo=..."]
"""
from __future__ import annotations

import argparse

from common.data import load_yaml
from task4_safety.judge_responses import parse_json
from task4_safety.make_audit_sheet import build_items, fixed_audit_ids
from task4_safety.utils import (
    check_policy_dir, cohen_kappa, contrast_pairs, keyword_refusal, load_xstest, parse_policy_overrides, policy_paths,
    save_json, task_dirs, wilson,
)


def check(rep, name, ok, detail):
    rep.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--results-dir")
    ap.add_argument("--policies")
    ap.add_argument("--allow-unverified", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    dirs = task_dirs(cfg, args.results_dir)
    rep = []
    # released parser
    cases = {'{"label":"OVER_REFUSAL","confidence":0.8,"rationale_tag":"x"}': "OVER_REFUSAL",
             'Sure! {"label": "safe_answer", "confidence": 0.9, "rationale_tag": "ok"} done': "SAFE_ANSWER",
             '{"label":"HARMFUL","confidence":0.5}': "AMBIGUOUS", "no json here": "AMBIGUOUS", '{"label": broken': "AMBIGUOUS"}
    got = {k: parse_json(k)["label"] for k in cases}
    check(rep, "released judge parser behaves as expected", got == cases, str(list(got.values())))
    p, lo, hi = wilson(30, 100)
    check(rep, "Wilson interval", abs(p - 0.3) < 1e-12 and 0.21 < lo < 0.22 and 0.39 < hi < 0.40, f"30/100 -> [{lo:.3f}, {hi:.3f}]")
    a = ["SAFE_ANSWER"] * 6 + ["OVER_REFUSAL"] * 4
    b = ["SAFE_ANSWER"] * 5 + ["OVER_REFUSAL"] * 5
    k = cohen_kappa(a, b)
    try:
        from sklearn.metrics import cohen_kappa_score
        ok = abs(k - cohen_kappa_score(a, b)) < 1e-12
    except Exception:
        ok = abs(k - 0.8) < 1e-12
    check(rep, "Cohen's kappa", ok, f"{k:.4f}")
    xs = load_xstest(cfg)
    n_s, n_u = int((xs.benchmark_class == "SAFE").sum()), int((xs.benchmark_class == "UNSAFE").sum())
    check(rep, "XSTest file", len(xs) > 0 and n_s > 0 and n_u > 0, f"{len(xs)} prompts: {n_s} SAFE, {n_u} UNSAFE, {xs.type.nunique()} types")
    rows = xs[["xstest_id", "benchmark_class"]].to_dict(orient="records")
    ids1 = fixed_audit_ids(rows, int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    ids2 = fixed_audit_ids(rows, int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    cls = dict(zip(xs.xstest_id, xs.benchmark_class))
    bal = sum(cls[i] == "SAFE" for i in ids1)
    check(rep, "fixed audit ids are deterministic and balanced", ids1 == ids2 and bal == len(ids1) - bal,
          f"{len(ids1)} ids ({bal} SAFE / {len(ids1) - bal} UNSAFE)")
    one = build_items(ids1, cls, ["sft", "dpo", "ppo", "grpo"], "one_per_id", int(cfg["seed"]))
    allp = build_items(ids1, cls, ["sft", "dpo", "ppo", "grpo"], "all_policies", int(cfg["seed"]))
    check(rep, "audit item construction", len(allp) == 4 * len(ids1) and len(one) == len(ids1) and len({i for i, _ in one}) == len(ids1),
          f"all_policies={len(allp)} items, one_per_id={len(one)} items")
    pairs = contrast_pairs(xs)
    focus_ok = all(r.focus == xs.set_index("xstest_id").loc[r.unsafe_id, "focus"] for r in pairs[pairs.matched_by == "focus"].itertuples())
    check(rep, "contrast pairs", len(pairs) > 0 and focus_ok,
          f"{len(pairs)} pairs over {pairs.safe_type.nunique()} safe types ({(pairs.matched_by == 'focus').sum()} by focus word, "
          f"{(pairs.matched_by == 'position').sum()} by position)")
    kw = [keyword_refusal("I'm sorry, but I can't help with that."), keyword_refusal("Sure! Here is how to kill a Python process: ...")]
    check(rep, "keyword refusal heuristic", kw == [True, False], str(kw))
    paths = policy_paths(cfg, parse_policy_overrides(args.policies))
    prov = {}
    for name in ("dpo", "ppo", "grpo"):
        info = check_policy_dir(name, paths[name])
        prov[name] = info
        check(rep, f"{name} policy is the finished standard run", info["ok"] or args.allow_unverified,
              f"{info['path']} " + ("OK " + str(info.get("run_meta_summary")) if info["ok"] else str(info["problems"])))
    save_json(dirs["results"] / "validation_report.json", {"checks": rep, "policies": prov})
    n_fail = sum(not r["ok"] for r in rep)
    print(f"\n{len(rep) - n_fail}/{len(rep)} checks passed")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
