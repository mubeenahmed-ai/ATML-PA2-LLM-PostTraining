"""Task 5 - checks before generating / judging.

python -m task5_feedback.validate --config configs/feedback.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from task5_feedback.rlvr import exact_reward, extract_designated_final
from task5_feedback.score_perturbations import EXPECTED_VARIANTS, load_diagnostic_groups
from task5_feedback.utils import failure_type, pref_score, save_json, task_dirs, verifier_variants


def check(rep, name, ok, detail):
    rep.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--results-dir")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    dirs = task_dirs(cfg, args.results_dir)
    rep = []
    cases = {"so 4 + 5 = 9\n#### 9": "9", "#### 1,600": "1600", "first 12 ... #### 7 then #### 8": "8",
             "The answer is 9.": None, "#### $18": None, "#### -3.5": "-3.5"}
    got = {k: extract_designated_final(k) for k in cases}
    check(rep, "released final-answer parser (designated `####` only, last one wins)", got == cases,
          "; ".join(f"{k[:18]!r}->{v}" for k, v in got.items()))
    check(rep, "intermediate gold number is NOT rewarded", exact_reward("A tempting candidate is 9, rejected.\n#### 10", "9") == 0.0
          and verifier_variants("A tempting candidate is 9, rejected.\n#### 10", 9)["any_num"] == 1.0,
          "strict = 0 while the naive any-number parser = 1 (the specification matters)")
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    rows = read_jsonl(cfg["paths"]["task5_diagnostics"])
    mism = [r for r in rows if float(r.get("expected_exact_reward", exact_reward(r["response"], str(r["gold_final"])))) !=
            exact_reward(r["response"], str(r["gold_final"]))]
    check(rep, "diagnostic set: 5 variants per problem", all(set(v) == EXPECTED_VARIANTS for v in groups.values()),
          f"{len(groups)} problems, {len(rows)} rows")
    check(rep, "verifier reproduces expected_exact_reward on every diagnostic row", not mism, f"{len(rows) - len(mism)}/{len(rows)} match")
    for key in ("gsm_eval", "math_transfer_eval"):
        rr = read_jsonl(cfg["paths"][key])
        ok = all({"question", "messages", "gold_final"} <= set(r) for r in rr)
        gold_parse = np.mean([extract_designated_final(f"#### {r['gold_final']}") is not None for r in rr])
        check(rep, f"{key}: schema and gold answers parseable", ok and gold_parse == 1.0, f"{len(rr)} problems, gold parseable {gold_parse:.2f}")
    check(rep, "pairwise win-rate convention", [pref_score(x) for x in ("A", "TIE", "B")] == [1.0, 0.5, 0.0], "win 1, tie 0.5, loss 0")
    # the RLAIF group reward always averages 0.5 in a group (so 'mean reward' carries no signal)
    import itertools
    rng = np.random.default_rng(0)
    means = []
    for _ in range(50):
        k = 4
        wins, ties = np.zeros(k), np.zeros(k)
        for i, j in itertools.combinations(range(k), 2):
            r = rng.choice(["A", "B", "TIE"])
            if r == "A":
                wins[i] += 1
            elif r == "B":
                wins[j] += 1
            else:
                ties[i] += 1
                ties[j] += 1
        means.append(((wins + 0.5 * ties) / (k - 1)).mean())
    check(rep, "RLAIF group reward has mean exactly 0.5 in every group", np.allclose(means, 0.5), "so its training 'mean_reward'=0.5 is uninformative")
    ft = failure_type("Let me think... #### $5", 5, False)["type"], failure_type("long text", 5, True)["type"], failure_type("#### 4", 5, False)["type"]
    check(rep, "failure typing", ft == ("unparseable_final", "no_final_truncated", "wrong_final"), str(ft))
    for name in ("rlvr", "rlaif"):
        p = repo_path(cfg["policies"][name])
        check(rep, f"{name} adapter present", (p / "adapter_config.json").exists(), str(p))
        meta = repo_path(f"checkpoints/{name}_metadata.json")
        if meta.exists():
            m = json.loads(meta.read_text())
            print(f"       {name} metadata: steps={m.get('steps')} K={m.get('num_generations')} feedback={m.get('feedback')} "
                  f"mean_frac_reward_zero_std={m.get('mean_frac_reward_zero_std')}")
    save_json(dirs["results"] / "validation_report.json", rep)
    n_fail = sum(not r["ok"] for r in rep)
    print(f"\n{len(rep) - n_fail}/{len(rep)} checks passed")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
