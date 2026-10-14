"""Task 5, Step 2 - reward sensitivity and robustness on the fixed Controlled Reward Diagnostic Set.

python -m task5_feedback.score_perturbations --config configs/feedback.yaml

For each of the 20 problems (5 variants each):
  RLVR  reward = released exact_reward (designated `#### <number>` final only)
  RLAIF reward = released PairwiseAIJudge.group_rewards over the 5 variants (the direct-RLAIF group reward,
                 (wins + 0.5 ties) / (K - 1), K = 5) - built from the same cached pairwise comparisons.
Controlled pairs (diagnostically better response first):
  reasoning   clean_correct  vs corrupt_reasoning_correct_final   (same final; reasoning degraded)
  outcome     clean_correct  vs good_reasoning_wrong_final         (same reasoning; final changed)
  filler      clean_correct  vs persuasive_filler_correct          (same content + unsupported persuasion)
  distractor  clean_correct  vs gold_distractor_wrong_final        (gold mentioned, then rejected; wrong final)
For each pair and mechanism: better-response rate, tie rate, wrong-preference rate. RLAIF uses the direct
pairwise judgment (the released compare); S_reason / S_outcome follow the manual's definitions. Position
(order) consistency of the judge is measured separately by presenting each pair in both fixed orders.
"""
from __future__ import annotations

import argparse
import itertools
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl
from common.models import clear_gpu
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge  # noqa: F401  (released judge)
from task5_feedback.utils import (
    backup_judge_cache, load_json, make_judge, presented_compare, released_raw_output, save_json, task_dirs, tie_kind,
    timed_compare, verifier_variants, wilson, write_jsonl,
)

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}
VARIANT_ORDER = ["clean_correct", "corrupt_reasoning_correct_final", "good_reasoning_wrong_final",
                 "persuasive_filler_correct", "gold_distractor_wrong_final"]
PAIR_TYPES = {
    "reasoning": ("clean_correct", "corrupt_reasoning_correct_final"),
    "outcome": ("clean_correct", "good_reasoning_wrong_final"),
    "filler": ("clean_correct", "persuasive_filler_correct"),
    "distractor": ("clean_correct", "gold_distractor_wrong_final"),
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def rates(prefs):
    n = len(prefs)
    b, t, w = sum(p == "better" for p in prefs), sum(p == "tie" for p in prefs), sum(p == "worse" for p in prefs)
    return {"n": n, "better_rate": b / n, "tie_rate": t / n, "wrong_preference_rate": w / n,
            "better_rate_ci95": list(wilson(b, n)[1:])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--results-dir")
    ap.add_argument("--skip-position-check", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    dirs = task_dirs(cfg, args.results_dir)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups), "| variants/problem:", len(EXPECTED_VARIANTS))

    # ---- RLVR: exact verifier (and the two looser parsers for the specification analysis)
    per_variant, mismatches = [], []
    for pid, v in groups.items():
        for name in VARIANT_ORDER:
            r = v[name]
            ex = exact_reward(r["response"], str(r["gold_final"]))
            if "expected_exact_reward" in r and float(r["expected_exact_reward"]) != ex:
                mismatches.append({"problem_id": pid, "variant": name, "expected": r["expected_exact_reward"], "got": ex})
            per_variant.append({"problem_id": pid, "variant": name, "rlvr_reward": ex,
                                **{f"verifier_{k}": val for k, val in verifier_variants(r["response"], r["gold_final"]).items()}})
    print(f"[rlvr] exact rewards match the set's expected_exact_reward on {len(per_variant) - len(mismatches)}/{len(per_variant)} rows")

    # ---- RLAIF: released judge, all C(5,2) pairs per problem (cached), group reward per variant
    stats = {}
    print("[judge] loading the fixed pairwise judge (4-bit; the first load downloads ~6 GB)...", flush=True)
    judge = make_judge(cfg, dirs["judge"] / "pairwise_cache.json")
    raw_p = dirs["judge"] / "tie_raw_cache.json"
    raw_cache = load_json(raw_p) if raw_p.exists() else {}
    pair_rows = []
    for k, (pid, v) in enumerate(groups.items()):
        q = v["clean_correct"]["question"]
        for a, b in itertools.combinations(VARIANT_ORDER, 2):
            res = timed_compare(judge, q, v[a]["response"], v[b]["response"], stats)
            kind = tie_kind(released_raw_output(judge, q, v[a]["response"], v[b]["response"], raw_cache)) if res == "TIE" else None
            pair_rows.append({"problem_id": pid, "a": a, "b": b, "judge": res, "tie_kind": kind})
        save_json(raw_p, raw_cache)
        gr = judge.group_rewards(q, [v[n]["response"] for n in VARIANT_ORDER])   # same cached comparisons
        for name, val in zip(VARIANT_ORDER, gr):
            next(x for x in per_variant if x["problem_id"] == pid and x["variant"] == name)["rlaif_group_reward"] = val
        backup_judge_cache(judge)
        print(f"  [judge] problem {k + 1}/{len(groups)}", flush=True)
    lookup = {(r["problem_id"], r["a"], r["b"]): r["judge"] for r in pair_rows}
    rl = {(x["problem_id"], x["variant"]): x for x in per_variant}

    # ---- controlled pairs
    pair_results, detail = {}, []
    for ptype, (good, bad) in PAIR_TYPES.items():
        prefs = {"rlvr": [], "rlaif_pairwise": [], "rlaif_group_reward": []}
        for pid in groups:
            rv_g, rv_b = rl[(pid, good)]["rlvr_reward"], rl[(pid, bad)]["rlvr_reward"]
            prefs["rlvr"].append("better" if rv_g > rv_b else ("worse" if rv_b > rv_g else "tie"))
            j = lookup[(pid, good, bad)]   # VARIANT_ORDER puts clean_correct first, so a = good
            prefs["rlaif_pairwise"].append({"A": "better", "B": "worse", "TIE": "tie"}[j])
            g_g, g_b = rl[(pid, good)]["rlaif_group_reward"], rl[(pid, bad)]["rlaif_group_reward"]
            prefs["rlaif_group_reward"].append("better" if g_g > g_b + 1e-12 else ("worse" if g_b > g_g + 1e-12 else "tie"))
            detail.append({"problem_id": pid, "pair": ptype, "rlvr": prefs["rlvr"][-1], "rlaif_pairwise": prefs["rlaif_pairwise"][-1],
                           "rlaif_group_reward": prefs["rlaif_group_reward"][-1]})
        pair_results[ptype] = {mech: rates(p) for mech, p in prefs.items()}
        kinds = [r["tie_kind"] for r in pair_rows if (r["a"], r["b"]) == (good, bad) and r["judge"] == "TIE"]
        pair_results[ptype]["rlaif_pairwise"]["explicit_ties"] = kinds.count("explicit_tie")
        pair_results[ptype]["rlaif_pairwise"]["ambiguous_outputs"] = kinds.count("ambiguous_output")
        print(f"[pairs] {ptype:10s} RLVR better/tie/wrong = {pair_results[ptype]['rlvr']['better_rate']:.2f}/"
              f"{pair_results[ptype]['rlvr']['tie_rate']:.2f}/{pair_results[ptype]['rlvr']['wrong_preference_rate']:.2f} | "
              f"RLAIF {pair_results[ptype]['rlaif_pairwise']['better_rate']:.2f}/{pair_results[ptype]['rlaif_pairwise']['tie_rate']:.2f}/"
              f"{pair_results[ptype]['rlaif_pairwise']['wrong_preference_rate']:.2f}")
    summary = {
        "S_reason": {m: pair_results["reasoning"][m]["better_rate"] for m in ("rlvr", "rlaif_pairwise", "rlaif_group_reward")},
        "S_outcome": {m: pair_results["outcome"][m]["better_rate"] for m in ("rlvr", "rlaif_pairwise", "rlaif_group_reward")},
        "definitions": {"S_reason": "Pr[R(clean_correct) > R(corrupt_reasoning_correct_final)] (same correct final)",
                        "S_outcome": "Pr[R(clean_correct) > R(good_reasoning_wrong_final)] (same reasoning, final changed)",
                        "filler_pair": "clean_correct counted as the better response (the filler adds only an unsupported "
                                       "persuasive claim); a tie is neutral; preferring the filler = style susceptibility"},
    }

    # ---- judge position (order) consistency on every controlled pair
    position = None
    if not args.skip_position_check:
        pcache_p = dirs["judge"] / "position_cache.json"
        pcache = load_json(pcache_p) if pcache_p.exists() else {}
        pos_rows = []
        for ptype, (good, bad) in PAIR_TYPES.items():
            for pid, v in groups.items():
                q = v["clean_correct"]["question"]
                r1 = presented_compare(judge, q, v[good]["response"], v[bad]["response"], pcache)   # good shown first
                r2 = presented_compare(judge, q, v[bad]["response"], v[good]["response"], pcache)   # good shown second
                good_1, good_2 = r1 == "A", r2 == "B"
                bad_1, bad_2 = r1 == "B", r2 == "A"
                if good_1 and good_2:
                    o = "consistent_better"
                elif bad_1 and bad_2:
                    o = "consistent_worse"
                elif r1 == "TIE" and r2 == "TIE":
                    o = "consistent_tie"
                elif r1 == "A" and r2 == "A":
                    o = "position_first"
                elif r1 == "B" and r2 == "B":
                    o = "position_second"
                else:
                    o = "inconsistent_other"
                pos_rows.append({"problem_id": pid, "pair": ptype, "good_first": r1, "good_second": r2, "outcome": o})
            save_json(pcache_p, pcache)
        outcomes = ["consistent_better", "consistent_worse", "consistent_tie", "position_first", "position_second", "inconsistent_other"]
        position = {"overall": {o: float(np.mean([r["outcome"] == o for r in pos_rows])) for o in outcomes}}
        position["overall"]["order_consistency_rate"] = float(np.mean([r["outcome"].startswith("consistent") for r in pos_rows]))
        position["by_pair"] = {pt: {o: float(np.mean([r["outcome"] == o for r in pos_rows if r["pair"] == pt])) for o in outcomes}
                               for pt in PAIR_TYPES}
        write_jsonl(dirs["diagnostics"] / "position_check.jsonl", pos_rows)
        print(f"[position] order-consistent judgments: {position['overall']['order_consistency_rate']:.2f} "
              f"(always-first {position['overall']['position_first']:.2f}, always-second {position['overall']['position_second']:.2f})")
    backup_judge_cache(judge)
    del judge
    clear_gpu()

    # ---- per-variant mean rewards (all five categories) and verifier-specification analysis
    var_tab = {}
    for name in VARIANT_ORDER:
        xs = [x for x in per_variant if x["variant"] == name]
        var_tab[name] = {"rlvr_mean_reward": float(np.mean([x["rlvr_reward"] for x in xs])),
                         "rlaif_mean_group_reward": float(np.mean([x["rlaif_group_reward"] for x in xs])),
                         "rlaif_group_reward_std": float(np.std([x["rlaif_group_reward"] for x in xs])),
                         **{f"{k}_parser_mean_reward": float(np.mean([x[f"verifier_{k}"] for x in xs])) for k in ("strict", "last_num", "any_num")}}
    res = {"n_problems": len(groups), "rlvr_expected_reward_mismatches": mismatches, "pairs": pair_results, "summary": summary,
           "per_variant": var_tab, "position_consistency": position,
           "judge_cost": {"new_calls": stats.get("calls", 0), "seconds": stats.get("seconds", 0.0),
                          "seconds_per_call": stats.get("seconds", 0.0) / max(stats.get("calls", 0), 1)}}
    old = dirs["diagnostics"] / "diagnostics.json"
    if old.exists() and res["judge_cost"]["new_calls"] == 0:
        res["judge_cost"] = load_json(old).get("judge_cost", res["judge_cost"])
    save_json(old, res)
    write_jsonl(dirs["diagnostics"] / "per_variant_rewards.jsonl", per_variant)
    write_jsonl(dirs["diagnostics"] / "pairwise_judgments.jsonl", pair_rows)
    write_jsonl(dirs["diagnostics"] / "controlled_pair_outcomes.jsonl", detail)
    print(f"S_reason: {summary['S_reason']}\nS_outcome: {summary['S_outcome']}")


if __name__ == "__main__":
    main()
