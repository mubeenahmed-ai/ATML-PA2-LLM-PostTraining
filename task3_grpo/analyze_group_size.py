"""Task 3, Step 2 - equal-generation group-size study (no training, no GPU).

python -m task3_grpo.analyze_group_size --config configs/grpo.yaml

The supplied cache has 8 completions (with rewards) per prompt. For K in group_sizes the SAME
completions are re-partitioned, so the total number of generations is identical for every K:

  partition (primary): each prompt's 8 completions, ordered by generation_index, are split into
      8/K disjoint consecutive groups (K=8: one group of 8; K=4: [0-3],[4-7]; K=2: 4 pairs).
  all-subsets (robustness): every one of the C(8, K) possible groups of a prompt is evaluated,
      giving the exact probability that a random K-group from this prompt is informative.

Prompt difficulty is defined ONCE as the mean reward of a prompt's 8 cached completions and
split into tertiles (hard = lowest mean reward, easy = highest). A group is informative when its
population reward std > 1e-6 (the tolerance of group_relative_advantages).
"""
from __future__ import annotations

import argparse
import itertools
from collections import defaultdict

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl
from common.models import load_tokenizer
from task3_grpo.grpo import INFORMATIVE_EPS, group_relative_advantages
from task3_grpo.utils import save_json, task_dirs, write_jsonl


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Disjoint consecutive K-groups inside each prompt (first 8 completions by generation_index).
    Every K uses exactly the same N_prompts x 8 completions."""
    groups = []
    for pid, comps in by_prompt.items():
        comps = comps[:8]
        for g in range(8 // k):
            groups.append({"prompt": pid, "members": comps[g * k : (g + 1) * k]})
    return groups


def _group_metrics(groups, prompt_mean8):
    """Metrics over a list of groups; each group is a list of rewards (+ its prompt id)."""
    if not groups:
        return {"n_groups": 0}
    rewards = torch.tensor([r for g in groups for r in g["rewards"]], dtype=torch.float64)
    gid = torch.tensor([i for i, g in enumerate(groups) for _ in g["rewards"]])
    adv = group_relative_advantages(rewards, gid)
    stds = np.array([np.std(g["rewards"]) for g in groups])
    means = np.array([np.mean(g["rewards"]) for g in groups])
    centered = np.concatenate([np.asarray(g["rewards"]) - np.mean(g["rewards"]) for g in groups])
    base_err = means - np.array([prompt_mean8[g["prompt"]] for g in groups])
    informative = stds > INFORMATIVE_EPS
    return {
        "n_groups": len(groups),
        "n_completions": int(rewards.numel()),
        "informative_rate": float(informative.mean()),
        "uninformative_fraction": float(1 - informative.mean()),
        "mean_within_group_std": float(stds.mean()),
        "median_within_group_std": float(np.median(stds)),
        "var_centered_reward": float(np.var(centered)),          # relative signal before std scaling
        "var_group_relative_advantage": float(adv.var(unbiased=False)),  # after std scaling
        "mean_abs_advantage": float(adv.abs().mean()),
        "frac_nonzero_advantage": float((adv.abs() > 0).double().mean()),
        "baseline_mse_vs_8sample_mean": float(np.mean(base_err ** 2)),
    }


def prompt_truncation_flags(cfg, by_prompt):
    """Prompts whose chat-rendered length exceeds max_prompt_length were probably cut from the RIGHT
    when the cache was generated (the starter's batch_generate truncates that way), so their cached
    completions may continue the user text instead of answering it. Returns {prompt: bool or None}."""
    try:
        tok = load_tokenizer(cfg["base_model"])
    except Exception as exc:
        print("[group-size] tokenizer unavailable, skipping prompt-length check:", exc)
        return {p: None for p in by_prompt}
    lookup = {}
    for key in ("rl_prompt_eval", "rl_prompt_train"):
        for r in read_jsonl(cfg["paths"][key]):
            lookup.setdefault(str(r.get("source_index")), r)
    maxp = int(cfg["max_prompt_length"])
    out = {}
    for p in by_prompt:
        r = lookup.get(p)
        out[p] = None if r is None else len(tok.apply_chat_template(prompt_messages(r), tokenize=True,
                                                                      add_generation_prompt=True)) > maxp
    return out


def analyze(cfg, dirs, practical_rel_tol=0.05):
    by_prompt = load_k8_cache(cfg["group_cache"])
    long_prompt = prompt_truncation_flags(cfg, by_prompt)
    ks = [int(k) for k in cfg["group_sizes"]]
    pm8 = {p: float(np.mean([c["reward"] for c in comps[:8]])) for p, comps in by_prompt.items()}
    q1, q2 = np.quantile(list(pm8.values()), [1 / 3, 2 / 3])
    diff_bin = {p: ("hard" if m <= q1 else "medium" if m <= q2 else "easy") for p, m in pm8.items()}
    all_r = np.array([c["reward"] for comps in by_prompt.values() for c in comps[:8]])
    practical_tol = practical_rel_tol * float(all_r.std())
    trunc = {p: [bool(c.get("clipped_at_max", False)) for c in comps[:8]] for p, comps in by_prompt.items()}

    result = {
        "n_prompts": len(by_prompt), "completions_per_prompt": 8, "total_generations_per_K": len(by_prompt) * 8,
        "informative_tolerance": INFORMATIVE_EPS,
        "difficulty_rule": "mean reward of the prompt's 8 cached completions; tertiles (hard <= q1 < medium <= q2 < easy)",
        "difficulty_thresholds": {"q1": float(q1), "q2": float(q2)},
        "difficulty_counts": {b: sum(v == b for v in diff_bin.values()) for b in ("hard", "medium", "easy")},
        "practical_tolerance_note": f"sensitivity only: groups with std <= {practical_rel_tol} x pooled reward std "
                                    f"({practical_tol:.4f}) counted as 'near-uninformative'",
        "cache_truncated_fraction": float(np.mean([t for v in trunc.values() for t in v])),
        "n_prompts_longer_than_max_prompt_length": int(sum(bool(v) for v in long_prompt.values())),
        "n_prompts_not_found_in_prompt_pools": int(sum(v is None for v in long_prompt.values())),
        "prompts_longer_than_max_prompt_length": sorted(p for p, v in long_prompt.items() if v),
        "theory_note": "E[population variance of K iid rewards] = sigma^2 (K-1)/K, so mean within-group std and "
                       "var_centered_reward should grow with K roughly like sqrt((K-1)/K) and (K-1)/K. With population-std "
                       "scaling, var_group_relative_advantage equals the informative-group rate by construction.",
        "per_K": {},
    }
    group_rows = []
    for k in ks:
        part = [{"prompt": g["prompt"], "rewards": [c["reward"] for c in g["members"]],
                 "truncated": [bool(c.get("clipped_at_max", False)) for c in g["members"]]}
                for g in regroup_equal_generation_budget(by_prompt, k)]
        entry = {"partition": _group_metrics(part, pm8), "by_difficulty": {},
                 "theory_pop_var_factor_(K-1)/K": (k - 1) / k,
                 "partition_excluding_long_prompts": _group_metrics([g for g in part if not long_prompt.get(g["prompt"])], pm8)}
        for b in ("hard", "medium", "easy"):
            entry["by_difficulty"][b] = _group_metrics([g for g in part if diff_bin[g["prompt"]] == b], pm8)
        stds = np.array([np.std(g["rewards"]) for g in part])
        entry["partition"]["near_uninformative_fraction_practical_tol"] = float((stds <= practical_tol).mean())
        # if truncated completions are masked from the loss (as in training), how many groups still carry a signal?
        keep_inf = []
        for g in part:
            kept = [r for r, t in zip(g["rewards"], g["truncated"]) if not t]
            keep_inf.append(len(kept) >= 1 and np.std(g["rewards"]) > INFORMATIVE_EPS
                            and any(abs(r - np.mean(g["rewards"])) > 0 for r in kept))
        entry["partition"]["informative_with_nonmasked_signal_rate"] = float(np.mean(keep_inf))
        # all C(8,K) subsets per prompt
        sub_inf, sub_std, sub_bins = [], [], defaultdict(list)
        for p, comps in by_prompt.items():
            r = [c["reward"] for c in comps[:8]]
            vals = [np.std([r[i] for i in idx]) for idx in itertools.combinations(range(8), k)]
            pinf = float(np.mean(np.array(vals) > INFORMATIVE_EPS))
            sub_inf.append(pinf)
            sub_std.append(float(np.mean(vals)))
            sub_bins[diff_bin[p]].append(pinf)
        entry["all_subsets"] = {"n_subsets_per_prompt": len(list(itertools.combinations(range(8), k))),
                                "p_informative_mean_over_prompts": float(np.mean(sub_inf)),
                                "expected_within_group_std": float(np.mean(sub_std)),
                                "p_informative_by_difficulty": {b: float(np.mean(v)) for b, v in sub_bins.items()}}
        result["per_K"][str(k)] = entry
        for g in part:
            group_rows.append({"K": k, "prompt": g["prompt"], "difficulty": diff_bin[g["prompt"]],
                               "std": float(np.std(g["rewards"])), "informative": bool(np.std(g["rewards"]) > INFORMATIVE_EPS),
                               "rewards": g["rewards"], "truncated": g["truncated"]})
        pe = entry["partition"]
        print(f"[group-size] K={k}: groups={pe['n_groups']} informative={pe['informative_rate']:.3f} "
              f"mean std={pe['mean_within_group_std']:.3f} var(centered)={pe['var_centered_reward']:.3f} "
              f"| all-subsets P(inf)={entry['all_subsets']['p_informative_mean_over_prompts']:.3f}")
    # uninformative examples (identical rewards) for qualitative inspection
    examples = []
    for p, comps in by_prompt.items():
        for i, j in itertools.combinations(range(8), 2):
            if abs(comps[i]["reward"] - comps[j]["reward"]) <= INFORMATIVE_EPS:
                examples.append({"prompt": p, "difficulty": diff_bin[p], "gen_i": i, "gen_j": j, "reward": comps[i]["reward"],
                                 "identical_text": comps[i]["completion"] == comps[j]["completion"],
                                 "completion_i": comps[i]["completion"][:400], "completion_j": comps[j]["completion"][:400]})
    result["n_tied_pairs"] = len(examples)
    result["tied_pairs_identical_text_fraction"] = float(np.mean([e["identical_text"] for e in examples])) if examples else None
    save_json(dirs["results"] / "group_size_study.json", result)
    write_jsonl(dirs["results"] / "group_size_groups.jsonl", group_rows)
    write_jsonl(dirs["results"] / "group_size_tied_pairs.jsonl", examples[:200])
    save_json(dirs["results"] / "prompt_difficulty_bins.json", {"rule": result["difficulty_rule"], "mean_reward_8": pm8, "bin": diff_bin})
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--work-dir")
    ap.add_argument("--results-dir")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    dirs = task_dirs(cfg, args.work_dir, args.results_dir)
    print("Group sizes to analyze:", cfg["group_sizes"])
    analyze(cfg, dirs)
    print("Group-size study written to", dirs["results"])


if __name__ == "__main__":
    main()
