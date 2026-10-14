"""Task 1, Step 3 - length-confounding study.

python -m task1_dpo.analyze_length --config configs/dpo.yaml --stats-only   # dataset statistics only
python -m task1_dpo.analyze_length --config configs/dpo.yaml                # full study

1. Dataset statistics: how often the preferred response is longer in each fixed file
   (a property of the DATA, measured with the policy tokenizer).
2. Train one DPO model from the original initialisation on the supplied length-balanced
   subset (same hyper-parameters as the standard run, one epoch).
3. Evaluate standard vs length-balanced on the length-stratified held-out set (per
   stratum), plus generated length and word-limit compliance on the common prompts
   (properties of the learned POLICY).
"""
from __future__ import annotations

import argparse

import numpy as np

from common.data import load_yaml, preference_responses, read_jsonl
from common.metrics import safe_corr
from common.models import clear_gpu, load_tokenizer
from task1_dpo.evaluate import evaluate_model
from task1_dpo.train import run_training
from task1_dpo.utils import describe, file_fingerprint, rel_to_repo, response_token_count, save_json, task_dirs

DATA_KEYS = ["dpo_standard_train", "dpo_standard_eval", "dpo_length_train", "dpo_length_eval"]


def dataset_length_stats(cfg, tokenizer, matched_tol: int = 16) -> dict:
    out = {}
    for key in DATA_KEYS:
        path = cfg["paths"][key]
        rows = read_jsonl(path)
        lc, lr, sd, strata = [], [], [], {}
        for row in rows:
            yc, yr = preference_responses(row)
            lc.append(response_token_count(tokenizer, yc))
            lr.append(response_token_count(tokenizer, yr))
            if row.get("score_chosen") is not None and row.get("score_rejected") is not None:
                sd.append(float(row["score_chosen"]) - float(row["score_rejected"]))
            if row.get("length_stratum"):
                strata[row["length_stratum"]] = strata.get(row["length_stratum"], 0) + 1
        lc, lr = np.array(lc), np.array(lr)
        diff = lc - lr
        out[key] = {
            "file": rel_to_repo(path),
            "sha": file_fingerprint(path),
            "n_pairs": len(rows),
            "frac_preferred_longer": float((diff > 0).mean()),
            "frac_rejected_longer": float((diff < 0).mean()),
            "frac_equal_length": float((diff == 0).mean()),
            f"frac_abs_diff_le_{matched_tol}_tokens": float((np.abs(diff) <= matched_tol).mean()),
            "length_difference_chosen_minus_rejected": describe(diff),
            "chosen_tokens": describe(lc),
            "rejected_tokens": describe(lr),
            "corr_length_diff_vs_score_diff": safe_corr(diff[: len(sd)], sd) if len(sd) == len(rows) else None,
            "length_stratum_counts": strata or None,
            "tokenizer": cfg["base_model"],
        }
        print(f"[length-stats] {key}: n={len(rows)} preferred_longer={out[key]['frac_preferred_longer']:.3f} "
              f"mean diff={diff.mean():+.1f} tokens strata={strata or '-'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--work-dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--gen-n", type=int, default=200)
    ap.add_argument("--gen-batch-size", type=int, default=16)
    ap.add_argument("--gen-max-prompt-tokens", type=int, default=512)
    ap.add_argument("--wl-samples", type=int, default=4)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    dirs = task_dirs(cfg, args.work_dir, args.results_dir)

    stats_path = dirs["results"] / "dataset_length_stats.json"
    if not stats_path.exists():
        tok = load_tokenizer(cfg["base_model"])
        save_json(stats_path, dataset_length_stats(cfg, tok))
    else:
        print(f"[length-stats] exists: {stats_path}")
    if args.stats_only:
        return

    std_dir = dirs["work"] / "standard"
    if not (std_dir / "DONE.json").exists():
        raise SystemExit(f"Standard DPO model not finished ({std_dir}). Run Step 1 first.")

    run_training(args.config, "length_balanced", dataset_path=cfg["paths"]["dpo_length_train"],
                 work_dir=args.work_dir, results_dir=args.results_dir, save_every=args.save_every)
    clear_gpu()

    common = dict(work_dir=args.work_dir, results_dir=args.results_dir, gen_n=args.gen_n,
                  gen_batch_size=args.gen_batch_size, gen_max_prompt_tokens=args.gen_max_prompt_tokens,
                  wl_samples=args.wl_samples)
    for name in ["standard", "length_balanced"]:
        evaluate_model(args.config, str(dirs["work"] / name), name,
                       stages=["pref", "length_pref", "gen", "wordlimit", "reward"], **common)
        clear_gpu()
    print("Length study complete. Run `python -m task1_dpo.summarize` to build the tables/figures.")


if __name__ == "__main__":
    main()
