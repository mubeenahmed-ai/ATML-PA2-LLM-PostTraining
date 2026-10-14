"""Task 1, Step 2 - regularisation-strength study.

python -m task1_dpo.ablate_beta --config configs/dpo.yaml

For every beta in cfg['betas'] a SHORT fork is trained from the ORIGINAL policy
initialisation (fresh LoRA, same seed => identical LoRA init) on the first
cfg['short_ablation_examples'] training pairs, then evaluated with the common protocol.
Only beta differs between forks. Resumable: finished forks / eval stages are skipped.
"""
from __future__ import annotations

import argparse

from common.data import load_yaml
from common.models import clear_gpu
from task1_dpo.evaluate import evaluate_model
from task1_dpo.train import run_training
from task1_dpo.utils import task_dirs


def beta_run_name(beta: float) -> str:
    return f"beta_{float(beta):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--work-dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--betas", help="comma list to run a subset, e.g. 0.03,0.10")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--gen-n", type=int, default=200)
    ap.add_argument("--gen-batch-size", type=int, default=16)
    ap.add_argument("--gen-max-prompt-tokens", type=int, default=512)
    ap.add_argument("--wl-samples", type=int, default=4)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    dirs = task_dirs(cfg, args.work_dir, args.results_dir)
    betas = [float(b) for b in (args.betas.split(",") if args.betas else cfg["betas"])]
    n_short = int(cfg["short_ablation_examples"])
    print("Required beta values:", betas, "| short-run examples per condition:", n_short)

    for beta in betas:
        run = beta_run_name(beta)
        run_training(args.config, run, beta=beta, max_examples=n_short,
                     work_dir=args.work_dir, results_dir=args.results_dir, save_every=args.save_every)
        clear_gpu()
        if not args.skip_eval:
            evaluate_model(
                args.config, str(dirs["work"] / run), run, beta=beta,
                stages=["pref", "length_pref", "gen", "wordlimit", "reward"],
                work_dir=args.work_dir, results_dir=args.results_dir, gen_n=args.gen_n,
                gen_batch_size=args.gen_batch_size, gen_max_prompt_tokens=args.gen_max_prompt_tokens,
                wl_samples=args.wl_samples,
            )
            clear_gpu()
    print("Beta sweep complete. Run `python -m task1_dpo.summarize` to build the tables/figures.")


if __name__ == "__main__":
    main()
