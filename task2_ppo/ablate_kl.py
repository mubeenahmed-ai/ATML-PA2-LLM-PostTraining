"""Task 2, Step 3 - reward-overoptimisation / KL-pressure study.

python -m task2_ppo.ablate_kl --config configs/ppo.yaml [--skip-eval]

Matched short forks (fork_updates) from the identical supplied midpoint policy + critic for
every kl_beta in cfg['kl_values'], with clip_epsilon fixed at cfg['clip_epsilon'], the same
prompts, sampling seeds and update budget, followed by the common held-out evaluation.
The (eps=0.20, kl=0.10) fork is shared with the clipping study (identical configuration).
"""
from __future__ import annotations

import argparse

from common.data import load_yaml
from common.models import clear_gpu
from task2_ppo.continue_train import fork_name, run_ppo
from task2_ppo.evaluate import evaluate_model
from task2_ppo.utils import task_dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--work-dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--save-every", type=int, default=2)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--gen-batch-size", type=int, default=16)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    dirs = task_dirs(cfg, args.work_dir, args.results_dir)
    print("KL beta conditions:", cfg["kl_values"], "| fork update budget:", cfg["fork_updates"])
    for beta in [float(b) for b in cfg["kl_values"]]:
        name = fork_name(cfg["clip_epsilon"], beta)
        run_ppo(args.config, updates=int(cfg["fork_updates"]), clip_epsilon=float(cfg["clip_epsilon"]), kl_beta=beta,
                run_name=name, work_dir=args.work_dir, results_dir=args.results_dir, save_every=args.save_every)
        clear_gpu()
        if not args.skip_eval:
            evaluate_model(args.config, str(dirs["work"] / name), name, work_dir=args.work_dir,
                           results_dir=args.results_dir, eval_n=args.eval_n, gen_batch_size=args.gen_batch_size)
            clear_gpu()
    print("KL-pressure study complete.")


if __name__ == "__main__":
    main()
