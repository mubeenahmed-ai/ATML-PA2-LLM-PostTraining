"""Task 3, Step 3 - canonical GRPO (1/T_k) vs Dr. GRPO-style (1/max_completion_length) normalisation.

python -m task3_grpo.compare_normalization --config configs/grpo.yaml [--skip-eval]

Two matched short continuations (fork_updates) from the identical supplied midpoint; ONLY
loss_type differs (rewards, group-relative advantages incl. std scaling, beta, eps, prompts,
sampling seeds, K and caps are identical). Per-completion records (length, advantage, the
per-token loss weight implied by the normaliser and the measured per-completion policy-gradient
norm) are logged for the length-conditioned analysis; both forks get the common held-out evaluation.
Note: the full Dr. GRPO recipe also drops the std scaling of the advantage; here only the length
normaliser is changed so that its effect is isolated.
"""
from __future__ import annotations

import argparse

import numpy as np

from common.data import load_yaml
from common.metrics import safe_corr
from common.models import clear_gpu
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate_model
from task3_grpo.utils import read_jsonl_abs, save_json, task_dirs

FORKS = {"grpo": "fork_grpo", "dr_grpo": "fork_dr_grpo"}


def _boot_corr_ci(x, y, n_boot=2000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 4:
        return None
    vals = []
    rng = np.random.default_rng(0)
    for _ in range(n_boot):
        i = rng.integers(0, x.size, x.size)
        if np.std(x[i]) > 0 and np.std(y[i]) > 0:
            vals.append(np.corrcoef(x[i], y[i])[0, 1])
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])] if vals else None


def length_conditioned_stats(dirs, run):
    p = dirs["train_results"] / run / "completions_log.jsonl"
    if not p.exists():
        return None
    rows = [r for r in read_jsonl_abs(p) if r["in_loss"]]
    if not rows:
        return {"n": 0}
    L = np.array([r["length"] for r in rows], float)
    A = np.array([r["advantage"] for r in rows], float)
    W = np.array([r["per_token_weight"] for r in rows], float)
    G = np.array([r["seq_policy_grad_norm"] if r["seq_policy_grad_norm"] is not None else np.nan for r in rows], float)
    med = float(np.median(L))
    short, long_ = L <= med, L > med
    out = {"n_completions_in_loss": len(rows), "median_length": med,
           "note": ("corr_length_vs_per_token_weight is DEFINITIONAL (|A|/T_k for grpo, |A|/L for dr_grpo); the "
                    "empirical length-conditioned evidence is the measured per-completion gradient norm and the "
                    "generated lengths. Completions that hit the cap are excluded (masked), which narrows the length range.")}
    for name, sel in [("all", np.ones_like(L, bool)), ("positive_adv", A > 0), ("negative_adv", A < 0)]:
        s = sel & short
        lng = sel & long_
        gs = G[sel & ~np.isnan(G)]
        out[name] = {
            "n": int(sel.sum()),
            "corr_length_vs_per_token_weight": safe_corr(L[sel], W[sel]),
            "corr_length_vs_seq_grad_norm": safe_corr(L[sel & ~np.isnan(G)], G[sel & ~np.isnan(G)]) if gs.size > 2 else None,
            "corr_length_vs_seq_grad_norm_ci95": _boot_corr_ci(L[sel & ~np.isnan(G)], G[sel & ~np.isnan(G)]) if gs.size > 3 else None,
            "mean_per_token_weight_short": float(W[s].mean()) if s.any() else None,
            "mean_per_token_weight_long": float(W[lng].mean()) if lng.any() else None,
            "share_of_seq_grad_norm_on_long": (float(np.nansum(G[lng]) / np.nansum(G[sel])) if np.nansum(G[sel]) > 0 else None),
            "mean_seq_grad_norm_short": float(np.nanmean(G[s])) if s.any() and not np.isnan(G[s]).all() else None,
            "mean_seq_grad_norm_long": float(np.nanmean(G[lng])) if lng.any() and not np.isnan(G[lng]).all() else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--work-dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--save-every", type=int, default=2)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--no-seq-grad-norms", action="store_true")
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--gen-batch-size", type=int, default=16)
    ap.add_argument("--eval-max-new-tokens", type=int)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    dirs = task_dirs(cfg, args.work_dir, args.results_dir)
    print("Fork updates:", cfg["fork_updates"], "| comparing loss_type grpo vs dr_grpo from the identical midpoint")
    for lt, run in FORKS.items():
        run_grpo(args.config, updates=int(cfg["fork_updates"]), loss_type=lt, run_name=run, work_dir=args.work_dir,
                 results_dir=args.results_dir, save_every=args.save_every, seq_grad_norms=not args.no_seq_grad_norms)
        clear_gpu()
        if not args.skip_eval:
            evaluate_model(args.config, str(dirs["work"] / run), run, work_dir=args.work_dir, results_dir=args.results_dir,
                           eval_n=args.eval_n, gen_batch_size=args.gen_batch_size, eval_max_new_tokens=args.eval_max_new_tokens)
            clear_gpu()
    res = {run: length_conditioned_stats(dirs, run) for run in FORKS.values()}
    save_json(dirs["results"] / "normalization_length_stats.json", res)
    for run, st in res.items():
        if st and st.get("all"):
            a = st["all"]
            print(f"[norm] {run}: corr(length, per-token weight)={a['corr_length_vs_per_token_weight']} "
                  f"corr(length, seq grad norm)={a['corr_length_vs_seq_grad_norm']} share on long={a['share_of_seq_grad_norm_on_long']}")
    print("Normalisation comparison complete.")


if __name__ == "__main__":
    main()
