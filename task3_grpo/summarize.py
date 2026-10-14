"""Task 3 - tables / figures / qualitative candidates from saved results (no GPU).

python -m task3_grpo.summarize --config configs/grpo.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml
from task3_grpo.utils import load_json, read_jsonl_abs, task_dirs

TRAJ = [("reward_mean", "reward (RM score, all K)"), ("kl_token_mean", "KL to reference (token mean)"),
        ("group_reward_std_mean", "within-group reward std"), ("uninformative_group_fraction", "uninformative-group fraction"),
        ("policy_loss", "policy loss"), ("grad_norm", "gradient norm (pre-clip)"), ("entropy_mean", "entropy (full dist.)"),
        ("response_length", "response length (tokens)"), ("truncated_fraction", "fraction truncated (masked)")]


def _g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _short(t, n=500):
    t = " ".join(str(t).split())
    return t if len(t) <= n else t[:n] + " [...]"


def save_table(df, path):
    df.to_csv(path, index=False)
    try:
        path.with_suffix(".md").write_text(df.to_markdown(index=False, floatfmt=".4f"))
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--work-dir")
    ap.add_argument("--results-dir")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    dirs = task_dirs(cfg, args.work_dir, args.results_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = {p.parent.name: load_json(p) for p in dirs["eval_results"].glob("*/metrics.json")}
    trains = {p.parent.name: load_json(p) for p in dirs["train_results"].glob("*/train_summary.json")}
    logs = {p.parent.name: pd.DataFrame(read_jsonl_abs(p)) for p in dirs["train_results"].glob("*/train_log.jsonl")}
    names = [n for n in ["midpoint", "standard", "fork_grpo", "fork_dr_grpo"] if n in metrics]
    print("models with held-out results:", names)

    # ---- standard continuation
    if "standard" in logs:
        log = logs["standard"]
        keep = ["update"] + [k for k, _ in TRAJ] + ["reward_mean_untruncated", "sample_entropy", "kl_k3_in_loss",
                                                    "policy_shift_kl_k3", "tokens_in_loss", "update_seconds"]
        save_table(log[[c for c in keep if c in log]], dirs["tables"] / "standard_trajectory.csv")
        t = trains.get("standard", {})
        rows = [{"metric": k, "first5_mean": float(log[k].head(5).mean()), "last5_mean": float(log[k].tail(5).mean())}
                for k, _ in TRAJ if log[k].notna().any()]
        rows += [{"metric": "wall_clock_min", "last5_mean": t.get("wall_clock_min")},
                 {"metric": "peak_vram_gb", "last5_mean": t.get("peak_vram_gb")},
                 {"metric": "uninformative_group_fraction_overall", "last5_mean": t.get("uninformative_group_fraction_mean")}]
        save_table(pd.DataFrame(rows), dirs["tables"] / "standard_summary.csv")
        fig, axes = plt.subplots(3, 3, figsize=(14, 10))
        for ax, (k, title) in zip(axes.flat, TRAJ):
            ax.plot(log["update"], log[k], "o-", ms=3, alpha=0.5)
            ax.plot(log["update"], log[k].rolling(5, min_periods=1).mean(), lw=2)
            ax.set_title(title)
            ax.set_xlabel("update")
        fig.suptitle(f"Standard GRPO continuation ({t.get('wall_clock_min', float('nan')):.1f} min, "
                     f"peak VRAM {t.get('peak_vram_gb', float('nan')):.2f} GB)")
        fig.tight_layout()
        fig.savefig(dirs["figures"] / "standard_trajectories.png", dpi=150)
        plt.close(fig)

    # ---- held-out summary
    rows = []
    for n in names:
        h = metrics[n].get("heldout", {})
        t = trains.get(n, {})
        rows.append({"model": n, "loss_type": _g(t, "run_meta", "loss_type"), "updates": _g(t, "run_meta", "updates"),
                     "train_generated_tokens": t.get("generated_tokens_total"), "train_tokens_in_loss": t.get("tokens_in_loss_total"),
                     "heldout_reward": _g(h, "reward_model_score", "mean"), "heldout_reward_std": _g(h, "reward_model_score", "std"),
                     "heldout_reward_untruncated": _g(h, "reward_untruncated", "mean"), "heldout_kl_token": h.get("kl_token_mean"),
                     "heldout_kl_seq": h.get("kl_sequence_mean"), "heldout_entropy": h.get("entropy_token_mean"),
                     "heldout_len_mean": _g(h, "response_tokens", "mean"), "heldout_len_std": _g(h, "response_tokens", "std"),
                     "heldout_len_median": _g(h, "response_tokens", "median"), "heldout_truncated": h.get("truncated_rate"),
                     "train_wall_min": t.get("wall_clock_min"), "train_peak_vram_gb": t.get("peak_vram_gb")})
    df = pd.DataFrame(rows)
    if not df.empty:
        save_table(df, dirs["tables"] / "heldout_summary.csv")

    # ---- group-size study
    gp = dirs["results"] / "group_size_study.json"
    if gp.exists():
        g = load_json(gp)
        rows = []
        for k, e in g["per_K"].items():
            base = {"K": int(k), "bin": "all", **{m: e["partition"][m] for m in
                    ["n_groups", "informative_rate", "mean_within_group_std", "var_centered_reward",
                     "var_group_relative_advantage", "mean_abs_advantage", "baseline_mse_vs_8sample_mean",
                     "near_uninformative_fraction_practical_tol", "informative_with_nonmasked_signal_rate"]},
                    "all_subsets_p_informative": e["all_subsets"]["p_informative_mean_over_prompts"],
                    "all_subsets_expected_std": e["all_subsets"]["expected_within_group_std"]}
            rows.append(base)
            for b, m in e["by_difficulty"].items():
                if m.get("n_groups"):
                    rows.append({"K": int(k), "bin": b, "n_groups": m["n_groups"], "informative_rate": m["informative_rate"],
                                 "mean_within_group_std": m["mean_within_group_std"], "var_centered_reward": m["var_centered_reward"],
                                 "var_group_relative_advantage": m["var_group_relative_advantage"],
                                 "mean_abs_advantage": m["mean_abs_advantage"],
                                 "baseline_mse_vs_8sample_mean": m["baseline_mse_vs_8sample_mean"],
                                 "all_subsets_p_informative": e["all_subsets"]["p_informative_by_difficulty"].get(b)})
        gs = pd.DataFrame(rows).sort_values(["bin", "K"])
        save_table(gs, dirs["tables"] / "group_size_study.csv")
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
        for ax, (col, title) in zip(axes, [("informative_rate", "informative-group rate (partition)"),
                                           ("all_subsets_p_informative", "P(informative), all C(8,K) groups"),
                                           ("mean_within_group_std", "mean within-group reward std")]):
            for b in ["all", "hard", "medium", "easy"]:
                sub = gs[gs["bin"] == b].sort_values("K")
                if not sub.empty:
                    ax.plot(sub["K"], sub[col], "o-", label=b)
            ax.set_xticks(sorted(gs["K"].unique()))
            ax.set_xlabel("group size K (equal total generations)")
            ax.set_title(title)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(dirs["figures"] / "group_size_study.png", dpi=150)
        plt.close(fig)

    # ---- normalisation comparison
    ns = dirs["results"] / "normalization_length_stats.json"
    comp = {r: pd.DataFrame([x for x in read_jsonl_abs(dirs["train_results"] / r / "completions_log.jsonl") if x["in_loss"]])
            for r in ["fork_grpo", "fork_dr_grpo"] if (dirs["train_results"] / r / "completions_log.jsonl").exists()}
    if ns.exists():
        st = load_json(ns)
        rows = []
        for r in ["fork_grpo", "fork_dr_grpo"]:
            s = st.get(r) or {}
            h = metrics.get(r, {}).get("heldout", {})
            for subset in ["all", "positive_adv", "negative_adv"]:
                if subset in s:
                    rows.append({"run": r, "subset": subset, **s[subset],
                                 "heldout_len_mean": _g(h, "response_tokens", "mean"), "heldout_reward": _g(h, "reward_model_score", "mean"),
                                 "heldout_kl_token": h.get("kl_token_mean")})
        if rows:
            save_table(pd.DataFrame(rows), dirs["tables"] / "normalization_comparison.csv")
    if comp:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for r, d in comp.items():
            for sign, mk in [(1, "o"), (-1, "x")]:
                sub = d[np.sign(d["advantage"]) == sign]
                axes[0].scatter(sub["length"], sub["per_token_weight"], marker=mk, alpha=0.6, label=f"{r} A{'>' if sign > 0 else '<'}0")
                if sub["seq_policy_grad_norm"].notna().any():
                    axes[1].scatter(sub["length"], sub["seq_policy_grad_norm"], marker=mk, alpha=0.6, label=f"{r} A{'>' if sign > 0 else '<'}0")
        axes[0].set_title("per-token loss weight |A_k| / normaliser")
        axes[1].set_title("per-completion policy-gradient norm (measured)")
        for ax in axes:
            ax.set_xlabel("completion length (tokens)")
            ax.set_yscale("log")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(dirs["figures"] / "normalization_length_conditioned.png", dpi=150)
        plt.close(fig)
    fl = [n for n in ["fork_grpo", "fork_dr_grpo"] if n in logs]
    if fl:
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
        for n in fl:
            for ax, (k, t) in zip(axes, [("reward_mean", "reward"), ("kl_token_mean", "KL"), ("response_length", "length"),
                                         ("grad_norm", "grad norm")]):
                ax.plot(logs[n]["update"], logs[n][k], "o-", ms=3, label=n)
                ax.set_title(t)
                ax.set_xlabel("update")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(dirs["figures"] / "normalization_fork_trajectories.png", dpi=150)
        plt.close(fig)

    # ---- paired comparisons on identical held-out prompts (ties = identical responses / equal reward)
    def _gens(n):
        p, r = dirs["eval_results"] / n / "generations.jsonl", dirs["eval_results"] / n / "rewards.json"
        if not (p.exists() and r.exists()):
            return None
        recs = read_jsonl_abs(p)
        return {x["row"]: (x["response"], s, x["response_tokens"]) for x, s in zip(recs, load_json(r)["scores"])}

    pair_rows = []
    for a_name, b_name in [("standard", "midpoint"), ("fork_grpo", "midpoint"), ("fork_dr_grpo", "midpoint"),
                           ("fork_dr_grpo", "fork_grpo")]:
        a, b = _gens(a_name), _gens(b_name)
        if not a or not b:
            continue
        ks = [k for k in a if k in b]
        dr = np.array([a[k][1] - b[k][1] for k in ks])
        dl = np.array([a[k][2] - b[k][2] for k in ks])
        same = np.array([a[k][0] == b[k][0] for k in ks])
        tie = same | (np.abs(dr) <= 1e-6)
        rng = np.random.default_rng(0)
        boot = [dr[rng.integers(0, len(dr), len(dr))].mean() for _ in range(2000)]
        pair_rows.append({"model": a_name, "vs": b_name, "n": len(ks), "identical_response_rate": float(same.mean()),
                          "win_rate": float(((dr > 0) & ~tie).mean()), "tie_rate": float(tie.mean()),
                          "loss_rate": float(((dr < 0) & ~tie).mean()), "mean_reward_diff": float(dr.mean()),
                          "mean_reward_diff_ci95_lo": float(np.percentile(boot, 2.5)),
                          "mean_reward_diff_ci95_hi": float(np.percentile(boot, 97.5)),
                          "mean_len_diff": float(dl.mean()), "mean_len_diff_nonidentical": float(dl[~same].mean()) if (~same).any() else None})
    if pair_rows:
        save_table(pd.DataFrame(pair_rows), dirs["tables"] / "paired_comparisons.csv")

    # ---- qualitative candidates
    def gens(n):
        p, r = dirs["eval_results"] / n / "generations.jsonl", dirs["eval_results"] / n / "rewards.json"
        if not (p.exists() and r.exists()):
            return None
        recs = read_jsonl_abs(p)
        for x, s in zip(recs, load_json(r)["scores"]):
            x["reward"] = s
        return {x["row"]: x for x in recs}

    lines = ["# Task 3 qualitative candidates (auto-selected - read and judge them yourself)\n"]
    a, b = gens("fork_grpo"), gens("fork_dr_grpo")
    if a and b:
        ks = [k for k in a if k in b]
        diff = sorted(ks, key=lambda k: -abs(a[k]["response_tokens"] - b[k]["response_tokens"]))[:5]
        lines.append("## A. Held-out prompts where the two normalisations produce the most different lengths\n")
        for k in diff:
            lines += [f"### eval row {k}", f"- GRPO: {a[k]['response_tokens']} tok, reward {a[k]['reward']:+.3f}, truncated {a[k]['truncated']}",
                      f"- Dr.GRPO: {b[k]['response_tokens']} tok, reward {b[k]['reward']:+.3f}, truncated {b[k]['truncated']}",
                      f"- **Prompt:** {_short(a[k]['messages'][-1]['content'], 300)}",
                      f"- **GRPO:** {_short(a[k]['response'])}", f"- **Dr.GRPO:** {_short(b[k]['response'])}\n"]
    tp = dirs["results"] / "group_size_tied_pairs.jsonl"
    if tp.exists():
        ties = read_jsonl_abs(tp)[:6]
        lines.append("## B. Tied-reward completion pairs in the K=8 cache (these make small groups uninformative)\n")
        for t in ties:
            lines += [f"### prompt {t['prompt']} ({t['difficulty']}), generations {t['gen_i']} & {t['gen_j']}: reward {t['reward']:.4f}, "
                      f"identical text: {t['identical_text']}", f"- {_short(t['completion_i'], 300)}", f"- {_short(t['completion_j'], 300)}\n"]
    m, s = gens("midpoint"), gens("standard")
    if m and s:
        ks = [k for k in s if k in m]
        up = sorted(ks, key=lambda k: -(s[k]["reward"] - m[k]["reward"]))[:3]
        lines.append("## C. Standard vs midpoint: largest held-out reward gains\n")
        for k in up:
            lines += [f"### eval row {k}: reward {s[k]['reward']:+.3f} vs {m[k]['reward']:+.3f}; tokens {s[k]['response_tokens']} vs {m[k]['response_tokens']}",
                      f"- **Prompt:** {_short(s[k]['messages'][-1]['content'], 300)}",
                      f"- **standard:** {_short(s[k]['response'])}", f"- **midpoint:** {_short(m[k]['response'])}\n"]
    (dirs["qualitative"] / "candidates.md").write_text("\n".join(lines), encoding="utf-8")
    with pd.option_context("display.max_columns", 40, "display.width", 250):
        print(df.to_string(index=False) if not df.empty else "no held-out results yet")
    print(f"Tables -> {dirs['tables']}\nFigures -> {dirs['figures']}\nQualitative -> {dirs['qualitative']}")


if __name__ == "__main__":
    main()
