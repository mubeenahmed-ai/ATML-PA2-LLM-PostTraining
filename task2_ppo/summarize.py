"""Task 2 - tables / figures / qualitative candidates from the saved results (no GPU).

python -m task2_ppo.summarize --config configs/ppo.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml
from task2_ppo.continue_train import fork_name
from task2_ppo.utils import load_json, read_jsonl_abs, task_dirs

TRAJ_KEYS = [("reward_model_score", "learned reward (RM score)"), ("kl_token_mean", "KL to reference (token mean)"),
             ("policy_loss", "policy loss"), ("value_loss", "value loss"), ("entropy_mean", "entropy (full dist.)"),
             ("clip_fraction_last_epoch", "clip fraction (last epoch)"), ("grad_norm_policy", "grad norm: policy (solid) / value (dashed)"),
             ("response_length", "response length (tokens)"), ("explained_variance_before", "critic explained variance")]


def _g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _short(t, n=600):
    t = " ".join(str(t).split())
    return t if len(t) <= n else t[:n] + " [...]"


def save_table(df, path):
    df.to_csv(path, index=False)
    try:
        path.with_suffix(".md").write_text(df.to_markdown(index=False, floatfmt=".4f"))
    except Exception:
        pass


def eval_row(name, m, t):
    h = m.get("heldout", {})
    c = m.get("critic", {})
    st = t.get("stability", {})
    return {
        "model": name, "updates": _g(t, "run_meta", "updates"), "clip_epsilon": _g(t, "run_meta", "clip_epsilon"),
        "kl_beta": _g(t, "run_meta", "kl_beta"), "gen_tokens_train": t.get("generated_tokens_total"),
        "heldout_reward": _g(h, "reward_model_score", "mean"), "heldout_reward_std": _g(h, "reward_model_score", "std"),
        "heldout_effective_reward": _g(h, "effective_reward", "mean"), "heldout_kl_token": h.get("kl_token_mean"),
        "heldout_kl_seq": h.get("kl_sequence_mean"), "heldout_entropy": h.get("entropy_token_mean"),
        "heldout_len_mean": _g(h, "response_tokens", "mean"), "heldout_len_std": _g(h, "response_tokens", "std"),
        "heldout_truncated": h.get("truncated_rate"), "critic_token_EV": c.get("token_explained_variance"),
        "critic_token_corr": c.get("token_corr"), "stab_policy_shift_kl_mean": st.get("policy_shift_kl_k3_mean"),
        "stab_policy_shift_kl_max": st.get("policy_shift_kl_k3_max"), "stab_max_abs_log_ratio": st.get("max_abs_log_ratio"),
        "stab_grad_norm_max": st.get("grad_norm_policy_max"), "stab_policy_loss_std": st.get("policy_loss_std"),
        "train_clip_fraction_mean": st.get("clip_fraction_last_epoch_mean"), "train_wall_min": t.get("wall_clock_min"),
        "train_peak_vram_gb": t.get("peak_vram_gb"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
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
    eps_forks = [fork_name(e, cfg["kl_beta"]) for e in cfg["clip_values"]]
    kl_forks = [fork_name(cfg["clip_epsilon"], b) for b in cfg["kl_values"]]
    order = ["midpoint", "standard"] + list(dict.fromkeys(eps_forks + kl_forks))
    names = [n for n in order if n in metrics]
    print("models with held-out results:", names)

    # ---- standard continuation
    if "standard" in logs:
        log = logs["standard"]
        cols = ["update"] + [k for k, _ in TRAJ_KEYS] + ["grad_norm_value", "sample_entropy", "policy_shift_kl_k3",
                                                          "task_reward", "eos_rate", "kl_sequence_mean", "update_seconds"]
        save_table(log[[c for c in cols if c in log]], dirs["tables"] / "standard_trajectory.csv")
        t = trains.get("standard", {})
        first, last = log.head(5), log.tail(5)
        save_table(pd.DataFrame([{"metric": k, "first5_mean": float(first[k].mean()), "last5_mean": float(last[k].mean())}
                                 for k, _ in TRAJ_KEYS]
                                + [{"metric": "wall_clock_min", "first5_mean": None, "last5_mean": t.get("wall_clock_min")},
                                   {"metric": "peak_vram_gb", "first5_mean": None, "last5_mean": t.get("peak_vram_gb")}]),
                   dirs["tables"] / "standard_summary.csv")
        fig, axes = plt.subplots(3, 3, figsize=(14, 10))
        for ax, (k, title) in zip(axes.flat, TRAJ_KEYS):
            ax.plot(log["update"], log[k], "o-", ms=3, alpha=0.5, label="per update")
            ax.plot(log["update"], log[k].rolling(5, min_periods=1).mean(), lw=2, label="rolling mean (5)")
            if k == "grad_norm_policy" and "grad_norm_value" in log:
                ax.plot(log["update"], log["grad_norm_value"], "s--", ms=3, label="value grad norm")
            ax.set_title(title)
            ax.set_xlabel("update")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"Standard PPO continuation ({t.get('wall_clock_min', float('nan')):.1f} min, "
                     f"peak VRAM {t.get('peak_vram_gb', float('nan')):.2f} GB)")
        fig.tight_layout()
        fig.savefig(dirs["figures"] / "standard_trajectories.png", dpi=150)
        plt.close(fig)

    # ---- held-out summary of every model
    df = pd.DataFrame([eval_row(n, metrics[n], trains.get(n, {})) for n in names])
    if not df.empty:
        save_table(df, dirs["tables"] / "heldout_summary.csv")

    # ---- clipping study
    cc = dirs["results"] / "clipping_cached.json"
    if cc.exists():
        c = load_json(cc)
        rows = []
        for e in cfg["clip_values"]:
            pe = c["per_eps"][f"{float(e):.2f}"]
            n = fork_name(e, cfg["kl_beta"])
            r = {"eps": e, "cached_clip_fraction": pe["clip_fraction"], "cached_clip_binding_fraction": pe["clip_binding_fraction"],
                 "cached_clipped_surrogate": pe["clipped_surrogate"], "cached_unclipped_surrogate": pe["unclipped_surrogate"]}
            if n in metrics:
                er = eval_row(n, metrics[n], trains.get(n, {}))
                r.update({k: er[k] for k in ["heldout_reward", "heldout_kl_token", "heldout_len_mean", "heldout_entropy",
                                             "stab_policy_shift_kl_mean", "stab_policy_shift_kl_max", "stab_max_abs_log_ratio",
                                             "stab_grad_norm_max", "train_clip_fraction_mean"]})
            rows.append(r)
        save_table(pd.DataFrame(rows), dirs["tables"] / "clipping_study.csv")
        hp = dirs["results"] / "clipping_cached_ratio_hist.json"
        if hp.exists():
            h = load_json(hp)
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ed = np.array(h["edges"])
            ax.bar(ed[:-1], h["counts"], width=np.diff(ed), align="edge", alpha=0.7)
            for e in cfg["clip_values"]:
                for x in (1 - e, 1 + e):
                    ax.axvline(x, ls="--", lw=1, c={0.05: "C1", 0.2: "C2", 0.5: "C3"}.get(float(e), "k"))
            ax.set_xlabel("importance ratio pi_new / pi_old (cached batch, clipped to [0,3] for display)")
            ax.set_ylabel("tokens")
            ax.set_title("Cached-rollout ratio distribution with clip bands for each eps")
            fig.tight_layout()
            fig.savefig(dirs["figures"] / "cached_ratio_hist.png", dpi=150)
            plt.close(fig)

    # ---- KL study
    rows = []
    for b, n in zip(cfg["kl_values"], kl_forks):
        if n in metrics:
            er = eval_row(n, metrics[n], trains.get(n, {}))
            lg = logs.get(n)
            rows.append({"kl_beta": b, **{k: er[k] for k in ["heldout_reward", "heldout_effective_reward", "heldout_kl_token",
                                                             "heldout_kl_seq", "heldout_entropy", "heldout_len_mean",
                                                             "heldout_truncated"]},
                         "train_reward_last3": float(lg["reward_model_score"].tail(3).mean()) if lg is not None else None,
                         "train_kl_last3": float(lg["kl_token_mean"].tail(3).mean()) if lg is not None else None})
    if rows:
        save_table(pd.DataFrame(rows), dirs["tables"] / "kl_study.csv")

    # ---- fork trajectories
    for title, group, fname in [("clip forks (kl fixed)", eps_forks, "fork_trajectories_clip.png"),
                                ("KL forks (eps fixed)", kl_forks, "fork_trajectories_kl.png")]:
        present = [n for n in group if n in logs]
        if not present:
            continue
        keys = [("reward_model_score", "learned reward"), ("kl_token_mean", "KL to reference"), ("entropy_mean", "entropy"),
                ("response_length", "response length"), ("policy_shift_kl_k3", "per-update policy shift KL"),
                ("clip_fraction_last_epoch", "clip fraction (last epoch)")]
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        for n in present:
            for ax, (k, t) in zip(axes.flat, keys):
                ax.plot(logs[n]["update"], logs[n][k], "o-", ms=3, label=n)
                ax.set_title(t)
                ax.set_xlabel("update")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(dirs["figures"] / fname, dpi=150)
        plt.close(fig)

    # ---- held-out bars
    if not df.empty:
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
        for ax, (k, t) in zip(axes, [("heldout_reward", "held-out RM score"), ("heldout_kl_token", "held-out KL"),
                                     ("heldout_entropy", "held-out entropy"), ("heldout_len_mean", "held-out length")]):
            ax.bar(range(len(df)), df[k])
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df["model"], rotation=60, ha="right", fontsize=7)
            ax.set_title(t)
        fig.tight_layout()
        fig.savefig(dirs["figures"] / "heldout_bars.png", dpi=150)
        plt.close(fig)

    # ---- paired comparisons + qualitative candidates
    def gens(n):
        p = dirs["eval_results"] / n / "generations.jsonl"
        r = dirs["eval_results"] / n / "rewards.json"
        if not (p.exists() and r.exists()):
            return None
        recs = read_jsonl_abs(p)
        for x, s in zip(recs, load_json(r)["scores"]):
            x["reward"] = s
        return {x["row"]: x for x in recs}

    base = gens("midpoint")
    lines = ["# Task 2 qualitative candidates (auto-selected - read and judge them yourself)\n"]
    pair_rows = []
    if base:
        for n in names:
            if n == "midpoint":
                continue
            g = gens(n)
            if not g:
                continue
            common = [k for k in g if k in base]
            dr = np.array([g[k]["reward"] - base[k]["reward"] for k in common])
            dl = np.array([g[k]["response_tokens"] - base[k]["response_tokens"] for k in common])
            same = np.array([g[k]["response"] == base[k]["response"] for k in common])
            tie = (np.abs(dr) <= 1e-6) | same
            pair_rows.append({"model": n, "n": len(common), "identical_response_rate": float(same.mean()),
                              "win_rate_vs_midpoint": float(((dr > 0) & ~tie).mean()), "tie_rate": float(tie.mean()),
                              "loss_rate": float(((dr < 0) & ~tie).mean()),
                              "win_rate_excluding_ties": float(((dr > 0) & ~tie).sum() / max((~tie).sum(), 1)),
                              "mean_reward_diff": float(dr.mean()),
                              "mean_reward_diff_ci95": [float(x) for x in np.percentile(
                                  [np.random.default_rng(i).choice(dr, len(dr)).mean() for i in range(2000)], [2.5, 97.5])],
                              "mean_len_diff": float(dl.mean()),
                              "corr_reward_diff_len_diff": float(np.corrcoef(dr, dl)[0, 1]) if dr.std() > 0 and dl.std() > 0 else None})
        if pair_rows:
            save_table(pd.DataFrame(pair_rows), dirs["tables"] / "paired_vs_midpoint.csv")

        def block(title, n, keys, note):
            g = gens(n)
            if not g:
                return
            lines.append(f"## {title}\n\n{note}\n")
            for k in keys:
                a, b = g[k], base[k]
                lines.extend([f"### eval row {k} (prompt_id {a.get('prompt_id')})",
                          f"- reward {n}={a['reward']:+.3f} vs midpoint={b['reward']:+.3f}; tokens {a['response_tokens']} vs "
                          f"{b['response_tokens']}; truncated {a['truncated']} vs {b['truncated']}",
                          f"- **Prompt:** {_short(a['messages'][-1]['content'], 350)}",
                          f"- **{n}:** {_short(a['response'])}", f"- **midpoint:** {_short(b['response'])}\n"])

        for n in [x for x in ["standard", fork_name(cfg["clip_epsilon"], 0.0)] if x in names]:
            g = gens(n)
            ks = [k for k in g if k in base]
            up_long = sorted([k for k in ks if g[k]["reward"] > base[k]["reward"] and
                              (g[k]["response_tokens"] > 1.3 * base[k]["response_tokens"] or g[k]["truncated"])],
                             key=lambda k: -(g[k]["reward"] - base[k]["reward"]))[:4]
            up_similar = sorted([k for k in ks if g[k]["reward"] > base[k]["reward"] and
                                 abs(g[k]["response_tokens"] - base[k]["response_tokens"]) <= 0.2 * max(base[k]["response_tokens"], 1)],
                                key=lambda k: -(g[k]["reward"] - base[k]["reward"]))[:3]
            down = sorted(ks, key=lambda k: g[k]["reward"] - base[k]["reward"])[:2]
            block(f"{n}: reward up AND much longer / truncated (possible disagreement)", n, up_long,
                  "Is the higher-reward response actually better, or just longer / unfinished?")
            block(f"{n}: reward up at similar length (possible agreement)", n, up_similar,
                  "Check whether the improvement is real (content, correctness, instruction following).")
            block(f"{n}: largest reward drops", n, down, "")
    (dirs["qualitative"] / "candidates.md").write_text("\n".join(lines), encoding="utf-8")
    with pd.option_context("display.max_columns", 40, "display.width", 250):
        print(df.to_string(index=False) if not df.empty else "no held-out results yet")
    print(f"Tables -> {dirs['tables']}\nFigures -> {dirs['figures']}\nQualitative -> {dirs['qualitative']}")


if __name__ == "__main__":
    main()
