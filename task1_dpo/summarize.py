"""Task 1 - collect every saved result into tables / figures / qualitative candidates.

python -m task1_dpo.summarize --config configs/dpo.yaml

Reads only files written by train.py / evaluate.py (no GPU needed). Outputs:
  tables/dpo_summary.csv(.md)        standard + beta forks (+ SFT, length-balanced) summary
  tables/length_strata.csv(.md)      per-stratum accuracy, standard vs length-balanced
  tables/wordlimit_per_prompt.csv    word-limit compliance per prompt and model
  tables/paired_reward_vs_sft.csv    paired reward / length change vs SFT on identical prompts
  figures/*.png                      training curves, beta trade-off, strata, length distributions
  qualitative/candidates.md          automatically surfaced cases to INSPECT MANUALLY
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import load_yaml, preference_responses, read_jsonl
from common.metrics import word_limit_compliance
from task1_dpo.utils import load_json, read_jsonl_abs, task_dirs

ORDER = ["sft", "standard", "beta_0.03", "beta_0.10", "beta_0.30", "length_balanced"]


def _get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _short(text, n=700):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + " [...]"


def load_all(dirs):
    metrics, trains = {}, {}
    for p in sorted(dirs["eval_results"].glob("*/metrics.json")):
        metrics[p.parent.name] = load_json(p)
    for p in sorted(dirs["train_results"].glob("*/train_summary.json")):
        trains[p.parent.name] = load_json(p)
    names = [n for n in ORDER if n in metrics] + [n for n in metrics if n not in ORDER]
    return names, metrics, trains


def summary_table(names, metrics, trains):
    rows = []
    for n in names:
        m, t = metrics[n], trains.get(n, {})
        rm = t.get("run_meta", {})
        ps = _get(m, "pref_standard", "overall", default={})
        g = m.get("generation", {})
        wl = m.get("wordlimit", {})
        rows.append({
            "model": n,
            "budget": ("none (reference policy)" if n == "sft" else
                       f"{rm.get('pairs_used', '?')} pairs x {rm.get('epochs', '?')} ep, {t.get('opt_steps', '?')} opt steps"),
            "train_file": rm.get("data_file"),
            "beta": m.get("beta") if n != "sft" else None,
            "train_wall_min": t.get("wall_clock_min"),
            "train_peak_vram_gb": t.get("peak_vram_gb"),
            "heldout_dpo_loss_train_beta": ps.get("dpo_loss_train_beta"),
            "heldout_dpo_loss_beta0.10": ps.get("dpo_loss_common_beta"),
            "heldout_pref_accuracy": ps.get("preference_accuracy"),
            "heldout_margin_mean": _get(ps, "margin", "mean"),
            "heldout_margin_median": _get(ps, "margin", "median"),
            "heldout_chosen_logratio_mean": ps.get("chosen_logratio_mean"),
            "heldout_rejected_logratio_mean": ps.get("rejected_logratio_mean"),
            "kl_token_mean": g.get("kl_token_mean"),
            "kl_sequence_mean": g.get("kl_sequence_mean"),
            "reward_mean": _get(g, "reward", "mean"),
            "reward_std": _get(g, "reward", "std"),
            "gen_len_mean": _get(g, "response_tokens", "mean"),
            "gen_len_std": _get(g, "response_tokens", "std"),
            "gen_len_median": _get(g, "response_tokens", "median"),
            "gen_len_iqr": _get(g, "response_tokens", "iqr"),
            "gen_truncated_rate": g.get("truncated_rate"),
            "wordlimit_compliance": wl.get("compliance_rate"),
            "wordlimit_words_mean": _get(wl, "word_count", "mean"),
            "wordlimit_reward_mean": _get(wl, "reward", "mean"),
        })
    return pd.DataFrame(rows)


def strata_table(names, metrics):
    rows = []
    for n in names:
        bs = _get(metrics[n], "pref_length", "by_stratum")
        if not bs:
            continue
        for s, a in bs.items():
            rows.append({"model": n, "stratum": s, "n": a["n"], "pref_accuracy": a["preference_accuracy"],
                         "margin_mean": a["margin"]["mean"], "chosen_logratio_mean": a["chosen_logratio_mean"],
                         "rejected_logratio_mean": a["rejected_logratio_mean"],
                         "dpo_loss_train_beta": a["dpo_loss_train_beta"]})
        ov = _get(metrics[n], "pref_length", "overall")
        rows.append({"model": n, "stratum": "ALL", "n": ov["n"], "pref_accuracy": ov["preference_accuracy"],
                     "margin_mean": ov["margin"]["mean"], "chosen_logratio_mean": ov["chosen_logratio_mean"],
                     "rejected_logratio_mean": ov["rejected_logratio_mean"],
                     "dpo_loss_train_beta": ov["dpo_loss_train_beta"],
                     "corr_margin_vs_length_diff": ov["corr_margin_vs_length_difference"]})
    return pd.DataFrame(rows)


def load_gens(dirs, name, fname="generations.jsonl"):
    p = dirs["eval_results"] / name / fname
    if not p.exists():
        return None
    recs = read_jsonl_abs(p)
    rp = dirs["eval_results"] / name / "rewards.json"
    if rp.exists():
        key = "generations" if fname == "generations.jsonl" else "wordlimit"
        for r, v in zip(recs, load_json(rp).get(key, [])):
            r["reward"] = v
    return recs


def paired_vs_sft(dirs, names):
    base = load_gens(dirs, "sft")
    if base is None or "reward" not in base[0]:
        return pd.DataFrame()
    bmap = {r["row"]: r for r in base}
    rows = []
    for n in names:
        if n == "sft":
            continue
        g = load_gens(dirs, n)
        if g is None or "reward" not in g[0]:
            continue
        d_r = [r["reward"] - bmap[r["row"]]["reward"] for r in g if r["row"] in bmap]
        d_l = [r["response_tokens"] - bmap[r["row"]]["response_tokens"] for r in g if r["row"] in bmap]
        rows.append({"model": n, "n_prompts": len(d_r), "reward_win_rate_vs_sft": float(np.mean(np.array(d_r) > 0)),
                     "mean_reward_diff_vs_sft": float(np.mean(d_r)), "mean_len_diff_vs_sft": float(np.mean(d_l)),
                     "corr_reward_diff_len_diff": float(np.corrcoef(d_r, d_l)[0, 1]) if np.std(d_l) > 0 and np.std(d_r) > 0 else None})
    return pd.DataFrame(rows)


def wordlimit_table(dirs, names):
    rows = []
    for n in names:
        g = load_gens(dirs, n, "wordlimit_generations.jsonl")
        if g is None:
            continue
        df = pd.DataFrame(g)
        df["compliant"] = [word_limit_compliance(p, r) for p, r in zip(df["prompt_text"], df["response"])]
        for pid, sub in df.groupby("prompt_id"):
            rows.append({"model": n, "prompt_id": pid, "word_limit": int(sub["word_limit"].iloc[0]),
                         "compliance": float(sub["compliant"].mean()), "mean_words": float(sub["word_count"].mean()),
                         "mean_tokens": float(sub["response_tokens"].mean()),
                         "mean_reward": float(sub["reward"].mean()) if "reward" in sub else None})
    return pd.DataFrame(rows)


def save_table(df, path_csv):
    df.to_csv(path_csv, index=False)
    try:
        Path(str(path_csv).replace(".csv", ".md")).write_text(df.to_markdown(index=False, floatfmt=".4f"))
    except Exception:
        pass


def figures(dirs, names, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = dirs["figures"]
    # 1) training curves
    runs = [p.parent.name for p in sorted(dirs["train_results"].glob("*/train_log.jsonl"))]
    if runs:
        keys = [("loss", "train DPO loss"), ("preference_accuracy", "train batch accuracy"),
                ("margin_mean", "margin m (nats)"), ("chosen_logratio_mean", "log pi/ref (chosen)"),
                ("rejected_logratio_mean", "log pi/ref (rejected)"), ("grad_norm", "grad norm (pre-clip)")]
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        for run in runs:
            log = pd.DataFrame(read_jsonl_abs(dirs["train_results"] / run / "train_log.jsonl"))
            if log.empty:
                continue
            for ax, (k, title) in zip(axes.flat, keys):
                y = log[k].rolling(5, min_periods=1).mean()
                ax.plot(log["pairs_seen"], y, label=run)
                ax.set_title(title)
                ax.set_xlabel("training pairs seen")
        axes.flat[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "training_curves.png", dpi=150)
        plt.close(fig)

    # 2) beta trade-off
    bnames = [n for n in names if n.startswith("beta_")]
    if bnames:
        betas = [metrics[n]["beta"] for n in bnames]
        panels = [("held-out pref. accuracy", lambda m: _get(m, "pref_standard", "overall", "preference_accuracy")),
                  ("KL (token mean, sampled)", lambda m: _get(m, "generation", "kl_token_mean")),
                  ("reward-model score", lambda m: _get(m, "generation", "reward", "mean")),
                  ("generated tokens (mean)", lambda m: _get(m, "generation", "response_tokens", "mean"))]
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
        for ax, (title, f) in zip(axes, panels):
            ys = [f(metrics[n]) for n in bnames]
            ax.plot(betas, ys, "o-", label="short forks")
            if "standard" in metrics and f(metrics["standard"]) is not None:
                ax.axhline(f(metrics["standard"]), ls="--", c="gray", label="standard (1 epoch)")
            if "sft" in metrics and f(metrics["sft"]) is not None:
                ax.axhline(f(metrics["sft"]), ls=":", c="k", label="SFT")
            ax.set_xscale("log")
            ax.set_xticks(betas)
            ax.set_xticklabels([str(b) for b in betas])
            ax.set_xlabel("beta")
            ax.set_title(title)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "beta_tradeoff.png", dpi=150)
        plt.close(fig)

    # 3) strata
    st = strata_table(names, metrics)
    if not st.empty:
        st = st[st["stratum"] != "ALL"]
        piv = st.pivot(index="stratum", columns="model", values="pref_accuracy")
        ax = piv.plot.bar(figsize=(7, 4), rot=0)
        ax.axhline(0.5, ls=":", c="k")
        ax.set_ylabel("held-out preference accuracy")
        ax.set_title("Length-stratified held-out set")
        plt.tight_layout()
        plt.savefig(fig_dir / "length_strata_accuracy.png", dpi=150)
        plt.close()

    # 4) generated-length distributions
    data, labels = [], []
    for n in names:
        g = load_gens(dirs, n)
        if g:
            data.append([r["response_tokens"] for r in g])
            labels.append(n)
    if data:
        fig, ax = plt.subplots(figsize=(1.6 * len(data) + 2, 4))
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20)
        ax.set_ylabel("generated response tokens")
        ax.set_title("Held-out generation length (cap = max_generation_tokens)")
        fig.tight_layout()
        fig.savefig(fig_dir / "generation_length_box.png", dpi=150)
        plt.close(fig)


def qualitative(dirs, cfg, names, k=6):
    """Surface CANDIDATE cases only. Whether a response is actually worse (correctness,
    concision, instruction compliance) must be judged by reading them yourself."""
    lines = ["# Task 1 qualitative candidates (auto-selected - verify manually)\n"]
    base = load_gens(dirs, "sft")
    std = load_gens(dirs, "standard")
    if base and std and "reward" in base[0] and "reward" in std[0]:
        bmap = {r["row"]: r for r in base}
        cands = []
        for r in std:
            b = bmap.get(r["row"])
            if b is None:
                continue
            cands.append((r["reward"] - b["reward"], r["response_tokens"] / max(b["response_tokens"], 1), r, b))
        lines.append("## A. Reward(standard) > Reward(SFT) with a much longer / truncated standard response\n")
        lines.append("Check whether the higher-reward response is actually better or just longer.\n")
        sel = sorted([c for c in cands if c[0] > 0 and (c[1] >= 1.3 or c[2]["truncated"])], key=lambda c: -c[0])[:k]
        for dr, ratio, r, b in sel:
            prompt = r["messages"][-1]["content"]
            lines += [f"### eval row {r['row']} (prompt_id {r['prompt_id']})",
                      f"- reward standard={r['reward']:.3f} vs SFT={b['reward']:.3f} (diff {dr:+.3f}); "
                      f"tokens {r['response_tokens']} vs {b['response_tokens']}; standard truncated={r['truncated']}",
                      f"- **Prompt:** {_short(prompt, 400)}", f"- **Standard DPO:** {_short(r['response'])}",
                      f"- **SFT:** {_short(b['response'])}\n"]
        lines.append("## B. Reward(standard) < Reward(SFT) by a large amount\n")
        for dr, ratio, r, b in sorted(cands, key=lambda c: c[0])[: max(2, k // 2)]:
            lines += [f"### eval row {r['row']}", f"- reward standard={r['reward']:.3f} vs SFT={b['reward']:.3f}; "
                      f"tokens {r['response_tokens']} vs {b['response_tokens']}",
                      f"- **Prompt:** {_short(r['messages'][-1]['content'], 400)}",
                      f"- **Standard DPO:** {_short(r['response'])}", f"- **SFT:** {_short(b['response'])}\n"]

    lines.append("## C. Word-limit prompts: violations that the reward model scores highly\n")
    for n in [x for x in ["standard", "length_balanced", "sft"] if x in names]:
        g = load_gens(dirs, n, "wordlimit_generations.jsonl")
        if not g or "reward" not in g[0]:
            continue
        viol = [r for r in g if r["word_limit"] and r["word_count"] > r["word_limit"]]
        comp = {}
        for r in g:
            if r["word_limit"] and r["word_count"] <= r["word_limit"]:
                comp.setdefault(r["prompt_id"], []).append(r)
        for r in sorted(viol, key=lambda r: -r["reward"])[:3]:
            best_ok = max(comp.get(r["prompt_id"], []), key=lambda x: x["reward"], default=None)
            lines += [f"### [{n}] {r['prompt_id']} (limit {r['word_limit']} words) sample {r['sample']}",
                      f"- violating response: {r['word_count']} words, reward {r['reward']:.3f}",
                      f"- **Prompt:** {r['prompt_text']}", f"- **Response:** {_short(r['response'])}"]
            if best_ok is not None:
                lines.append(f"- compliant sample from same model: {best_ok['word_count']} words, reward "
                             f"{best_ok['reward']:.3f}: {_short(best_ok['response'], 400)}")
            lines.append("")

    lines.append("## D. Length-stratified held-out pairs: strongest wrong preferences\n")
    eval_rows = read_jsonl(cfg["paths"]["dpo_length_eval"])
    for n in [x for x in ["standard", "length_balanced"] if x in names]:
        p = dirs["eval_results"] / n / "pref_length_examples.jsonl"
        if not p.exists():
            continue
        ex = read_jsonl_abs(p)
        for stratum in ["rejected_longer", "preferred_longer"]:
            wrong = sorted([e for e in ex if e["length_stratum"] == stratum and not e["correct"]], key=lambda e: e["margin"])
            for e in wrong[:2]:
                yc, yr = preference_responses(eval_rows[e["row"]])
                lines += [f"### [{n}] {stratum}, eval row {e['row']}: margin {e['margin']:+.2f} "
                          f"(chosen {e['chosen_tokens']} tok, rejected {e['rejected_tokens']} tok)",
                          f"- **Prompt:** {_short(eval_rows[e['row']].get('prompt', ''), 300)}",
                          f"- **Preferred (dataset):** {_short(yc, 500)}", f"- **Rejected (dataset):** {_short(yr, 500)}\n"]
    out = dirs["results"] / "qualitative"
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--work-dir")
    ap.add_argument("--results-dir")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    dirs = task_dirs(cfg, args.work_dir, args.results_dir)
    names, metrics, trains = load_all(dirs)
    if not names:
        raise SystemExit("No evaluation results found yet.")
    print("Models with results:", names)

    df = summary_table(names, metrics, trains)
    save_table(df, dirs["tables"] / "dpo_summary.csv")
    st = strata_table(names, metrics)
    if not st.empty:
        save_table(st, dirs["tables"] / "length_strata.csv")
    wl = wordlimit_table(dirs, names)
    if not wl.empty:
        save_table(wl, dirs["tables"] / "wordlimit_per_prompt.csv")
    pv = paired_vs_sft(dirs, names)
    if not pv.empty:
        save_table(pv, dirs["tables"] / "paired_reward_vs_sft.csv")
    ds = dirs["results"] / "dataset_length_stats.json"
    if ds.exists():
        d = load_json(ds)
        save_table(pd.DataFrame([{"file": k, "n": v["n_pairs"], "frac_preferred_longer": v["frac_preferred_longer"],
                                  "frac_rejected_longer": v["frac_rejected_longer"],
                                  "mean_len_diff": v["length_difference_chosen_minus_rejected"]["mean"],
                                  "median_len_diff": v["length_difference_chosen_minus_rejected"]["median"],
                                  "strata": json.dumps(v["length_stratum_counts"])} for k, v in d.items()]),
                   dirs["tables"] / "dataset_length_stats.csv")
    figures(dirs, names, metrics)
    qualitative(dirs, cfg, names)
    with pd.option_context("display.max_columns", 50, "display.width", 250):
        print(df.to_string(index=False))
    print(f"\nTables  -> {dirs['tables']}\nFigures -> {dirs['figures']}\nQualitative -> {dirs['results'] / 'qualitative'}")


if __name__ == "__main__":
    main()
