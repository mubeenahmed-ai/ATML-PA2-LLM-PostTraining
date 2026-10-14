"""Task 5 - combine the in-domain, controlled-diagnostic and transfer results (no GPU).

python -m task5_feedback.compare_feedback --config configs/feedback.yaml
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl
from task5_feedback.rlvr import exact_reward
from task5_feedback.utils import POLICIES, load_json, read_jsonl_abs, task_dirs


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
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--results-dir")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    dirs = task_dirs(cfg, args.results_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = {ds: load_json(dirs[ds] / "metrics.json") for ds in ("gsm", "transfer") if (dirs[ds] / "metrics.json").exists()}
    D = load_json(dirs["diagnostics"] / "diagnostics.json") if (dirs["diagnostics"] / "diagnostics.json").exists() else None

    # ---- in-domain / transfer tables
    rows = []
    for ds, m in M.items():
        for p in POLICIES:
            v = m["policies"].get(p)
            if not v:
                continue
            pair = m["pairs"].get(f"{p}_vs_sft", {})
            rows.append({"dataset": ds, "policy": p, "n": v["n"], "exact_accuracy": v["exact_accuracy"],
                         "acc_ci_lo": v["exact_accuracy_ci95"][0], "acc_ci_hi": v["exact_accuracy_ci95"][1],
                         "format_compliance": v["format_compliance"], "mean_tokens": v["response_tokens"]["mean"],
                         "std_tokens": v["response_tokens"]["std"], "truncated_rate": v["truncated_rate"],
                         "ai_win_rate_vs_sft": pair.get("ai_pairwise_win_rate", 0.5 if p == "sft" else None),
                         "win_ci_lo": (pair.get("win_rate_ci95") or [None, None])[0],
                         "win_ci_hi": (pair.get("win_rate_ci95") or [None, None])[1],
                         "judge_ties_vs_sft": pair.get("judge_ties"), "judge_explicit_ties": pair.get("judge_explicit_ties"),
                         "judge_ambiguous_outputs": pair.get("judge_ambiguous_outputs"), "identical_to_sft": pair.get("identical_responses"),
                         "verifier_judge_agreement": pair.get("verifier_judge_agreement"),
                         "judge_picks_correct_when_verifier_decisive": pair.get("judge_picks_verifier_correct_when_decisive"),
                         "judge_decisive_when_verifier_ties": pair.get("judge_decisive_when_verifier_ties")})
    df = pd.DataFrame(rows)
    if not df.empty:
        save_table(df, dirs["tables"] / "math_results.csv")
        if {"gsm", "transfer"} <= set(df.dataset):
            g, t = df[df.dataset == "gsm"].set_index("policy"), df[df.dataset == "transfer"].set_index("policy")
            drop = pd.DataFrame({"policy": g.index, "acc_in_domain": g.exact_accuracy.values,
                                 "acc_transfer": t.exact_accuracy.reindex(g.index).values,
                                 "acc_drop": (g.exact_accuracy - t.exact_accuracy.reindex(g.index)).values,
                                 "win_in_domain": g.ai_win_rate_vs_sft.values,
                                 "win_transfer": t.ai_win_rate_vs_sft.reindex(g.index).values,
                                 "tokens_in_domain": g.mean_tokens.values, "tokens_transfer": t.mean_tokens.reindex(g.index).values,
                                 "format_in_domain": g.format_compliance.values,
                                 "format_transfer": t.format_compliance.reindex(g.index).values})
            save_table(drop, dirs["tables"] / "transfer_drop.csv")
        fr = []
        for ds, m in M.items():
            for p, v in m["policies"].items():
                fr.append({"dataset": ds, "policy": p, **v["failure_types"],
                           "gold_mentioned_but_wrong": v["gold_mentioned_but_wrong_rate"],
                           **{f"acc_{k}_parser": val for k, val in v["verifier_variant_accuracy"].items()}})
        save_table(pd.DataFrame(fr).fillna(0.0), dirs["tables"] / "failure_types_and_parsers.csv")
        hh = [{"dataset": ds, **{k: v for k, v in m["pairs"].get("rlvr_vs_rlaif", {}).items() if not isinstance(v, list)}}
              for ds, m in M.items() if "rlvr_vs_rlaif" in m["pairs"]]
        if hh:
            save_table(pd.DataFrame(hh), dirs["tables"] / "rlvr_vs_rlaif_head_to_head.csv")

    # ---- diagnostics tables
    if D:
        pr = []
        for ptype, mechs in D["pairs"].items():
            for mech, r in mechs.items():
                pr.append({"pair": ptype, "mechanism": mech, "n": r["n"], "better_rate": r["better_rate"], "tie_rate": r["tie_rate"],
                           "explicit_ties": r.get("explicit_ties"), "ambiguous_outputs": r.get("ambiguous_outputs"),
                           "wrong_preference_rate": r["wrong_preference_rate"], "better_ci_lo": r["better_rate_ci95"][0],
                           "better_ci_hi": r["better_rate_ci95"][1]})
        save_table(pd.DataFrame(pr), dirs["tables"] / "diagnostic_pairs.csv")
        save_table(pd.DataFrame([{"variant": k, **v} for k, v in D["per_variant"].items()]), dirs["tables"] / "diagnostic_per_variant.csv")
        s = D["summary"]
        save_table(pd.DataFrame([{"mechanism": m, "S_reason": s["S_reason"][m], "S_outcome": s["S_outcome"][m]} for m in s["S_reason"]]),
                   dirs["tables"] / "diagnostic_sensitivity.csv")
        if D.get("position_consistency"):
            pc = D["position_consistency"]
            save_table(pd.DataFrame([{"pair": "ALL", **pc["overall"]}] + [{"pair": k, **v} for k, v in pc["by_pair"].items()]),
                       dirs["tables"] / "judge_position_consistency.csv")

    # ---- feedback-source summary: coverage, noise, exploitability, cost
    if D:
        g = {m: D["pairs"] for m in ("rlvr", "rlaif_pairwise")}
        rows = []
        t0 = time.time()
        diag = read_jsonl(cfg["paths"]["task5_diagnostics"])
        for _ in range(5):
            for r in diag:
                exact_reward(r["response"], str(r["gold_final"]))
        verifier_sec = (time.time() - t0) / (5 * len(diag))
        judge_sec = [x["judge_cost"]["seconds_per_call"] for x in list(M.values()) + [D] if x.get("judge_cost", {}).get("new_calls")]
        for mech in ("rlvr", "rlaif_pairwise"):
            P = D["pairs"]
            rows.append({
                "mechanism": "RLVR (exact verifier)" if mech == "rlvr" else "RLAIF (pairwise judge)",
                "coverage_decisive_rate_all_pairs": float(np.mean([1 - P[p][mech]["tie_rate"] for p in P])),
                "reasoning_sensitivity_S_reason": P["reasoning"][mech]["better_rate"],
                "outcome_sensitivity_S_outcome": P["outcome"][mech]["better_rate"],
                "filler_wrong_preference": P["filler"][mech]["wrong_preference_rate"],
                "distractor_wrong_preference": P["distractor"][mech]["wrong_preference_rate"],
                "order_inconsistency (noise)": (0.0 if mech == "rlvr" else
                                                (1 - D["position_consistency"]["overall"]["order_consistency_rate"]
                                                 if D.get("position_consistency") else None)),
                "seconds_per_reward_call": verifier_sec if mech == "rlvr" else (float(np.mean(judge_sec)) if judge_sec else None),
                "calls_per_K4_group": 4 if mech == "rlvr" else 6,
            })
        save_table(pd.DataFrame(rows), dirs["tables"] / "feedback_source_summary.csv")

    # ---- figures
    if not df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
        for ax, (col, title) in zip(axes, [("exact_accuracy", "exact final-answer accuracy"), ("ai_win_rate_vs_sft", "AI pairwise win rate vs SFT"),
                                           ("mean_tokens", "mean response tokens")]):
            piv = df.pivot(index="policy", columns="dataset", values=col).reindex(POLICIES)
            piv.plot.bar(ax=ax, rot=0)
            ax.set_title(title)
            if col == "ai_win_rate_vs_sft":
                ax.axhline(0.5, ls=":", c="k")
        fig.tight_layout()
        fig.savefig(dirs["figures"] / "math_results.png", dpi=150)
        plt.close(fig)
    if D:
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
        for ax, mech, title in [(axes[0], "rlvr", "RLVR exact verifier"), (axes[1], "rlaif_pairwise", "RLAIF pairwise judge")]:
            pt = list(D["pairs"])
            b = [D["pairs"][p][mech]["better_rate"] for p in pt]
            t = [D["pairs"][p][mech]["tie_rate"] for p in pt]
            w = [D["pairs"][p][mech]["wrong_preference_rate"] for p in pt]
            ax.bar(pt, b, color="#4c9f70", label="prefers better")
            ax.bar(pt, t, bottom=b, color="#9a9a9a", label="tie")
            ax.bar(pt, w, bottom=np.array(b) + np.array(t), color="#c8443c", label="prefers worse")
            ax.set_title(title)
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("fraction of the 20 controlled pairs")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(dirs["figures"] / "diagnostic_pairs.png", dpi=150)
        plt.close(fig)

    # ---- qualitative candidates
    lines = ["# Task 5 qualitative candidates (auto-selected - read and judge them yourself)\n"]
    if D:
        diag = {(str(r["problem_id"]), r["variant_type"]): r for r in read_jsonl(cfg["paths"]["task5_diagnostics"])}
        det = read_jsonl_abs(dirs["diagnostics"] / "controlled_pair_outcomes.jsonl")
        from task5_feedback.score_perturbations import PAIR_TYPES
        for ptype, title in [("reasoning", "Reasoning-only corruption"), ("filler", "Persuasive filler"),
                             ("distractor", "Gold distractor with wrong final"), ("outcome", "Same reasoning, wrong final")]:
            good, bad = PAIR_TYPES[ptype]
            ex = [d for d in det if d["pair"] == ptype and d["rlaif_pairwise"] != "better"][:2] or \
                 [d for d in det if d["pair"] == ptype][:1]
            lines.append(f"## {title}: judge did not prefer the better response (or first example)\n")
            for d in ex:
                g, b = diag[(d["problem_id"], good)], diag[(d["problem_id"], bad)]
                lines += [f"### problem {d['problem_id']}: RLVR {d['rlvr']}, RLAIF pairwise {d['rlaif_pairwise']}",
                          f"- **Question:** {_short(g['question'], 300)} (gold {g['gold_final']})",
                          f"- **{good}:** {_short(g['response'], 400)}", f"- **{bad}:** {_short(b['response'], 400)}\n"]
    for ds in M:
        jp = dirs[ds] / "judge_pairs.jsonl"
        if not jp.exists():
            continue
        gens = {p: {r["id"]: r for r in read_jsonl_abs(dirs[ds] / f"generations_{p}.jsonl")} for p in POLICIES}
        jr = read_jsonl_abs(jp)
        opp = [r for r in jr if r["b"] == "sft" and r["verifier"] != "TIE" and r["judge"] not in ("TIE", r["verifier"])][:3]
        lines.append(f"## [{ds}] judge prefers the response the verifier marks wrong\n")
        for r in opp:
            a, b = gens[r["a"]][r["id"]], gens["sft"][r["id"]]
            lines += [f"### {r['id']}: {r['a']} (exact {r['exact_a']}) vs sft (exact {r['exact_b']}), judge {r['judge']}",
                      f"- **Question:** {_short(a['question'], 300)} (gold {a['gold_final']})",
                      f"- **{r['a']}:** {_short(a['response'])}", f"- **sft:** {_short(b['response'])}\n"]
        both = [r for r in jr if r["b"] == "sft" and r["verifier"] == "TIE" and not r["identical"] and r["judge"] != "TIE"][:2]
        lines.append(f"## [{ds}] verifier ties (both right or both wrong) but the judge expresses a preference\n")
        for r in both:
            a, b = gens[r["a"]][r["id"]], gens["sft"][r["id"]]
            lines += [f"### {r['id']}: {r['a']} vs sft, both exact={r['exact_a']}, judge {r['judge']}",
                      f"- **{r['a']}:** {_short(a['response'], 400)}", f"- **sft:** {_short(b['response'], 400)}\n"]
        if ds == "transfer":
            for p in POLICIES:
                fails = [r for r in gens[p].values() if r["fail_type"] != "correct"][:2]
                lines.append(f"## [transfer] {p} failures\n")
                for r in fails:
                    lines.append(f"- {r['id']} [{r['fail_type']}] gold {r['gold_final']}: {_short(r['response'], 350)}")
                lines.append("")
    (dirs["qualitative"] / "candidates.md").write_text("\n".join(lines), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.to_string(index=False) if not df.empty else "no math results yet")
    print(f"Tables -> {dirs['tables']}\nFigures -> {dirs['figures']}\nQualitative -> {dirs['qualitative']}")


if __name__ == "__main__":
    main()
