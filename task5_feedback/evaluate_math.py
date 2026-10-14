"""Task 5, Steps 1 and 3 - SFT / RLVR / RLAIF on the fixed GSM8K subset (in-domain) and the fixed SVAMP
transfer subset (out-of-domain).

python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer

Stages (resumable): generate - greedy responses (math_max_new_tokens) for the three policies, written per batch;
judge - the released PairwiseAIJudge on (RLVR vs SFT), (RLAIF vs SFT) and (RLVR vs RLAIF); metrics -
exact accuracy, format compliance, length, failure types, pairwise win rates, verifier-judge agreement.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import batch_generate
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge  # noqa: F401  (released judge, used via utils.make_judge)
from task5_feedback.rlvr import exact_reward
from task5_feedback.utils import (
    POLICIES, append_jsonl, backup_judge_cache, bootstrap_ci, describe, failure_type, load_json, make_judge, pref_score,
    read_jsonl_abs, released_raw_output, save_json, task_dirs, tie_kind, timed_compare, verifier_pref, verifier_variants,
    wilson, write_jsonl,
)

PAIRS = [("rlvr", "sft"), ("rlaif", "sft"), ("rlvr", "rlaif")]


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def row_id(row):
    return str(row.get("source_index", row.get("prompt_id")))


def generate_policy(cfg, rows, name, out_path, batch_size, limit=None):
    if limit:
        rows = rows[:limit]
    partial = out_path.with_name(out_path.name + ".partial")
    done = read_jsonl_abs(partial) if partial.exists() else []
    start = (len(done) // batch_size) * batch_size if len(done) < len(rows) else len(rows)
    if start < len(done):
        write_jsonl(partial, done[:start])
    if start >= len(rows):
        partial.replace(out_path)
        return
    print(f"[gen:{name}] loading policy (the first load in a session downloads ~3 GB)...", flush=True)
    tok = load_tokenizer(cfg["base_model"])
    model = load_frozen_policy(cfg, name)
    t0 = time.time()
    for s in range(start, len(rows), batch_size):
        chunk = rows[s : s + batch_size]
        gen = batch_generate(model, tok, [r["messages"] for r in chunk], max_prompt_length=512,
                             max_new_tokens=int(cfg["math_max_new_tokens"]), temperature=0.0, top_p=1.0, do_sample=False)
        recs = []
        for j, r in enumerate(chunk):
            resp = gen["responses"][j]
            ft = failure_type(resp, r["gold_final"], bool(gen["truncated"][j]))
            recs.append({"id": row_id(r), "policy": name, "question": r["question"], "gold_final": str(r["gold_final"]),
                         "response": resp, "response_tokens": int(gen["response_lengths"][j]),
                         "truncated": bool(gen["truncated"][j]), "exact": exact_reward(resp, str(r["gold_final"])),
                         **{f"fail_{k}": v for k, v in ft.items()},
                         **{f"verifier_{k}": v for k, v in verifier_variants(resp, r["gold_final"]).items()}})
        append_jsonl(partial, recs)
        print(f"  [gen:{name}] {min(s + batch_size, len(rows))}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    del model
    clear_gpu()
    partial.replace(out_path)


def judge_pairs(cfg, dirs, ds_dir, gens, stats):
    judge = make_judge(cfg, dirs["judge"] / "pairwise_cache.json")
    raw_p = dirs["judge"] / "tie_raw_cache.json"
    raw_cache = load_json(raw_p) if raw_p.exists() else {}
    out = []
    ids = [i for i in gens["sft"] if all(i in gens[p] for p in POLICIES)]
    for a_name, b_name in PAIRS:
        t0 = time.time()
        for k, i in enumerate(ids):
            a, b = gens[a_name][i], gens[b_name][i]
            res = timed_compare(judge, a["question"], a["response"], b["response"], stats)
            vp = verifier_pref(a["exact"], b["exact"])
            ident = a["response"] == b["response"]
            kind = None
            if res == "TIE" and not ident:   # explicit TIE or an unparseable output mapped to TIE?
                kind = tie_kind(released_raw_output(judge, a["question"], a["response"], b["response"], raw_cache))
            out.append({"id": i, "a": a_name, "b": b_name, "judge": res, "score_a": pref_score(res), "verifier": vp,
                        "exact_a": a["exact"], "exact_b": b["exact"], "identical": ident, "tie_kind": kind})
            if (k + 1) % 25 == 0:
                backup_judge_cache(judge)
                save_json(raw_p, raw_cache)
                print(f"  [judge] {a_name} vs {b_name}: {k + 1}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)
        backup_judge_cache(judge)
        save_json(raw_p, raw_cache)
    del judge
    clear_gpu()
    write_jsonl(ds_dir / "judge_pairs.jsonl", out)


def metrics(ds_dir, dataset, gens, stats):
    m = {"dataset": dataset, "policies": {}, "pairs": {}}
    for p in POLICIES:
        recs = list(gens[p].values())
        n = len(recs)
        k = int(sum(r["exact"] for r in recs))
        acc, lo, hi = wilson(k, n)
        types = {}
        for r in recs:
            types[r["fail_type"]] = types.get(r["fail_type"], 0) + 1
        m["policies"][p] = {
            "n": n, "exact_accuracy": acc, "exact_accuracy_ci95": [lo, hi],
            "format_compliance": float(np.mean([r["fail_format_ok"] for r in recs])),
            "response_tokens": describe([r["response_tokens"] for r in recs]),
            "truncated_rate": float(np.mean([r["truncated"] for r in recs])),
            "failure_types": {t: c / n for t, c in sorted(types.items())},
            "gold_mentioned_but_wrong_rate": float(np.mean([r["fail_gold_mentioned_but_wrong"] for r in recs])),
            "verifier_variant_accuracy": {v: float(np.mean([r[f"verifier_{v}"] for r in recs])) for v in ("strict", "last_num", "any_num")},
        }
    jp = ds_dir / "judge_pairs.jsonl"
    if jp.exists():
        rows = read_jsonl_abs(jp)
        for a_name, b_name in PAIRS:
            sub = [r for r in rows if r["a"] == a_name and r["b"] == b_name]
            if not sub:
                continue
            scores = [r["score_a"] for r in sub]
            dec = [r for r in sub if r["verifier"] != "TIE"]
            vtie = [r for r in sub if r["verifier"] == "TIE" and not r["identical"]]
            m["pairs"][f"{a_name}_vs_{b_name}"] = {
                "n": len(sub), "ai_pairwise_win_rate": float(np.mean(scores)), "win_rate_ci95": bootstrap_ci(scores),
                "wins": int(sum(r["judge"] == "A" for r in sub)), "losses": int(sum(r["judge"] == "B" for r in sub)),
                "judge_ties": int(sum(r["judge"] == "TIE" for r in sub)), "identical_responses": int(sum(r["identical"] for r in sub)),
                "judge_explicit_ties": int(sum(r.get("tie_kind") == "explicit_tie" for r in sub)),
                "judge_ambiguous_outputs": int(sum(r.get("tie_kind") == "ambiguous_output" for r in sub)),
                "verifier_judge_agreement": float(np.mean([r["judge"] == r["verifier"] for r in sub])),
                "n_verifier_decisive": len(dec),
                "judge_picks_verifier_correct_when_decisive": float(np.mean([r["judge"] == r["verifier"] for r in dec])) if dec else None,
                "judge_tie_when_verifier_decisive": float(np.mean([r["judge"] == "TIE" for r in dec])) if dec else None,
                "judge_opposes_verifier_when_decisive": float(np.mean([r["judge"] not in ("TIE", r["verifier"]) for r in dec])) if dec else None,
                "n_verifier_tie_nonidentical": len(vtie),
                "judge_decisive_when_verifier_ties": float(np.mean([r["judge"] != "TIE" for r in vtie])) if vtie else None,
            }
    m["judge_cost"] = {"new_calls": stats.get("calls", 0), "seconds": stats.get("seconds", 0.0),
                       "seconds_per_call": stats.get("seconds", 0.0) / max(stats.get("calls", 0), 1),
                       "identical_pairs_short_circuited": stats.get("identical_short_circuit", 0)}
    old = ds_dir / "metrics.json"
    if old.exists() and m["judge_cost"]["new_calls"] == 0:   # keep the timing measured on the run that made the calls
        m["judge_cost"] = load_json(old).get("judge_cost", m["judge_cost"])
    save_json(ds_dir / "metrics.json", m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--results-dir")
    ap.add_argument("--stage", choices=["all", "generate", "judge", "metrics"], default="all")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, help="first N problems only (smoke test)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    dirs = task_dirs(cfg, args.results_dir)
    ds_dir = dirs[args.dataset]
    rows = read_jsonl(dataset_path(cfg, args.dataset))
    print(f"[{args.dataset}] {len(rows)} problems" + (f" (limit {args.limit})" if args.limit else ""))
    meta = {"dataset_file": dataset_path(cfg, args.dataset), "n_rows": len(rows), "limit": args.limit, "decoding": "greedy",
            "max_new_tokens": int(cfg["math_max_new_tokens"]), "batch_size": args.batch_size,
            "policies": {k: v for k, v in policy_specs(cfg).items()}, "judge_model": cfg["ai_judge_model"]}
    mp = ds_dir / "generation_meta.json"
    if args.stage in ("all", "generate"):   # decoding settings only matter when (re)generating
        if mp.exists():
            old = load_json(mp)
            for key in ("limit", "max_new_tokens", "batch_size"):
                if old.get(key) != meta[key]:
                    raise SystemExit(f"existing generations used {key}={old.get(key)}; keep decoding identical "
                                     f"(pass --batch-size {old.get('batch_size')} / --limit {old.get('limit')}, or delete {ds_dir}).")
        save_json(mp, meta)
        for p in POLICIES:
            out = ds_dir / f"generations_{p}.jsonl"
            if out.exists() and len(read_jsonl_abs(out)) == (args.limit or len(rows)):
                print(f"[gen:{p}] complete - skipping")
                continue
            generate_policy(cfg, rows, p, out, args.batch_size, args.limit)
    gens = {p: {r["id"]: r for r in read_jsonl_abs(ds_dir / f"generations_{p}.jsonl")}
            for p in POLICIES if (ds_dir / f"generations_{p}.jsonl").exists()}
    stats = {}
    if args.stage in ("all", "judge"):
        if len(gens) < 3:
            raise SystemExit("generate all three policies first")
        print("[judge] loading the fixed pairwise judge (4-bit; the first load downloads ~6 GB)...", flush=True)
        judge_pairs(cfg, dirs, ds_dir, gens, stats)
    if len(gens) == 3:
        m = metrics(ds_dir, args.dataset, gens, stats)
        for p, v in m["policies"].items():
            print(f"[{args.dataset}] {p}: exact {v['exact_accuracy']:.3f} format {v['format_compliance']:.3f} "
                  f"len {v['response_tokens']['mean']:.0f}")
        for k, v in m["pairs"].items():
            print(f"[{args.dataset}] {k}: AI win rate {v['ai_pairwise_win_rate']:.3f} (ties {v['judge_ties']}, identical "
                  f"{v['identical_responses']}), verifier-judge agreement {v['verifier_judge_agreement']:.3f}")


if __name__ == "__main__":
    main()
