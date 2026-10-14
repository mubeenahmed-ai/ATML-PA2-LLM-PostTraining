"""Task 1 - evaluation of a DPO adapter (or the untouched SFT policy).

python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter none --name sft

Stages (each writes its own file and is skipped if that file exists -> resumable):
  pref        held-out DPO loss / preference accuracy / margins on dpo_standard_eval
  length_pref the same on dpo_length_stratified_eval, broken down by length stratum
  gen         sampled responses on a FIXED subset of held-out prompts -> length, KL
  wordlimit   sampled responses on the common word-limit prompts -> compliance, length
  reward      fixed course reward-model score of the gen + wordlimit responses
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, prompt_messages_from_preference, read_jsonl, repo_path
from common.generation import score_reward_pairs
from common.metrics import parse_word_limit, preference_accuracy, safe_corr, word_limit_compliance
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer
from task1_dpo.utils import (
    COMMON_LOSS_BETA,
    describe,
    encode_preference_rows,
    ensure_ref_cache,
    file_fingerprint,
    generate_records,
    load_json,
    policy_logps_for_items,
    read_jsonl_abs,
    ref_cache_meta,
    rel_to_repo,
    save_json,
    softplus_np,
    task_dirs,
    write_jsonl,
)

ALL_STAGES = ["pref", "length_pref", "gen", "wordlimit", "reward"]
STAGE_FILES = {
    "pref": "pref_standard.json",
    "length_pref": "pref_length.json",
    "gen": "generations.jsonl",
    "wordlimit": "wordlimit_generations.jsonl",
    "reward": "rewards.json",
}


# ----------------------------------------------------------------------------- preference metrics
def aggregate_pref(per_ex: list[dict], beta: float) -> dict:
    if not per_ex:
        return {"n": 0}
    dc = torch.tensor([r["chosen_logratio"] for r in per_ex])
    dr = torch.tensor([r["rejected_logratio"] for r in per_ex])
    m = np.array([r["margin"] for r in per_ex])
    ldiff = [r["length_difference"] for r in per_ex]
    return {
        "n": len(per_ex),
        # course helper: fraction with (logpi-logref)(y+) > (logpi-logref)(y-), i.e. m > 0
        "preference_accuracy": preference_accuracy(dc, dr),
        "dpo_loss_train_beta": float(softplus_np(-beta * m).mean()),
        "train_beta": beta,
        "dpo_loss_common_beta": float(softplus_np(-COMMON_LOSS_BETA * m).mean()),
        "common_beta": COMMON_LOSS_BETA,
        "margin": describe(m),
        "implicit_reward_margin_mean": float(beta * m.mean()),
        "chosen_logratio_mean": float(dc.mean()),
        "rejected_logratio_mean": float(dr.mean()),
        "frac_chosen_logratio_negative": float((dc < 0).float().mean()),
        "corr_margin_vs_length_difference": safe_corr(m, ldiff),
    }


def run_pref_stage(model, tokenizer, cfg, dirs, data_key, beta, out_dir, prefix, batch_size):
    path = cfg["paths"][data_key]
    rows = read_jsonl(path)
    items, skipped = encode_preference_rows(tokenizer, rows, int(cfg["max_sequence_length"]))
    ref = ensure_ref_cache(
        model, tokenizer, items, dirs["cache"] / f"ref_{Path(path).stem}.json", meta=ref_cache_meta(cfg, path)
    )
    pol_c, pol_r = policy_logps_for_items(model, tokenizer, items, batch_size=batch_size, desc=prefix)
    per_ex = []
    for it in items:
        k = it["key"]
        dc = pol_c[k] - ref["chosen"][k]
        dr = pol_r[k] - ref["rejected"][k]
        m = dc - dr
        per_ex.append(
            {
                "row": it["row"],
                "prompt_id": it["prompt_id"],
                "length_stratum": it["length_stratum"],
                "chosen_tokens": it["chosen_tokens_full"],
                "rejected_tokens": it["rejected_tokens_full"],
                "length_difference": it["chosen_tokens_full"] - it["rejected_tokens_full"],
                "policy_chosen_logp": pol_c[k],
                "policy_rejected_logp": pol_r[k],
                "ref_chosen_logp": ref["chosen"][k],
                "ref_rejected_logp": ref["rejected"][k],
                "chosen_logratio": dc,
                "rejected_logratio": dr,
                "margin": m,
                "correct": bool(m > 0),
                "dpo_loss_train_beta": float(softplus_np(-beta * m)),
            }
        )
    result = {
        "data_file": rel_to_repo(path),
        "data_sha": file_fingerprint(path),
        "skipped_rows": skipped,
        "overall": aggregate_pref(per_ex, beta),
    }
    strata = sorted({r["length_stratum"] for r in per_ex if r["length_stratum"]})
    if strata:
        result["by_stratum"] = {s: aggregate_pref([r for r in per_ex if r["length_stratum"] == s], beta) for s in strata}
    write_jsonl(out_dir / f"{prefix}_examples.jsonl", per_ex)
    save_json(out_dir / f"{prefix}.json", result)
    acc = result["overall"]["preference_accuracy"]
    print(f"[eval] {prefix}: n={len(per_ex)} accuracy={acc:.3f}", flush=True)
    if strata:
        print("       by stratum: " + ", ".join(f"{s}={result['by_stratum'][s]['preference_accuracy']:.3f}" for s in strata))


# ----------------------------------------------------------------------------- generation prompts
def select_gen_prompts(cfg, tokenizer, dirs, n, max_prompt_tokens):
    """Fixed generation prompt set: the first n held-out prompts (file order) whose chat-
    rendered prompt fits in max_prompt_tokens (so nothing is truncated). Saved once and
    reused for every model so all conditions see identical prompts."""
    sel_path = dirs["results"] / "gen_prompt_ids.json"
    path = cfg["paths"]["dpo_standard_eval"]
    rows = read_jsonl(path)
    if sel_path.exists():
        sel = load_json(sel_path)
        if sel["n_requested"] != n or sel["max_prompt_tokens"] != max_prompt_tokens:
            raise SystemExit(
                f"{sel_path} was created with n={sel['n_requested']}, max_prompt_tokens="
                f"{sel['max_prompt_tokens']}. Keep these fixed across conditions (or delete the file "
                "AND all eval/*/generations.jsonl to change them)."
            )
    else:
        chosen = []
        for i, row in enumerate(rows):
            msgs = prompt_messages_from_preference(row)
            L = len(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
            if L <= max_prompt_tokens:
                chosen.append(i)
            if len(chosen) == n:
                break
        sel = {
            "source_file": rel_to_repo(path),
            "source_sha": file_fingerprint(path),
            "rule": "first n rows of the held-out file (file order) whose rendered prompt <= max_prompt_tokens",
            "n_requested": n,
            "max_prompt_tokens": max_prompt_tokens,
            "rows": chosen,
            "prompt_ids": [rows[i].get("prompt_id") for i in chosen],
        }
        save_json(sel_path, sel)
    return [
        {"row": i, "prompt_id": rows[i].get("prompt_id"), "messages": prompt_messages_from_preference(rows[i])}
        for i in sel["rows"]
    ]


def wordlimit_prompts(cfg, samples):
    out = []
    for r in read_jsonl(cfg["paths"]["word_limit_prompts"]):
        msgs = prompt_messages(r)
        text = msgs[-1]["content"]
        for k in range(samples):
            out.append({"prompt_id": r["prompt_id"], "sample": k, "messages": msgs,
                        "prompt_text": text, "word_limit": parse_word_limit(text)})
    return out


# ----------------------------------------------------------------------------- metrics assembly
def gen_summary(recs, rewards=None):
    toks = [r["response_tokens"] for r in recs]
    tot_tok = sum(toks)
    out = {
        "n": len(recs),
        "response_tokens": describe(toks),
        "word_count": describe([r["word_count"] for r in recs]),
        "truncated_rate": float(np.mean([r["truncated"] for r in recs])) if recs else None,
        "eos_rate": float(np.mean([r["terminated_with_eos"] for r in recs])) if recs else None,
        # sampled-response KL estimator, token-averaged exactly like common.metrics.sampled_kl
        # (sum over all valid response tokens of [log pi - log ref] / number of valid tokens)
        "kl_token_mean": float(sum(r["kl_seq"] for r in recs) / max(tot_tok, 1)),
        # same estimator, sequence convention: mean over responses of the summed log-ratio
        "kl_sequence_mean": float(np.mean([r["kl_seq"] for r in recs])) if recs else None,
        "sample_neg_logp_token_mean": float(-sum(r["policy_logp_sum"] for r in recs) / max(tot_tok, 1)),
    }
    if rewards is not None:
        out["reward"] = describe(rewards)
    return out


def assemble_metrics(out_dir: Path, name: str, adapter, beta):
    m = {"name": name, "adapter": adapter, "beta": beta}
    for key, fn in [("pref_standard", "pref_standard.json"), ("pref_length", "pref_length.json")]:
        if (out_dir / fn).exists():
            m[key] = load_json(out_dir / fn)
            m[key].pop("skipped_rows", None)
    rewards = load_json(out_dir / "rewards.json") if (out_dir / "rewards.json").exists() else {}
    if (out_dir / "generations.jsonl").exists():
        recs = read_jsonl_abs(out_dir / "generations.jsonl")
        m["generation"] = gen_summary(recs, rewards.get("generations"))
    if (out_dir / "wordlimit_generations.jsonl").exists():
        recs = read_jsonl_abs(out_dir / "wordlimit_generations.jsonl")
        comp = [word_limit_compliance(r["prompt_text"], r["response"]) for r in recs]
        per_prompt = {}
        for r, c in zip(recs, comp):
            per_prompt.setdefault(r["prompt_id"], []).append(c)
        m["wordlimit"] = gen_summary(recs, rewards.get("wordlimit"))
        m["wordlimit"]["compliance_rate"] = float(np.mean([c for c in comp if c is not None]))
        m["wordlimit"]["per_prompt_compliance"] = {k: float(np.mean(v)) for k, v in per_prompt.items()}
        m["wordlimit"]["words_over_limit"] = describe(
            [max(0, r["word_count"] - r["word_limit"]) for r in recs if r["word_limit"]]
        )
    save_json(out_dir / "metrics.json", m)
    return m


# ----------------------------------------------------------------------------- main entry
def evaluate_model(
    config_path: str,
    adapter: str | None,
    name: str,
    beta: float | None = None,
    stages: list[str] | None = None,
    work_dir: str | None = None,
    results_dir: str | None = None,
    gen_n: int = 200,
    gen_batch_size: int = 16,
    gen_max_prompt_tokens: int = 512,
    wl_samples: int = 4,
    eval_batch_size: int = 4,
    force: bool = False,
):
    cfg = load_yaml(config_path)
    seed = int(cfg["seed"])
    dirs = task_dirs(cfg, work_dir, results_dir)
    out_dir = dirs["eval_results"] / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if adapter is not None and str(adapter).lower() in {"", "none", "sft", "base"}:
        adapter = None
    if adapter is not None:
        adir = repo_path(adapter)
        if not (adir / "adapter_config.json").exists():
            raise SystemExit(f"[eval:{name}] adapter not found at {adir}. Train it first.")
        if beta is None and (adir / "DONE.json").exists():
            beta = float(load_json(adir / "DONE.json")["run_meta"]["beta"])
        adapter = str(adir)
    beta = float(cfg["beta"] if beta is None else beta)

    stages = list(stages or ALL_STAGES)
    if adapter is None:  # SFT: policy == reference, margins are identically 0
        stages = [s for s in stages if s not in ("pref", "length_pref")]
    if force:
        for s in stages:
            f = out_dir / STAGE_FILES[s]
            if f.exists():
                f.unlink()
    todo = [s for s in stages if not (out_dir / STAGE_FILES[s]).exists()]
    print(f"[eval:{name}] adapter={adapter} beta={beta} stages_todo={todo}", flush=True)

    model_stages = [s for s in todo if s != "reward"]
    if model_stages:
        tokenizer = load_tokenizer(cfg["base_model"])
        model = load_policy(cfg, adapter_path=adapter, trainable=False)
        gen_cfg = cfg.get("generation", {})
        max_new = int(cfg["max_generation_tokens"])
        if "pref" in model_stages:
            run_pref_stage(model, tokenizer, cfg, dirs, "dpo_standard_eval", beta, out_dir, "pref_standard", eval_batch_size)
        if "length_pref" in model_stages:
            run_pref_stage(model, tokenizer, cfg, dirs, "dpo_length_eval", beta, out_dir, "pref_length", eval_batch_size)
        if "gen" in model_stages:
            prompts = select_gen_prompts(cfg, tokenizer, dirs, gen_n, gen_max_prompt_tokens)
            print(f"[eval:{name}] generating {len(prompts)} held-out responses (max_new_tokens={max_new})", flush=True)
            recs = generate_records(model, tokenizer, prompts, gen_cfg, max_new, gen_max_prompt_tokens,
                                    gen_batch_size, seed, compute_ref=adapter is not None)
            write_jsonl(out_dir / "generations.jsonl", recs)
        if "wordlimit" in model_stages:
            prompts = wordlimit_prompts(cfg, wl_samples)
            print(f"[eval:{name}] generating {len(prompts)} word-limit responses", flush=True)
            recs = generate_records(model, tokenizer, prompts, gen_cfg, max_new, 256,
                                    gen_batch_size, seed + 100_000, compute_ref=adapter is not None)
            write_jsonl(out_dir / "wordlimit_generations.jsonl", recs)
        del model
        clear_gpu()

    if "reward" in todo:
        missing = [f for f in ("generations.jsonl", "wordlimit_generations.jsonl") if not (out_dir / f).exists()]
        if missing:
            print(f"[eval:{name}] reward stage skipped; missing {missing}")
        else:
            rm, rm_tok = load_reward_model(cfg)
            scores = {}
            for key, fn in [("generations", "generations.jsonl"), ("wordlimit", "wordlimit_generations.jsonl")]:
                recs = read_jsonl_abs(out_dir / fn)
                vals = []
                for s in range(0, len(recs), 8):
                    chunk = recs[s : s + 8]
                    sc = score_reward_pairs(rm, rm_tok, [r["messages"] for r in chunk],
                                            [r["response"] for r in chunk], max_length=1024)
                    vals += sc.float().cpu().tolist()
                scores[key] = vals
                print(f"[eval:{name}] reward[{key}] mean={np.mean(vals):.4f}", flush=True)
            scores["reward_model"] = cfg["reward_model"]
            save_json(out_dir / "rewards.json", scores)
            del rm
            clear_gpu()

    m = assemble_metrics(out_dir, name, adapter, beta)
    brief = {
        "acc": m.get("pref_standard", {}).get("overall", {}).get("preference_accuracy"),
        "kl_tok": m.get("generation", {}).get("kl_token_mean"),
        "reward": m.get("generation", {}).get("reward", {}).get("mean"),
        "len": m.get("generation", {}).get("response_tokens", {}).get("mean"),
        "wl_compliance": m.get("wordlimit", {}).get("compliance_rate"),
    }
    print(f"[eval:{name}] summary {brief}", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the SFT policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta for the held-out loss (default: read from the run)")
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--work-dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--gen-n", type=int, default=200)
    ap.add_argument("--gen-batch-size", type=int, default=16)
    ap.add_argument("--gen-max-prompt-tokens", type=int, default=512)
    ap.add_argument("--wl-samples", type=int, default=4)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    evaluate_model(
        args.config, args.adapter, args.name, args.beta, [s for s in args.stages.split(",") if s],
        args.work_dir, args.results_dir, args.gen_n, args.gen_batch_size, args.gen_max_prompt_tokens,
        args.wl_samples, args.eval_batch_size, args.force,
    )


if __name__ == "__main__":
    main()
