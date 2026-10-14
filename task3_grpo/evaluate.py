"""Task 3 - common held-out evaluation protocol (resumable, stage-wise).

python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter midpoint --name midpoint

Stages:
  gen     sampled responses on the fixed held-out prompts (released sampling config, fixed seed
          per batch, cap --eval-max-new-tokens) with per-token policy / reference log-probs
          (sampled-response KL) and full-distribution entropy
  reward  fixed course reward-model score
"""
from __future__ import annotations

import argparse

import numpy as np

from common.data import load_yaml, repo_path
from common.generation import score_reward_pairs
from common.models import clear_gpu, load_reward_model, load_tokenizer
from task3_grpo.utils import (
    describe, generate_eval_records, load_json, load_policy_from, read_jsonl_abs, save_json, select_eval_prompts,
    task_dirs, write_jsonl,
)

ALL_STAGES = ["gen", "reward"]
STAGE_FILES = {"gen": "generations.jsonl", "reward": "rewards.json"}


def _resolve(adapter):
    if adapter is None or str(adapter).lower() in {"none", "sft", "base", ""}:
        return None, "sft"
    if str(adapter).lower() == "midpoint":
        return "midpoint", "midpoint"
    adir = repo_path(adapter)
    if not (adir / "adapter_config.json").exists():
        raise SystemExit(f"adapter not found: {adir} (train it first)")
    return str(adir), str(adir)


def assemble(out_dir, name, label, cap):
    m = {"name": name, "policy": label, "eval_max_new_tokens": cap}
    if (out_dir / "generations.jsonl").exists():
        recs = read_jsonl_abs(out_dir / "generations.jsonl")
        tot = sum(r["response_tokens"] for r in recs)
        h = {
            "n": len(recs),
            "response_tokens": describe([r["response_tokens"] for r in recs]),
            "truncated_rate": float(np.mean([r["truncated"] for r in recs])),
            "eos_rate": float(np.mean([r["terminated_with_eos"] for r in recs])),
            "kl_token_mean": float(sum(r["kl_seq"] for r in recs) / max(tot, 1)),
            "kl_sequence_mean": float(np.mean([r["kl_seq"] for r in recs])),
            "entropy_token_mean": float(sum(r["entropy_sum"] for r in recs) / max(tot, 1)),
        }
        if (out_dir / "rewards.json").exists():
            sc = load_json(out_dir / "rewards.json")["scores"]
            h["reward_model_score"] = describe(sc)
            h["reward_untruncated"] = describe([s for s, r in zip(sc, recs) if not r["truncated"]])
            h["reward_truncated"] = describe([s for s, r in zip(sc, recs) if r["truncated"]])
        m["heldout"] = h
    save_json(out_dir / "metrics.json", m)
    return m


def evaluate_model(config_path, adapter, name, stages=None, work_dir=None, results_dir=None, eval_n=64,
                   gen_batch_size=16, eval_max_new_tokens=None, force=False):
    cfg = load_yaml(config_path)
    seed = int(cfg["seed"])
    cap = int(eval_max_new_tokens or cfg.get("cache_generation_cap", cfg["max_completion_length"]))
    dirs = task_dirs(cfg, work_dir, results_dir)
    out_dir = dirs["eval_results"] / name
    out_dir.mkdir(parents=True, exist_ok=True)
    spec, label = _resolve(adapter)
    stages = list(stages or ALL_STAGES)
    if force:
        for s in stages:
            (out_dir / STAGE_FILES[s]).unlink(missing_ok=True)
    todo = [s for s in stages if not (out_dir / STAGE_FILES[s]).exists()]
    print(f"[eval:{name}] policy={label} cap={cap} todo={todo}", flush=True)
    if "gen" in todo:
        tok = load_tokenizer(cfg["base_model"])
        prompts = select_eval_prompts(cfg, tok, dirs, eval_n)
        model = load_policy_from(cfg, spec, trainable=False)
        recs = generate_eval_records(model, tok, prompts, cfg.get("generation", {}), cap, int(cfg["max_prompt_length"]),
                                     gen_batch_size, seed + 500_000, compute_ref=spec is not None)
        write_jsonl(out_dir / "generations.jsonl", recs)
        del model
        clear_gpu()
    if "reward" in todo:
        recs = read_jsonl_abs(out_dir / "generations.jsonl")
        rm, rm_tok = load_reward_model(cfg)
        scores = []
        for s in range(0, len(recs), 4):
            ch = recs[s : s + 4]
            scores += score_reward_pairs(rm, rm_tok, [r["messages"] for r in ch], [r["response"] for r in ch],
                                         max_length=1280).float().cpu().tolist()
        save_json(out_dir / "rewards.json", {"reward_model": cfg["reward_model"], "scores": scores})
        del rm
        clear_gpu()
    m = assemble(out_dir, name, label, cap)
    h = m.get("heldout", {})
    print(f"[eval:{name}] reward={h.get('reward_model_score', {}).get('mean')} KL={h.get('kl_token_mean')} "
          f"H={h.get('entropy_token_mean')} len={h.get('response_tokens', {}).get('mean')} trunc={h.get('truncated_rate')}", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, 'midpoint' or 'none'")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--work-dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--gen-batch-size", type=int, default=16)
    ap.add_argument("--eval-max-new-tokens", type=int, help="default: cache_generation_cap from the config (768)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    evaluate_model(args.config, args.adapter, args.name, [s for s in args.stages.split(",") if s], args.work_dir,
                   args.results_dir, args.eval_n, args.gen_batch_size, args.eval_max_new_tokens, args.force)


if __name__ == "__main__":
    main()
