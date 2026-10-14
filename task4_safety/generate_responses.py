"""Task 4, Step 1 - deterministic responses of the four FIXED policies on every XSTest prompt.

python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.generate_responses --config configs/feedback.yaml \
       --policies "dpo=/path/task1_dpo/outputs/standard,ppo=/path/task2_ppo/outputs/standard,grpo=/path/task3_grpo/outputs/standard"

SFT = untouched base policy; DPO / PPO / GRPO = the standard Task 1-3 adapters (verified from their
DONE.json; forks / ablations are rejected). Greedy decoding, the common safety_max_new_tokens cap and
the same batch size for every policy. Written batch by batch (resumable at batch boundaries).
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from common.data import load_yaml, repo_path
from common.generation import batch_generate
from common.models import clear_gpu, load_policy, load_tokenizer
from task4_safety.utils import (
    POLICY_ORDER, append_jsonl, check_policy_dir, load_json, parse_policy_overrides, policy_paths, read_jsonl_abs,
    save_json, task_dirs, write_jsonl,
)


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name, adapter, out_path, batch_size=16, limit=None, max_prompt_length=256):
    """Greedy responses for every XSTest prompt in fixed xstest_id order; appends one batch at a time."""
    df = load_xstest(cfg).sort_values("xstest_id").reset_index(drop=True)
    if limit:
        df = df.head(int(limit))
    partial = out_path.with_name(out_path.name + ".partial")
    done_rows = read_jsonl_abs(partial) if partial.exists() else []
    start = len(done_rows)
    if start % batch_size != 0 and start != len(df):
        # keep batches identical to an uninterrupted run (greedy outputs can depend on padding)
        start = (start // batch_size) * batch_size
        write_jsonl(partial, done_rows[:start])
    if start >= len(df):
        partial.replace(out_path)
        return
    print(f"[gen:{policy_name}] adapter={adapter} rows {start}..{len(df) - 1}", flush=True)
    print(f"[gen:{policy_name}] loading {cfg['base_model']}" + (f" + adapter" if adapter else "") +
          " (the first run in a session downloads ~3 GB; this can take a few minutes)...", flush=True)
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    print(f"[gen:{policy_name}] model loaded; generating", flush=True)
    t0 = time.time()
    for s in range(start, len(df), batch_size):
        chunk = df.iloc[s : s + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(model, tokenizer, prompts, max_prompt_length=max_prompt_length,
                             max_new_tokens=int(cfg["safety_max_new_tokens"]), temperature=0.0, top_p=1.0, do_sample=False)
        recs = []
        for j, ((_, row), response) in enumerate(zip(chunk.iterrows(), gen["responses"])):
            recs.append({"xstest_id": int(row["xstest_id"]), "policy": policy_name, "prompt": str(row["prompt"]),
                         "benchmark_class": str(row["benchmark_class"]), "type": str(row["type"]), "response": response,
                         "response_tokens": int(gen["response_lengths"][j]), "truncated": bool(gen["truncated"][j])})
        append_jsonl(partial, recs)
        print(f"  [gen:{policy_name}] {min(s + batch_size, len(df))}/{len(df)} ({time.time() - t0:.0f}s)", flush=True)
    del model
    clear_gpu()
    partial.replace(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--results-dir")
    ap.add_argument("--policies", help="override adapter dirs: dpo=...,ppo=...,grpo=...")
    ap.add_argument("--only", help="comma list of policies to run (default: all four)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, help="first N prompts only (smoke test)")
    ap.add_argument("--allow-unverified", action="store_true", help="skip the standard-run provenance check")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    dirs = task_dirs(cfg, args.results_dir)
    paths = policy_paths(cfg, parse_policy_overrides(args.policies))
    names = [n for n in POLICY_ORDER if not args.only or n in args.only.split(",")]
    prov = {"sft": {"policy": "sft", "path": None, "ok": True, "note": f"{cfg['base_model']} without adapter"}}
    for n in names:
        if n != "sft":
            prov[n] = check_policy_dir(n, paths[n])
            if not prov[n]["ok"] and not args.allow_unverified:
                raise SystemExit(f"[gen] {n}: {prov[n]['problems']} ({prov[n]['path']}). The manual requires the standard "
                                 "Task 1-3 policies; fix the path or pass --allow-unverified.")
    meta = {"decoding": "greedy (do_sample=False)", "max_new_tokens": int(cfg["safety_max_new_tokens"]),
            "max_prompt_length": 256, "batch_size": args.batch_size, "limit": args.limit, "prompt_order": "xstest_id ascending",
            "chat_format": "single user turn, base chat template", "provenance": prov}
    meta_path = dirs["gen"] / "generation_meta.json"
    if meta_path.exists():
        old = load_json(meta_path)
        for k in ("max_new_tokens", "batch_size", "limit"):
            if old.get(k) != meta[k]:
                raise SystemExit(f"[gen] existing generations used {k}={old.get(k)}; keep decoding identical across policies "
                                 f"(or delete {dirs['gen']} to start over).")
        old_prov = old.get("provenance", {})
        old_prov.update(prov)
        meta["provenance"] = old_prov
    save_json(meta_path, meta)
    n_total = len(load_xstest(cfg)) if not args.limit else args.limit
    for n in names:
        out = dirs["gen"] / f"generated_{n}.jsonl"
        if out.exists() and len(read_jsonl_abs(out)) == n_total:
            print(f"[gen:{n}] complete ({n_total} rows) - skipping")
            continue
        generate_for_policy(cfg, n, paths[n], out, args.batch_size, args.limit)
        print(f"[gen:{n}] wrote {out}")
    print("Generation complete:", {n: (dirs['gen'] / f'generated_{n}.jsonl').exists() for n in names})


if __name__ == "__main__":
    main()
