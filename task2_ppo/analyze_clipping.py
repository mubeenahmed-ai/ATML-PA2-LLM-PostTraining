"""Task 2, Step 2 - clipping study.

python -m task2_ppo.analyze_clipping --config configs/ppo.yaml [--cached-only] [--skip-eval]

(a) Cached batch: reconstruct the supplied rollouts, compute KL-shaped rewards, critic
    values, GAE advantages (whitened over the cached batch) and returns, then for every
    eps evaluate the clipped surrogate, the clip fraction (ratio outside [1-eps, 1+eps]
    before clipping) and the fraction of tokens where the clip binds (zero gradient),
    with the same response mask. "new" = the supplied midpoint policy, "old" = the cached
    old-policy log-probs.
(b) Matched short forks from the identical midpoint for every eps (kl_beta fixed), with
    the same prompts, sampling seeds and update budget, followed by the common evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import score_reward_pairs
from common.metrics import masked_mean
from common.models import clear_gpu, load_reward_model, load_tokenizer
from task2_ppo.continue_train import fork_name, run_ppo
from task2_ppo.evaluate import evaluate_model
from task2_ppo.ppo import clip_binding_fraction, compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards
from task2_ppo.utils import (
    critic_values, describe, load_critic, load_policy_from, response_logps_entropy, save_json, task_dirs, write_jsonl,
)


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


# ----------------------------------------------------------------------------- reconstruction
def _type_desc(v):
    if isinstance(v, torch.Tensor):
        return f"tensor{tuple(v.shape)}:{str(v.dtype).replace('torch.', '')}"
    if isinstance(v, (list, tuple)):
        inner = type(v[0]).__name__ if v else "-"
        return f"{type(v).__name__}[{len(v)}] of {inner}"
    if isinstance(v, str):
        return f"str[{len(v)}]"
    return type(v).__name__


def _vec(x):
    return torch.as_tensor(x, dtype=torch.float32).flatten()


def _ids(x):
    """Token-id SEQUENCE or None. Scalars (e.g. a token COUNT such as `response_tokens: 313`) are
    rejected so they can never be mistaken for a one-token response."""
    if x is None or isinstance(x, (str, int, float, bool)):
        return None
    t = torch.as_tensor(x)
    if t.ndim == 0 or t.numel() < 1 or t.dtype.is_floating_point:
        return None
    return [int(v) for v in t.flatten().tolist()]


ANALYSIS_VERSION = 2


def _first(row, keys):
    for k in keys:
        if k in row and row[k] is not None:
            return k, row[k]
    return None, None


def reconstruct_cached_batch(rows, tok, cfg):
    """Rebuild (prompt token ids, response token ids, old/ref log-probs, mask) for every cached
    rollout, preferring token ids stored in the cache and falling back to re-tokenising the
    response text. Rows whose re-tokenised length disagrees with the cached log-prob length
    are reported and excluded (their log-probs could be misaligned)."""
    prompts = {}
    for key in ("rl_prompt_train", "rl_prompt_eval"):
        for r in read_jsonl(cfg["paths"][key]):
            prompts.setdefault(("src", str(r.get("source_index"))), r)
            prompts.setdefault(("pid", str(r.get("prompt_id"))), r)
    eos = tok.eos_token_id
    items, report = [], Counter()
    for i, c in enumerate(rows):
        old, ref = _vec(c["old_logprobs"]), _vec(c["ref_logprobs"])
        mk, mval = _first(c, ["response_mask", "mask", "completion_mask"])
        if mval is not None:
            n_valid = int(_vec(mval).sum().item())
            old, ref = old[:n_valid], ref[:n_valid]
        # prompt
        msgs = None
        k, v = _first(c, ["messages", "prompt_messages"])
        if v is not None:
            msgs = list(v)
        elif isinstance(c.get("prompt"), list):
            msgs = c["prompt"]
        if msgs is None:
            src = prompts.get(("pid", str(c.get("prompt_id")))) or prompts.get(("src", str(c.get("source_index"))))
            msgs = prompt_messages(src) if src is not None else None
        pk, pv = _first(c, ["prompt_ids", "query_ids", "prompt_input_ids"])
        if pv is not None:
            p_ids, report["prompt_ids_from_cache"] = _ids(pv), report["prompt_ids_from_cache"] + 1
        elif msgs is not None:
            p_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        elif isinstance(c.get("prompt"), str):
            p_ids = tok(c["prompt"], add_special_tokens=False)["input_ids"]
        else:
            report["excluded_no_prompt"] += 1
            continue
        # response ids
        r_ids, source = None, None
        for key in ("response_ids", "response_token_ids", "completion_ids", "response_tokens"):
            ids_ = _ids(c.get(key))
            if ids_ is not None and len(ids_) > 1:
                r_ids, source = ids_, f"cache:{key}"
                break
        if r_ids is not None and mval is not None:
            r_ids = r_ids[: len(old)]
        if r_ids is None:
            r_ids, source = tok(str(c["response"]), add_special_tokens=False)["input_ids"], "retokenized"
            if c.get("terminated_with_eos") or len(r_ids) + 1 == len(old):
                if not r_ids or r_ids[-1] != eos:
                    r_ids = r_ids + [eos]
        expected = len(old)
        n_count = c.get("response_tokens")
        if isinstance(n_count, (int, np.integer)) and int(n_count) != expected:
            report["warning_response_tokens_count_differs_from_logprob_length"] += 1
        if len(r_ids) != expected:
            report[f"excluded_length_mismatch_{source.split(':')[0]}"] += 1
            continue
        L = min(len(r_ids), len(old), len(ref))
        if L == 0:
            report["excluded_empty"] += 1
            continue
        report[f"response_ids_{source}"] += 1
        # effective_terminal_reward already includes the missing-EOS penalty (the reward used in the rollout)
        sk, sv = _first(c, ["effective_terminal_reward", "task_reward", "reward", "score", "rm_score", "reward_model_score"])
        cached_reward = float(_vec(sv)[0]) if sv is not None and _vec(sv).numel() == 1 else None
        vk, vv = _first(c, ["values", "old_values", "value_predictions"])
        cached_values = _vec(vv)[: len(old)] if vv is not None and _vec(vv).numel() >= len(old) else None
        items.append({"i": i, "source_index": c.get("source_index"), "messages": msgs, "prompt_ids": p_ids,
                      "response_ids": r_ids[:L], "old": old[:L], "ref": ref[:L], "response": str(c["response"]),
                      "terminated": r_ids[L - 1] == eos, "cached_reward": cached_reward, "cached_reward_key": sk,
                      "cached_values": cached_values[:L] if cached_values is not None else None})
    report["usable"] = len(items)
    report["total"] = len(rows)
    return items, dict(report)


@torch.no_grad()
def cached_clipping_analysis(cfg, dirs, eps_values, reward_source="auto", logprob_temperature="auto", values_source="auto"):
    out_path = dirs["results"] / "clipping_cached.json"
    if out_path.exists():
        try:
            old_res = __import__("json").loads(out_path.read_text())
        except Exception:
            old_res = {}
        if old_res.get("analysis_version", 1) >= ANALYSIS_VERSION:
            print(f"[clip-cache] exists: {out_path}")
            return
        print(f"[clip-cache] {out_path.name} was produced by an older (buggy) version -> recomputing")
    tok = load_tokenizer(cfg["base_model"])
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    schema = {k: _type_desc(v) for k, v in rows[0].items()}
    print("[clip-cache] rollouts:", len(rows), "| schema of row 0:", schema)
    items, report = reconstruct_cached_batch(rows, tok, cfg)
    print("[clip-cache] reconstruction:", report)
    if not items:
        raise SystemExit("No cached rollout could be reconstructed; see the schema printed above.")

    policy = load_policy_from(cfg, "midpoint", trainable=False)
    critic = load_critic(cfg)
    critic.eval()
    dev = next(policy.parameters()).device
    use_cached = reward_source == "cached" or (reward_source == "auto" and all(it["cached_reward"] is not None for it in items))
    rm_scores = None
    if not use_cached:
        rm, rm_tok = load_reward_model(cfg)
        ok = [it for it in items if it["messages"] is not None]
        if len(ok) != len(items):
            raise SystemExit("Some cached rollouts have no recoverable prompt messages for reward scoring.")
        rm_scores = []
        for s in range(0, len(items), 4):
            ch = items[s : s + 4]
            rm_scores += score_reward_pairs(rm, rm_tok, [it["messages"] for it in ch], [it["response"] for it in ch],
                                            max_length=int(cfg.get("reward_max_length", 1024))).float().cpu().tolist()
        del rm
        clear_gpu()

    beta = float(cfg["kl_beta"])
    pen = float(cfg.get("missing_eos_penalty", 0.0))
    T = max(len(it["response_ids"]) for it in items)
    B = len(items)
    t_gen = float(cfg.get("generation", {}).get("temperature", 1.0))
    temps = [1.0] if logprob_temperature == "1.0" else ([t_gen] if logprob_temperature == "sampling" else sorted({1.0, t_gen}))
    new_by_t = {t: torch.zeros(B, T) for t in temps}
    old = torch.zeros(B, T)
    ref = torch.zeros(B, T)
    val = torch.zeros(B, T)
    mask = torch.zeros(B, T)
    task_r = torch.zeros(B)
    use_cached_values = values_source == "cached" or (values_source == "auto" and all(it["cached_values"] is not None for it in items))
    for b, it in enumerate(items):
        ids = torch.tensor([it["prompt_ids"] + it["response_ids"]], device=dev)
        L, pw = len(it["response_ids"]), len(it["prompt_ids"])
        rid = torch.tensor([it["response_ids"]], device=dev)
        for tkey in temps:
            lp, _ = response_logps_entropy(policy, ids, torch.ones_like(ids), rid, entropy=False, temperature=tkey)
            new_by_t[tkey][b, :L] = lp[0].float().cpu()
        if use_cached_values:
            val[b, :L] = it["cached_values"]
        else:
            val[b, :L] = critic_values(critic, ids, torch.ones_like(ids), pw, L)[0].float().cpu()
        old[b, :L], ref[b, :L], mask[b, :L] = it["old"], it["ref"], 1.0
        if use_cached:
            task_r[b] = it["cached_reward"]
        else:
            task_r[b] = rm_scores[b] - (0.0 if it["terminated"] else pen)
    # which log-prob convention were the cached old log-probs computed with? (raw logits vs logits / T)
    mismatch = {f"{t:g}": float(((new_by_t[t] - old).abs() * mask).sum() / mask.sum()) for t in temps}
    t_used = min(temps, key=lambda t: mismatch[f"{t:g}"])
    new = new_by_t[t_used]
    print(f"[clip-cache] mean |new - old| log-prob per token by convention: {mismatch} -> using temperature {t_used:g}")
    rewards = shaped_rewards(task_r, old, ref, mask, beta)
    adv, ret = compute_gae(rewards, val, mask, float(cfg.get("gamma", 1.0)), float(cfg.get("gae_lambda", 0.95)))
    adv_n = normalize_advantages(adv, mask)  # whitened over the whole cached batch
    log_ratio = (new - old) * mask
    ratio = torch.exp(log_ratio)
    valid_lr = log_ratio[mask.bool()]
    per_eps = {}
    for eps in eps_values:
        loss, _, clipfrac = ppo_policy_loss(new, old, adv_n, mask, eps)
        binds = clip_binding_fraction(ratio, adv_n, mask, eps)
        unclipped = masked_mean(ratio * adv_n, mask)
        per_eps[f"{eps:.2f}"] = {
            "eps": eps,
            "clipped_surrogate": float(-loss),
            "unclipped_surrogate": float(unclipped),
            "surrogate_reduction_by_clipping": float(unclipped + loss),
            "clip_fraction": float(clipfrac),
            "clip_binding_fraction": float(binds),
            "frac_ratio_above_1_plus_eps": float(masked_mean((ratio > 1 + eps).float(), mask)),
            "frac_ratio_below_1_minus_eps": float(masked_mean((ratio < 1 - eps).float(), mask)),
            "per_rollout_clip_fraction": [float(((ratio[b] < 1 - eps) | (ratio[b] > 1 + eps)).float()[mask[b].bool()].mean())
                                          for b in range(B)],
        }
        print(f"[clip-cache] eps={eps:.2f}: clip_fraction={float(clipfrac):.4f} binding={float(binds):.4f} "
              f"surrogate={float(-loss):+.4f} (unclipped {float(unclipped):+.4f})")
    if int(mask.sum()) <= B:
        raise SystemExit("Reconstruction produced <= 1 token per rollout - refusing to report (check the cache schema).")
    res = {
        "analysis_version": ANALYSIS_VERSION,
        "n_rollouts_total": len(rows), "n_rollouts_used": B, "n_tokens": int(mask.sum()),
        "response_tokens_per_rollout": describe(mask.sum(-1).tolist()),
        "values_source": "cached rollout-time values" if use_cached_values else "supplied midpoint critic",
        "cached_reward_key": items[0]["cached_reward_key"] if use_cached else None,
        "reconstruction": report, "cache_schema_row0": schema,
        "new_policy": "supplied PPO midpoint policy (dropout off)", "old_policy": "cached old_logprobs",
        "logprob_convention_mismatch": mismatch, "logprob_temperature_used": t_used,
        "reward_source": (f"cached scalar reward ({items[0]['cached_reward_key']})" if use_cached
                          else "course reward model (+ missing-EOS penalty)"),
        "kl_beta_for_shaping": beta, "advantages": "GAE(gamma, lambda) with the supplied midpoint critic, whitened over the cached batch",
        "ratio": {"log_ratio": describe(valid_lr.tolist()), "ratio": describe(torch.exp(valid_lr).tolist()),
                  "approx_kl_old_to_new_k1": float(-valid_lr.mean()),
                  "approx_kl_k3": float((torch.exp(valid_lr) - 1 - valid_lr).mean()),
                  "mean_abs_new_minus_old_logp": float(valid_lr.abs().mean())},
        "task_reward": describe(task_r.tolist()),
        "advantage_raw": describe(adv[mask.bool()].tolist()),
        "per_eps": per_eps,
    }
    save_json(out_path, res)
    write_jsonl(dirs["results"] / "clipping_cached_rollouts.jsonl", [
        {"cache_index": it["i"], "source_index": it["source_index"], "tokens": len(it["response_ids"]),
         "task_reward": float(task_r[b]), "mean_log_ratio": float(log_ratio[b][mask[b].bool()].mean())}
        for b, it in enumerate(items)])
    hist, edges = np.histogram(torch.exp(valid_lr).clamp(0, 3).numpy(), bins=60, range=(0, 3))
    save_json(dirs["results"] / "clipping_cached_ratio_hist.json", {"counts": hist.tolist(), "edges": edges.tolist()})
    del policy, critic
    clear_gpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--work-dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--save-every", type=int, default=2)
    ap.add_argument("--cached-only", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--reward-source", choices=["auto", "cached", "recompute"], default="auto")
    ap.add_argument("--values-source", choices=["auto", "cached", "critic"], default="auto",
                    help="auto = cached rollout-time values when present, else the supplied midpoint critic")
    ap.add_argument("--logprob-temperature", choices=["auto", "1.0", "sampling"], default="auto",
                    help="convention of the cached old log-probs; auto = whichever matches them better")
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--gen-batch-size", type=int, default=16)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    dirs = task_dirs(cfg, args.work_dir, args.results_dir)
    eps_values = [float(e) for e in cfg["clip_values"]]
    print("Required epsilon values:", eps_values)
    cached_clipping_analysis(cfg, dirs, eps_values, args.reward_source, args.logprob_temperature, args.values_source)
    if args.cached_only:
        return
    for eps in eps_values:
        name = fork_name(eps, cfg["kl_beta"])
        run_ppo(args.config, updates=int(cfg["fork_updates"]), clip_epsilon=eps, kl_beta=float(cfg["kl_beta"]),
                run_name=name, work_dir=args.work_dir, results_dir=args.results_dir, save_every=args.save_every)
        clear_gpu()
        if not args.skip_eval:
            evaluate_model(args.config, str(dirs["work"] / name), name, work_dir=args.work_dir,
                           results_dir=args.results_dir, eval_n=args.eval_n, gen_batch_size=args.gen_batch_size)
            clear_gpu()
    print("Clipping study complete.")


if __name__ == "__main__":
    main()
