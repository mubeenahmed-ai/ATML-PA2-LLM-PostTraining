"""Task 2 - common held-out evaluation protocol (resumable, stage-wise).

python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter midpoint --name midpoint

Stages:
  gen     sampled responses on the fixed held-out prompts (eval_max_response_length cap,
          released sampling config, fixed seed per batch) with per-token policy / reference
          log-probs (sampled-response KL) and full-distribution entropy
  reward  fixed course reward model score; effective reward = score - missing_eos_penalty if truncated
  critic  the run's critic vs Monte-Carlo returns of the held-out samples (explained variance,
          correlation), using the run's KL-shaped rewards
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common.data import load_yaml, repo_path
from common.generation import score_reward_pairs
from common.metrics import safe_corr
from common.models import clear_gpu, load_reward_model, load_tokenizer
from task2_ppo.utils import (
    critic_values,
    describe,
    generate_eval_records,
    load_critic,
    load_json,
    load_policy_from,
    read_jsonl_abs,
    save_json,
    select_eval_prompts,
    task_dirs,
    write_jsonl,
)

ALL_STAGES = ["gen", "reward", "critic"]
STAGE_FILES = {"gen": "generations.jsonl", "reward": "rewards.json", "critic": "critic.json"}


def _resolve(cfg, adapter):
    """-> (policy_spec for load_policy_from, critic_state_path or 'midpoint' or None, kl_beta, label)"""
    if adapter is None or str(adapter).lower() in {"none", "sft", "base", ""}:
        return None, None, float(cfg["kl_beta"]), "sft"
    if str(adapter).lower() == "midpoint":
        return "midpoint", "midpoint", float(cfg["kl_beta"]), "midpoint"
    adir = repo_path(adapter)
    if not (adir / "adapter_config.json").exists():
        raise SystemExit(f"adapter not found: {adir} (train it first)")
    beta = float(cfg["kl_beta"])
    if (adir / "DONE.json").exists():
        beta = float(load_json(adir / "DONE.json")["run_meta"]["kl_beta"])
    crit = adir / "value_trainable.pt"
    return str(adir), (str(crit) if crit.exists() else None), beta, str(adir)


def critic_stage(cfg, tok, recs, rewards, critic_spec, kl_beta, out_dir):
    critic = load_critic(cfg, None if critic_spec == "midpoint" else critic_spec)
    critic.eval()
    dev = next(critic.parameters()).device
    eos_pen = float(cfg.get("missing_eos_penalty", 0.0))
    gamma = float(cfg.get("gamma", 1.0))
    all_v, all_g, first_v, first_g = [], [], [], []
    with torch.no_grad():
        for r, score in zip(recs, rewards):
            p_ids = tok.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=True)
            ids = torch.tensor([p_ids + r["response_ids"]], device=dev)
            T = len(r["response_ids"])
            v = critic_values(critic, ids, torch.ones_like(ids), len(p_ids), T)[0].cpu().numpy()
            rew = -kl_beta * np.asarray(r["logratio_tokens"], dtype=float)
            rew[-1] += score - (0.0 if r["terminated_with_eos"] else eos_pen)
            g = np.zeros(T)
            acc = 0.0
            for t in range(T - 1, -1, -1):
                acc = rew[t] + gamma * acc
                g[t] = acc
            all_v.append(v)
            all_g.append(g)
            first_v.append(float(v[0]))
            first_g.append(float(g[0]))
    v, g = np.concatenate(all_v), np.concatenate(all_g)
    ev = float(1 - np.var(g - v) / np.var(g)) if np.var(g) > 0 else float("nan")
    fv, fg = np.array(first_v), np.array(first_g)
    ev0 = float(1 - np.var(fg - fv) / np.var(fg)) if np.var(fg) > 0 else float("nan")
    res = {
        "critic": critic_spec, "kl_beta_used_for_returns": kl_beta, "n_responses": len(recs), "n_tokens": int(v.size),
        "token_explained_variance": ev, "token_corr": safe_corr(v, g), "token_mse": float(np.mean((v - g) ** 2)),
        "first_token_explained_variance": ev0, "first_token_corr": safe_corr(fv, fg),
        "value_std": float(np.std(v)), "return_std": float(np.std(g)),
        "first_token_value": describe(fv), "first_token_return": describe(fg),
    }
    save_json(out_dir / "critic.json", res)
    print(f"[eval] critic: token EV={ev:+.3f} corr={res['token_corr']:+.3f}; first-token EV={ev0:+.3f}", flush=True)
    del critic
    clear_gpu()


def assemble(out_dir, name, spec, kl_beta, cfg):
    m = {"name": name, "policy": spec, "kl_beta": kl_beta}
    if (out_dir / "generations.jsonl").exists():
        recs = read_jsonl_abs(out_dir / "generations.jsonl")
        tot = sum(r["response_tokens"] for r in recs)
        g = {
            "n": len(recs),
            "response_tokens": describe([r["response_tokens"] for r in recs]),
            "truncated_rate": float(np.mean([r["truncated"] for r in recs])),
            "eos_rate": float(np.mean([r["terminated_with_eos"] for r in recs])),
            "kl_token_mean": float(sum(r["kl_seq"] for r in recs) / max(tot, 1)),
            "kl_sequence_mean": float(np.mean([r["kl_seq"] for r in recs])),
            "entropy_token_mean": float(sum(r["entropy_sum"] for r in recs) / max(tot, 1)),
            "sample_neg_logp_token_mean": float(-sum(r["policy_logp_sum"] for r in recs) / max(tot, 1)),
        }
        if (out_dir / "rewards.json").exists():
            sc = load_json(out_dir / "rewards.json")["scores"]
            pen = float(cfg.get("missing_eos_penalty", 0.0))
            eff = [s - (0.0 if r["terminated_with_eos"] else pen) for s, r in zip(sc, recs)]
            g["reward_model_score"] = describe(sc)
            g["effective_reward"] = describe(eff)
        m["heldout"] = g
    if (out_dir / "critic.json").exists():
        m["critic"] = load_json(out_dir / "critic.json")
    save_json(out_dir / "metrics.json", m)
    return m


def evaluate_model(config_path, adapter, name, stages=None, work_dir=None, results_dir=None,
                   eval_n=64, gen_batch_size=16, force=False):
    cfg = load_yaml(config_path)
    seed = int(cfg["seed"])
    dirs = task_dirs(cfg, work_dir, results_dir)
    out_dir = dirs["eval_results"] / name
    out_dir.mkdir(parents=True, exist_ok=True)
    spec, critic_spec, kl_beta, label = _resolve(cfg, adapter)
    stages = list(stages or ALL_STAGES)
    if critic_spec is None:
        stages = [s for s in stages if s != "critic"]
    if force:
        for s in stages:
            (out_dir / STAGE_FILES[s]).unlink(missing_ok=True)
    todo = [s for s in stages if not (out_dir / STAGE_FILES[s]).exists()]
    print(f"[eval:{name}] policy={label} critic={critic_spec} kl_beta={kl_beta} todo={todo}", flush=True)
    tok = load_tokenizer(cfg["base_model"])

    if "gen" in todo:
        prompts = select_eval_prompts(cfg, tok, dirs, eval_n)
        model = load_policy_from(cfg, spec, trainable=False)
        print(f"[eval:{name}] generating {len(prompts)} held-out responses (cap {cfg['eval_max_response_length']})", flush=True)
        recs = generate_eval_records(model, tok, prompts, cfg.get("generation", {}), int(cfg["eval_max_response_length"]),
                                     int(cfg["max_prompt_length"]), gen_batch_size, seed + 500_000, compute_ref=spec is not None)
        write_jsonl(out_dir / "generations.jsonl", recs)
        del model
        clear_gpu()

    if "reward" in todo:
        recs = read_jsonl_abs(out_dir / "generations.jsonl")
        rm, rm_tok = load_reward_model(cfg)
        scores = []
        for s in range(0, len(recs), 4):
            chunk = recs[s : s + 4]
            scores += score_reward_pairs(rm, rm_tok, [r["messages"] for r in chunk], [r["response"] for r in chunk],
                                         max_length=int(cfg.get("reward_max_length", 1024))).float().cpu().tolist()
        save_json(out_dir / "rewards.json", {"reward_model": cfg["reward_model"], "scores": scores})
        print(f"[eval:{name}] mean reward-model score {np.mean(scores):+.4f}", flush=True)
        del rm
        clear_gpu()

    if "critic" in todo:
        recs = read_jsonl_abs(out_dir / "generations.jsonl")
        scores = load_json(out_dir / "rewards.json")["scores"]
        critic_stage(cfg, tok, recs, scores, critic_spec, kl_beta, out_dir)

    m = assemble(out_dir, name, label, kl_beta, cfg)
    h = m.get("heldout", {})
    print(f"[eval:{name}] reward={h.get('reward_model_score', {}).get('mean')} KL={h.get('kl_token_mean')} "
          f"H={h.get('entropy_token_mean')} len={h.get('response_tokens', {}).get('mean')} trunc={h.get('truncated_rate')}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, 'midpoint' or 'none'")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--work-dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--gen-batch-size", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    evaluate_model(args.config, args.adapter, args.name, [s for s in args.stages.split(",") if s],
                   args.work_dir, args.results_dir, args.eval_n, args.gen_batch_size, args.force)


if __name__ == "__main__":
    main()
