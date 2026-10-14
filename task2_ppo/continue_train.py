"""Task 2 - PPO continuation from the supplied midpoint (resumable).

python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name fork_eps0.05_kl0.10 \
       --updates 8 --clip-epsilon 0.05 --kl-beta 0.10

One update = prompts_per_update prompts -> sampled responses (on-policy rollout) ->
old/reference log-probs, learned reward (+ missing-EOS penalty), KL-shaped token rewards,
critic values, GAE / returns, whitened advantages -> ppo_epochs passes of the clipped policy
loss and the critic MSE. Every run starts from the IDENTICAL supplied policy + critic and
sees the same prompt sequence and per-update sampling seeds.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import set_seed
from common.metrics import masked_mean, sample_entropy, sampled_kl
from common.models import (
    clear_gpu,
    load_reward_model,
    load_tokenizer,
    reference_mode,
    resolve_dtype,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import (
    clip_binding_fraction,
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)
from task2_ppo.utils import (
    append_jsonl,
    clear_resume,
    critic_values,
    explained_variance,
    gpu_peak_gb,
    load_critic,
    load_latest_slot,
    load_policy_from,
    load_trainable_state,
    masked_stats,
    read_jsonl_abs,
    rel_to_repo,
    response_logps,
    response_logps_entropy,
    rng_state,
    save_json,
    save_slot_checkpoint,
    set_rng_state,
    task_dirs,
    train_prompt_order,
    trainable_state,
    truncate_log,
)


def fork_name(eps: float, kl: float) -> str:
    return f"fork_eps{float(eps):.2f}_kl{float(kl):.2f}"


def _step(loss, params, optimizer, scaler, max_norm):
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gn = float(torch.nn.utils.clip_grad_norm_(params, max_norm))
    before = scaler.get_scale() if scaler.is_enabled() else 1.0
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    skipped = bool(scaler.is_enabled() and scaler.get_scale() < before)
    return gn, skipped


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None,
            kl_beta: float | None = None, run_name: str = "standard", work_dir: str | None = None,
            results_dir: str | None = None, save_every: int = 2, force: bool = False,
            max_updates_this_session: int | None = None):
    cfg = load_yaml(config_path)
    U = int(cfg["updates"] if updates is None else updates)
    eps = float(cfg["clip_epsilon"] if clip_epsilon is None else clip_epsilon)
    beta_kl = float(cfg["kl_beta"] if kl_beta is None else kl_beta)
    seed = int(cfg["seed"])
    P = int(cfg.get("prompts_per_update", 1))
    epochs = int(cfg.get("ppo_epochs", 1))
    gamma, lam = float(cfg.get("gamma", 1.0)), float(cfg.get("gae_lambda", 0.95))
    vcoef = float(cfg.get("value_coef", 0.5))
    eos_pen = float(cfg.get("missing_eos_penalty", 0.0))
    max_gn = float(cfg.get("max_grad_norm", 1.0))
    gen_cfg = cfg.get("generation", {})

    dirs = task_dirs(cfg, work_dir, results_dir)
    final_dir = repo_path(output) if output else dirs["work"] / run_name
    resume_dir = dirs["resume"] / run_name
    run_res = dirs["train_results"] / run_name
    log_path = run_res / "train_log.jsonl"
    done_path = final_dir / "DONE.json"
    if done_path.exists() and not force:
        print(f"[ppo:{run_name}] already finished -> {final_dir} (use --force to rerun)")
        return final_dir
    if force:
        clear_resume(resume_dir)
        for p in (done_path, log_path):
            if p.exists():
                p.unlink()

    set_seed(seed)
    tok = load_tokenizer(cfg["base_model"])
    rows, order = train_prompt_order(cfg, tok, dirs)
    if U * P > len(order):
        raise SystemExit(f"Need {U * P} prompts but only {len(order)} are eligible.")

    run_meta = {
        "run_name": run_name, "updates": U, "prompts_per_update": P, "ppo_epochs": epochs,
        "clip_epsilon": eps, "kl_beta": beta_kl, "policy_learning_rate": float(cfg["policy_learning_rate"]),
        "value_lora_learning_rate": float(cfg["value_lora_learning_rate"]),
        "value_head_learning_rate": float(cfg["value_head_learning_rate"]), "value_coef": vcoef,
        "gamma": gamma, "gae_lambda": lam, "missing_eos_penalty": eos_pen, "max_grad_norm": max_gn,
        "max_prompt_length": int(cfg["max_prompt_length"]), "max_response_length": int(cfg["max_response_length"]),
        "reward_max_length": int(cfg.get("reward_max_length", 1024)), "generation": gen_cfg, "seed": seed,
        "start_policy": rel_to_repo(cfg["paths"]["ppo_midpoint_policy"]),
        "start_critic": rel_to_repo(cfg["paths"]["ppo_midpoint_value"]), "reward_model": cfg["reward_model"],
        "reference": f"{cfg['base_model']} (adapter disabled)", "dtype": str(cfg.get("dtype")),
        "prompt_rows": [order[i] for i in range(U * P)],
        "dropout": "disabled in policy and critic", "advantage_whitening": "per update batch (normalize_advantages)",
    }

    # -------- models (every run starts from the identical supplied midpoint) --------
    policy = load_policy_from(cfg, "midpoint", trainable=True)
    critic = load_critic(cfg)
    rm, rm_tok = load_reward_model(cfg)
    pol_params = trainable_parameters(policy)
    opt_pi = AdamW(pol_params, lr=float(cfg["policy_learning_rate"]))  # as in the released scaffold
    opt_v = AdamW(value_parameter_groups(critic, lora_lr=float(cfg["value_lora_learning_rate"]),
                                         head_lr=float(cfg["value_head_learning_rate"])), weight_decay=0.0)
    val_params = [p for g in opt_v.param_groups for p in g["params"]]
    fp16 = torch.cuda.is_available() and resolve_dtype(cfg.get("dtype", "float16")) == torch.float16
    scaler_pi = torch.amp.GradScaler("cuda", init_scale=2.0**12, growth_interval=100, enabled=fp16)
    scaler_v = torch.amp.GradScaler("cuda", init_scale=2.0**12, growth_interval=100, enabled=fp16)

    payload = None if force else load_latest_slot(resume_dir)
    if payload is not None:
        state = payload["state"]
        if state["run_meta"] != run_meta:
            raise SystemExit(f"[ppo:{run_name}] checkpoint in {resume_dir} has a different configuration; use --force.")
        load_trainable_state(policy, payload["policy_trainable"])
        load_trainable_state(critic, payload["value_trainable"])
        opt_pi.load_state_dict(payload["opt_pi"])
        opt_v.load_state_dict(payload["opt_v"])
        scaler_pi.load_state_dict(payload["scaler_pi"])
        scaler_v.load_state_dict(payload["scaler_v"])
        set_rng_state(payload["rng"])
        start_u, elapsed_prev, peak_prev = int(state["update"]), float(state["elapsed_s"]), float(state["peak_vram_gb"])
        truncate_log(log_path, start_u)
        print(f"[ppo:{run_name}] RESUMING from {resume_dir} after update {start_u}")
    else:
        start_u, elapsed_prev, peak_prev = 0, 0.0, 0.0
        if log_path.exists():
            log_path.unlink()
        save_json(run_res / "run_config.json", run_meta)
        save_json(run_res / "prompts_used.json", {
            "rows": run_meta["prompt_rows"], "prompt_ids": [rows[i].get("prompt_id") for i in run_meta["prompt_rows"]]})

    print(f"[ppo:{run_name}] updates={U} eps={eps} kl_beta={beta_kl} ppo_epochs={epochs} start={start_u} fp16_scaler={fp16}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    dev = next(policy.parameters()).device
    t_start = time.time()
    steps_this_session = 0

    for u in range(start_u, U):
        t_u = time.time()
        batch_rows = [rows[i] for i in order[u * P : (u + 1) * P]]
        msgs = [prompt_messages(r) for r in batch_rows]

        # ---------------- rollout (on-policy sample from the current policy) ----------------
        torch.manual_seed(seed + 1000 * u)  # same sampling seed per update index in every fork
        gen = batch_generate(policy, tok, msgs, max_prompt_length=int(cfg["max_prompt_length"]),
                             max_new_tokens=int(cfg["max_response_length"]),
                             temperature=float(gen_cfg.get("temperature", 0.7)), top_p=float(gen_cfg.get("top_p", 0.9)),
                             do_sample=bool(gen_cfg.get("do_sample", True)))
        # batch_generate runs under torch.inference_mode(); clone so the tensors can enter autograd
        seq, attn, rids = gen["sequences"].clone(), gen["attention_mask"].clone(), gen["response_ids"].clone()
        pw = gen["prompt_width"]
        mask = gen["response_mask"].clone().float()
        T = rids.shape[1]
        policy.eval()
        critic.eval()
        with torch.no_grad():
            old_lp, ent = response_logps_entropy(policy, seq, attn, rids)
            old_lp, ent = old_lp * mask, ent * mask
            with reference_mode(policy):
                ref_lp, _ = response_logps_entropy(policy, seq, attn, rids, entropy=False)
            ref_lp = ref_lp * mask
            v_old = critic_values(critic, seq, attn, pw, T) * mask
            rm_score = score_reward_pairs(rm, rm_tok, msgs, gen["responses"],
                                          max_length=int(cfg.get("reward_max_length", 1024))).float().to(dev)
        eos = torch.tensor(gen["terminated_with_eos"], device=dev)
        task_r = rm_score - eos_pen * (~eos).float()
        rewards = shaped_rewards(task_r, old_lp, ref_lp, mask, beta_kl)
        adv, ret = compute_gae(rewards, v_old, mask, gamma, lam)
        adv_n = normalize_advantages(adv, mask)

        # ---------------- PPO epochs ----------------
        policy.train()
        critic.train()
        ep_stats = []
        for ep in range(epochs):
            new_lp = response_logps(policy, seq, attn, rids)
            loss_pi, ratio, clipfrac = ppo_policy_loss(new_lp, old_lp, adv_n, mask, eps)
            binds = clip_binding_fraction(ratio, adv_n, mask, eps)
            gn_pi, skip_pi = _step(loss_pi, pol_params, opt_pi, scaler_pi, max_gn)
            v_pred = critic_values(critic, seq, attn, pw, T)
            loss_v = value_mse_loss(v_pred, ret, mask)
            gn_v, skip_v = _step(vcoef * loss_v, val_params, opt_v, scaler_v, max_gn)
            lr_ = torch.log(ratio.clamp_min(1e-12))
            ep_stats.append({
                "policy_loss": float(loss_pi.detach()), "value_loss": float(loss_v.detach()),
                "clip_fraction": float(clipfrac), "clip_binding_fraction": float(binds),
                "ratio_mean": masked_stats(ratio, mask)[0], "max_abs_log_ratio": float((lr_.abs() * mask).max()),
                "grad_norm_policy": gn_pi, "grad_norm_value": gn_v, "skipped_policy": skip_pi, "skipped_value": skip_v,
            })
            del new_lp, loss_pi, v_pred, loss_v

        # ---------------- post-update diagnostics ----------------
        policy.eval()
        critic.eval()
        with torch.no_grad():
            post_lp, _ = response_logps_entropy(policy, seq, attn, rids, entropy=False)
            post_lp = post_lp * mask
            log_r = (post_lp - old_lp) * mask
            shift_k1 = float(masked_mean(old_lp - post_lp, mask))
            shift_k3 = float(masked_mean((torch.exp(log_r) - 1.0 - log_r), mask))
            v_post = critic_values(critic, seq, attn, pw, T) * mask

        steps_this_session += 1
        elapsed = elapsed_prev + time.time() - t_start
        peak = max(peak_prev, gpu_peak_gb())
        rec = {
            "update": u + 1,
            "prompt_rows": order[u * P : (u + 1) * P],
            "reward_model_score": float(rm_score.mean()),
            "task_reward": float(task_r.mean()),
            "eos_rate": float(eos.float().mean()),
            "kl_token_mean": float(sampled_kl(old_lp, ref_lp, mask)),
            "kl_sequence_mean": float(((old_lp - ref_lp) * mask).sum(-1).mean()),
            "entropy_mean": float(masked_mean(ent, mask)),
            "sample_entropy": float(sample_entropy(old_lp, mask)),
            "response_length": float(mask.sum(-1).mean()),
            "shaped_return_mean": float(rewards.sum(-1).mean()),
            "value_mean": masked_stats(v_old, mask)[0],
            "value_first_token": float(v_old[:, 0].mean()),
            "return_mean": masked_stats(ret, mask)[0],
            "advantage_raw_std": masked_stats(adv, mask)[1],
            "explained_variance_before": explained_variance(v_old, ret, mask),
            "explained_variance_after": explained_variance(v_post, ret, mask),
            "policy_loss": float(np.mean([e["policy_loss"] for e in ep_stats])),
            "policy_loss_last_epoch": ep_stats[-1]["policy_loss"],
            "value_loss": float(np.mean([e["value_loss"] for e in ep_stats])),
            "clip_fraction": float(np.mean([e["clip_fraction"] for e in ep_stats])),
            "clip_fraction_last_epoch": ep_stats[-1]["clip_fraction"],
            "clip_binding_fraction_last_epoch": ep_stats[-1]["clip_binding_fraction"],
            "max_abs_log_ratio": float(max(e["max_abs_log_ratio"] for e in ep_stats)),
            "grad_norm_policy": float(np.mean([e["grad_norm_policy"] for e in ep_stats])),
            "grad_norm_policy_max": float(max(e["grad_norm_policy"] for e in ep_stats)),
            "grad_norm_value": float(np.mean([e["grad_norm_value"] for e in ep_stats])),
            "policy_shift_kl_k1": shift_k1,
            "policy_shift_kl_k3": shift_k3,
            "skipped_steps": int(sum(e["skipped_policy"] + e["skipped_value"] for e in ep_stats)),
            "epochs": ep_stats,
            "generated_tokens": int(mask.sum()),
            "update_seconds": time.time() - t_u,
            "elapsed_s": elapsed,
            "peak_vram_gb": peak,
        }
        append_jsonl(log_path, rec)
        print(f"[ppo:{run_name}] update {u + 1}/{U} reward={rec['reward_model_score']:+.3f} "
              f"KL={rec['kl_token_mean']:.4f} H={rec['entropy_mean']:.3f} len={rec['response_length']:.0f} "
              f"Lpi={rec['policy_loss']:+.4f} Lv={rec['value_loss']:.3f} clip={rec['clip_fraction_last_epoch']:.3f} "
              f"EV={rec['explained_variance_before']:+.2f} | {rec['update_seconds']:.0f}s, total {elapsed / 60:.1f} min",
              flush=True)
        del gen, seq, attn, rids, old_lp, ref_lp, v_old, post_lp, v_post
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        stop_now = max_updates_this_session is not None and steps_this_session >= max_updates_this_session
        if u + 1 < U and ((u + 1) % save_every == 0 or stop_now):
            save_slot_checkpoint(resume_dir, {
                "state": {"run_meta": run_meta, "update": u + 1, "elapsed_s": elapsed, "peak_vram_gb": peak,
                          "total_updates": U},
                "policy_trainable": trainable_state(policy), "value_trainable": trainable_state(critic),
                "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "scaler_pi": scaler_pi.state_dict(),
                "scaler_v": scaler_v.state_dict(), "rng": rng_state()}, u + 1)
            print(f"[ppo:{run_name}] checkpoint saved after update {u + 1}", flush=True)
        if stop_now and u + 1 < U:
            print(f"[ppo:{run_name}] stopping after {steps_this_session} updates this session (resumable).")
            del policy, critic, rm
            clear_gpu()
            return None

    elapsed = elapsed_prev + time.time() - t_start
    peak = max(peak_prev, gpu_peak_gb())
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(final_dir))
    torch.save(trainable_state(critic), final_dir / "value_trainable.pt")
    logs = read_jsonl_abs(log_path)
    summary = {
        "run_meta": run_meta, "adapter_dir": str(final_dir), "updates_done": len(logs),
        "wall_clock_s": elapsed, "wall_clock_min": elapsed / 60, "peak_vram_gb": peak,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "generated_tokens_total": int(sum(r["generated_tokens"] for r in logs)),
        "skipped_steps_total": int(sum(r["skipped_steps"] for r in logs)),
        "stability": {
            "policy_shift_kl_k3_mean": float(np.mean([r["policy_shift_kl_k3"] for r in logs])),
            "policy_shift_kl_k3_max": float(np.max([r["policy_shift_kl_k3"] for r in logs])),
            "max_abs_log_ratio": float(np.max([r["max_abs_log_ratio"] for r in logs])),
            "grad_norm_policy_max": float(np.max([r["grad_norm_policy_max"] for r in logs])),
            "policy_loss_std": float(np.std([r["policy_loss"] for r in logs])),
            "clip_fraction_last_epoch_mean": float(np.mean([r["clip_fraction_last_epoch"] for r in logs])),
            "reward_std_over_updates": float(np.std([r["reward_model_score"] for r in logs])),
        },
    }
    save_json(run_res / "train_summary.json", summary)
    save_json(done_path, summary)
    clear_resume(resume_dir)
    print(f"[ppo:{run_name}] DONE in {elapsed / 60:.1f} min, peak VRAM {peak:.2f} GB -> {final_dir}")
    del policy, critic, rm
    clear_gpu()
    return final_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--work-dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--save-every", type=int, default=2, help="updates between resume checkpoints")
    ap.add_argument("--max-updates-this-session", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name,
            args.work_dir, args.results_dir, args.save_every, args.force, args.max_updates_this_session)


if __name__ == "__main__":
    main()
