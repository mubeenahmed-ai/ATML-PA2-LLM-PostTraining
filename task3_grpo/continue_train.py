"""Task 3 - GRPO continuation from the supplied midpoint (resumable).

python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name fork_dr_grpo --updates 8 \
       --loss-type dr_grpo --seq-grad-norms

One update = prompts_per_update prompt(s) x num_generations sampled completions (cap
max_completion_length) -> reward-model scores -> within-prompt group-relative advantages ->
completions that hit the cap are masked from the loss (mask_truncated_completions) ->
policy_epochs passes of the clipped GRPO objective with the k3 KL penalty to the frozen
reference. Every run starts from the identical supplied midpoint, sees the same prompt
sequence and the same sampling seed per update index.
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
from common.models import clear_gpu, load_reward_model, load_tokenizer, reference_mode, resolve_dtype, trainable_parameters
from task3_grpo.grpo import (
    INFORMATIVE_EPS,
    group_relative_advantages,
    group_statistics,
    grpo_microbatch_loss,
    mask_truncated_sequences,
)
from task3_grpo.utils import (
    append_jsonl,
    clear_resume,
    gpu_peak_gb,
    load_latest_slot,
    load_policy_from,
    load_trainable_state,
    read_jsonl_abs,
    rel_to_repo,
    response_logps_entropy,
    response_logps_with_entropy,
    rng_state,
    save_json,
    save_slot_checkpoint,
    set_rng_state,
    task_dirs,
    train_prompt_order,
    trainable_state,
    truncate_log,
)


def _seq_policy_grad_norms(policy, params, seq, attn, rids, adv, train_mask, n_total, loss_type, max_len, scaler):
    """Length-conditioned diagnostic: norm of the policy-term gradient contributed by EACH
    completion alone (same normaliser as in the full loss, ratio = 1 at this point)."""
    norms = []
    for k in range(seq.shape[0]):
        if float(train_mask[k].sum()) == 0.0:
            norms.append(0.0)
            continue
        lp, _ = response_logps_with_entropy(policy, seq[k : k + 1], attn[k : k + 1], rids[k : k + 1])
        m = train_mask[k : k + 1]
        tok_sum = (lp * adv[k] * m).sum()
        denom = m.sum().clamp_min(1.0) if loss_type == "grpo" else float(max_len)
        part = -(tok_sum / denom) / float(n_total)
        scale = scaler.get_scale() if scaler.is_enabled() else 1.0
        (part * scale).backward()
        g = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)) / scale
        norms.append(float(g))
        for p in params:
            p.grad = None
    return norms


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo",
             run_name: str = "standard", work_dir: str | None = None, results_dir: str | None = None,
             save_every: int = 2, force: bool = False, max_updates_this_session: int | None = None,
             train_micro_batch: int = 2, seq_grad_norms: bool = False):
    cfg = load_yaml(config_path)
    U = int(cfg["updates"] if updates is None else updates)
    seed = int(cfg["seed"])
    P = int(cfg.get("prompts_per_update", 1))
    K = int(cfg["num_generations"])
    epochs = int(cfg.get("policy_epochs", 1))
    eps = float(cfg["clip_epsilon"])
    beta = float(cfg["kl_beta"])
    max_len = int(cfg["max_completion_length"])
    mask_trunc = bool(cfg.get("mask_truncated_completions", True))
    max_gn = float(cfg.get("max_grad_norm", 1.0))
    gen_cfg = cfg.get("generation", {})

    dirs = task_dirs(cfg, work_dir, results_dir)
    final_dir = repo_path(output) if output else dirs["work"] / run_name
    resume_dir = dirs["resume"] / run_name
    run_res = dirs["train_results"] / run_name
    log_path = run_res / "train_log.jsonl"
    comp_path = run_res / "completions_log.jsonl"
    done_path = final_dir / "DONE.json"
    if done_path.exists() and not force:
        print(f"[grpo:{run_name}] already finished -> {final_dir} (use --force to rerun)")
        return final_dir
    if force:
        clear_resume(resume_dir)
        for p in (done_path, log_path, comp_path):
            if p.exists():
                p.unlink()

    set_seed(seed)
    tok = load_tokenizer(cfg["base_model"])
    rows, order = train_prompt_order(cfg, tok, dirs)
    run_meta = {
        "run_name": run_name, "updates": U, "prompts_per_update": P, "num_generations": K, "policy_epochs": epochs,
        "loss_type": loss_type, "clip_epsilon": eps, "kl_beta": beta, "learning_rate": float(cfg["learning_rate"]),
        "max_prompt_length": int(cfg["max_prompt_length"]), "max_completion_length": max_len,
        "mask_truncated_completions": mask_trunc, "max_grad_norm": max_gn, "generation": gen_cfg, "seed": seed,
        "start_policy": rel_to_repo(cfg["paths"]["grpo_midpoint_policy"]), "reward_model": cfg["reward_model"],
        "reference": f"{cfg['base_model']} (adapter disabled)", "dtype": str(cfg.get("dtype")),
        "advantage": "within-group (r - mean) / std(population), std clamped at 1e-6",
        "kl_estimator_in_loss": "k3: exp(ref - new) - (ref - new) - 1, mean over valid tokens",
        "prompt_rows": [order[i] for i in range(U * P)], "dropout": "disabled",
        "old_logprobs": "new.detach() when policy_epochs == 1 (ratio = 1), else a no-grad pass",
    }

    policy = load_policy_from(cfg, "midpoint", trainable=True)
    rm, rm_tok = load_reward_model(cfg)
    params = trainable_parameters(policy)
    optimizer = AdamW(params, lr=float(cfg["learning_rate"]))  # as in the released scaffold
    fp16 = torch.cuda.is_available() and resolve_dtype(cfg.get("dtype", "float16")) == torch.float16
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0**12, growth_interval=100, enabled=fp16)

    payload = None if force else load_latest_slot(resume_dir)
    if payload is not None:
        state = payload["state"]
        if state["run_meta"] != run_meta:
            raise SystemExit(f"[grpo:{run_name}] checkpoint in {resume_dir} has a different configuration; use --force.")
        load_trainable_state(policy, payload["trainable"])
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        set_rng_state(payload["rng"])
        start_u, elapsed_prev, peak_prev = int(state["update"]), float(state["elapsed_s"]), float(state["peak_vram_gb"])
        truncate_log(log_path, start_u)
        truncate_log(comp_path, start_u)
        print(f"[grpo:{run_name}] RESUMING from {resume_dir} after update {start_u}")
    else:
        start_u, elapsed_prev, peak_prev = 0, 0.0, 0.0
        for p in (log_path, comp_path):
            if p.exists():
                p.unlink()
        save_json(run_res / "run_config.json", run_meta)
        save_json(run_res / "prompts_used.json", {
            "rows": run_meta["prompt_rows"], "prompt_ids": [rows[i].get("prompt_id") for i in run_meta["prompt_rows"]]})

    print(f"[grpo:{run_name}] updates={U} K={K} loss_type={loss_type} beta={beta} eps={eps} epochs={epochs} "
          f"start={start_u} fp16_scaler={fp16}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    dev = next(policy.parameters()).device
    t_start = time.time()
    steps_this_session = 0

    for u in range(start_u, U):
        t_u = time.time()
        batch_rows = [rows[i] for i in order[u * P : (u + 1) * P]]
        msgs = [prompt_messages(r) for r in batch_rows for _ in range(K)]
        group_ids = torch.arange(P, device=dev).repeat_interleave(K)

        torch.manual_seed(seed + 1000 * u)
        gen = batch_generate(policy, tok, msgs, max_prompt_length=int(cfg["max_prompt_length"]), max_new_tokens=max_len,
                             temperature=float(gen_cfg.get("temperature", 0.7)), top_p=float(gen_cfg.get("top_p", 0.9)),
                             do_sample=bool(gen_cfg.get("do_sample", True)))
        seq, attn, rids = gen["sequences"].clone(), gen["attention_mask"].clone(), gen["response_ids"].clone()
        mask = gen["response_mask"].clone().float()
        truncated = list(gen["truncated"])
        train_mask = mask_truncated_sequences(mask, truncated) if mask_trunc else mask
        n_seq = seq.shape[0]
        n_tok = float(train_mask.sum())

        with torch.no_grad():
            rewards = score_reward_pairs(rm, rm_tok, msgs, gen["responses"]).float().to(dev)
        adv = group_relative_advantages(rewards, group_ids)
        gstats = group_statistics(rewards, group_ids)

        policy.eval()
        with torch.no_grad(), reference_mode(policy):
            ref_lp, _ = response_logps_entropy(policy, seq, attn, rids, entropy=False)
        old_lp = None
        if epochs > 1:
            with torch.no_grad():
                old_lp, _ = response_logps_entropy(policy, seq, attn, rids, entropy=False)
        policy.train()

        seq_norms = (_seq_policy_grad_norms(policy, params, seq, attn, rids, adv, train_mask, n_seq, loss_type,
                                            max_len, scaler) if seq_grad_norms else None)

        ep_stats, ent_full, cur_lp = [], torch.zeros_like(mask), torch.zeros_like(mask)
        skipped_all_masked = n_tok == 0.0
        for ep in range(epochs if not skipped_all_masked else 0):
            optimizer.zero_grad(set_to_none=True)
            pol_part = kl_part = clipped = ratio_sum = 0.0
            for s in range(0, n_seq, train_micro_batch):
                sl = slice(s, s + train_micro_batch)
                new_lp, ent = response_logps_with_entropy(policy, seq[sl], attn[sl], rids[sl])
                if ep == 0:
                    ent_full[sl], cur_lp[sl] = ent, new_lp.detach()
                old = new_lp.detach() if old_lp is None else old_lp[sl]
                loss_mb, d = grpo_microbatch_loss(new_lp, old, adv[sl], train_mask[sl], ref_lp[sl], eps, beta,
                                                  loss_type, max_len, n_seq, n_tok)
                scaler.scale(loss_mb).backward()
                pol_part += float(d["policy_part"])
                kl_part += float(d["kl_part"])
                clipped += float(d["clipped_tokens"])
                ratio_sum += float(d["ratio_sum"])
                del new_lp, ent, loss_mb
            scaler.unscale_(optimizer)
            gn = float(torch.nn.utils.clip_grad_norm_(params, max_gn))
            before = scaler.get_scale() if fp16 else 1.0
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ep_stats.append({"policy_term": pol_part, "kl_penalty": kl_part, "loss": pol_part + beta * kl_part,
                             "clip_fraction": clipped / max(n_tok, 1.0), "ratio_mean": ratio_sum / max(n_tok, 1.0),
                             "grad_norm": gn, "skipped_nonfinite": bool(fp16 and scaler.get_scale() < before)})

        policy.eval()
        with torch.no_grad():
            post_lp, _ = response_logps_entropy(policy, seq, attn, rids, entropy=False)
        if skipped_all_masked:
            cur_lp = post_lp.clone()
        log_r = (post_lp - cur_lp) * train_mask
        shift_k3 = float(masked_mean(torch.exp(log_r) - 1.0 - log_r, train_mask)) if n_tok > 0 else 0.0

        lens = mask.sum(-1)
        valid_mask = mask  # every generated token (for length / entropy / KL logging)
        stds = [g["std"] for g in gstats]
        steps_this_session += 1
        elapsed = elapsed_prev + time.time() - t_start
        peak = max(peak_prev, gpu_peak_gb())
        rec = {
            "update": u + 1,
            "prompt_rows": order[u * P : (u + 1) * P],
            "reward_mean": float(rewards.mean()),
            "reward_mean_untruncated": float(rewards[~torch.tensor(truncated, device=dev)].mean()) if not all(truncated) else None,
            "group_reward_std_mean": float(np.mean(stds)),
            "uninformative_group_fraction": float(np.mean([not g["informative"] for g in gstats])),
            "kl_token_mean": float(sampled_kl(cur_lp if not skipped_all_masked else post_lp, ref_lp, valid_mask)),
            "kl_k3_in_loss": ep_stats[0]["kl_penalty"] if ep_stats else None,
            "policy_loss": ep_stats[0]["loss"] if ep_stats else None,
            "policy_term": ep_stats[0]["policy_term"] if ep_stats else None,
            "grad_norm": float(np.mean([e["grad_norm"] for e in ep_stats])) if ep_stats else None,
            "entropy_mean": float(masked_mean(ent_full, valid_mask)) if ep_stats else None,
            "sample_entropy": float(sample_entropy(cur_lp, valid_mask)) if ep_stats else None,
            "clip_fraction": float(np.mean([e["clip_fraction"] for e in ep_stats])) if ep_stats else None,
            "response_length": float(lens.mean()),
            "response_length_kept": float(lens[train_mask.sum(-1) > 0].mean()) if n_tok > 0 else None,
            "truncated_fraction": float(np.mean(truncated)),
            "tokens_in_loss": int(n_tok),
            "generated_tokens": int(lens.sum()),
            "policy_shift_kl_k3": shift_k3,
            "skipped_update_all_masked": skipped_all_masked,
            "skipped_nonfinite": int(sum(e["skipped_nonfinite"] for e in ep_stats)),
            "epochs": ep_stats,
            "update_seconds": time.time() - t_u,
            "elapsed_s": elapsed,
            "peak_vram_gb": peak,
        }
        append_jsonl(log_path, rec)
        for k in range(n_seq):
            T_k = float(lens[k])
            kept = float(train_mask[k].sum()) > 0
            a = float(adv[k])
            w = (abs(a) / max(T_k, 1.0) if loss_type == "grpo" else abs(a) / max_len) if kept else 0.0
            append_jsonl(comp_path, {
                "update": u + 1, "group": int(group_ids[k]), "k": k, "prompt_row": order[u * P + int(group_ids[k])],
                "length": int(T_k), "reward": float(rewards[k]), "advantage": a, "truncated": bool(truncated[k]),
                "in_loss": kept, "per_token_weight": w / n_seq, "sequence_weight": (w * T_k) / n_seq,
                "seq_policy_grad_norm": (seq_norms[k] if seq_norms is not None else None),
                "response": gen["responses"][k][:2000],
            })
        print(f"[grpo:{run_name}] update {u + 1}/{U} reward={rec['reward_mean']:+.3f} groupstd={rec['group_reward_std_mean']:.3f} "
              f"uninf={rec['uninformative_group_fraction']:.2f} KL={rec['kl_token_mean']:.4f} len={rec['response_length']:.0f} "
              f"trunc={rec['truncated_fraction']:.2f} gn={rec['grad_norm'] if rec['grad_norm'] is not None else float('nan'):.3f} "
              f"| {rec['update_seconds']:.0f}s, total {elapsed / 60:.1f} min", flush=True)
        del gen, seq, attn, rids, ref_lp, post_lp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        stop_now = max_updates_this_session is not None and steps_this_session >= max_updates_this_session
        if u + 1 < U and ((u + 1) % save_every == 0 or stop_now):
            save_slot_checkpoint(resume_dir, {
                "state": {"run_meta": run_meta, "update": u + 1, "elapsed_s": elapsed, "peak_vram_gb": peak, "total_updates": U},
                "trainable": trainable_state(policy), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "rng": rng_state()}, u + 1)
            print(f"[grpo:{run_name}] checkpoint saved after update {u + 1}", flush=True)
        if stop_now and u + 1 < U:
            print(f"[grpo:{run_name}] stopping after {steps_this_session} updates this session (resumable).")
            del policy, rm
            clear_gpu()
            return None

    elapsed = elapsed_prev + time.time() - t_start
    peak = max(peak_prev, gpu_peak_gb())
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(final_dir))
    logs = read_jsonl_abs(log_path)
    summary = {
        "run_meta": run_meta, "adapter_dir": str(final_dir), "updates_done": len(logs),
        "wall_clock_s": elapsed, "wall_clock_min": elapsed / 60, "peak_vram_gb": peak,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "generated_tokens_total": int(sum(r["generated_tokens"] for r in logs)),
        "tokens_in_loss_total": int(sum(r["tokens_in_loss"] for r in logs)),
        "uninformative_group_fraction_mean": float(np.mean([r["uninformative_group_fraction"] for r in logs])),
        "truncated_fraction_mean": float(np.mean([r["truncated_fraction"] for r in logs])),
        "skipped_updates_all_masked": int(sum(r["skipped_update_all_masked"] for r in logs)),
        "policy_shift_kl_k3_mean": float(np.mean([r["policy_shift_kl_k3"] for r in logs])),
        "policy_shift_kl_k3_max": float(np.max([r["policy_shift_kl_k3"] for r in logs])),
    }
    save_json(run_res / "train_summary.json", summary)
    save_json(done_path, summary)
    clear_resume(resume_dir)
    print(f"[grpo:{run_name}] DONE in {elapsed / 60:.1f} min, peak VRAM {peak:.2f} GB -> {final_dir}")
    del policy, rm
    clear_gpu()
    return final_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--work-dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--save-every", type=int, default=2)
    ap.add_argument("--max-updates-this-session", type=int)
    ap.add_argument("--train-micro-batch", type=int, default=2, help="completions per forward/backward (exact accumulation)")
    ap.add_argument("--seq-grad-norms", action="store_true", help="log the per-completion policy-gradient norm")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name, args.work_dir, args.results_dir,
             args.save_every, args.force, args.max_updates_this_session, args.train_micro_batch, args.seq_grad_norms)


if __name__ == "__main__":
    main()
