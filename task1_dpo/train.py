"""Task 1 - DPO training (resumable).

python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.train --config configs/dpo.yaml --run-name beta_0.03 --beta 0.03 --max-examples 600

Every --save-every optimizer steps the LoRA adapter, optimizer, GradScaler and RNG
states are written to <work-dir>/resume/<run>. Re-running the same command after a
crash/disconnect continues from the last checkpoint; a finished run (DONE.json) is skipped.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import set_seed
from common.models import clear_gpu, count_parameters, load_policy, load_tokenizer, resolve_dtype, trainable_parameters
from task1_dpo.dpo import dpo_loss
from task1_dpo.utils import (
    append_jsonl,
    clear_resume,
    collate_pairs,
    encode_preference_rows,
    ensure_fp32_trainable,
    ensure_ref_cache,
    file_fingerprint,
    find_resume_checkpoint,
    gpu_peak_gb,
    load_json,
    model_device,
    read_jsonl_abs,
    ref_cache_meta,
    rel_to_repo,
    save_json,
    save_resume_checkpoint,
    sequence_logps,
    set_rng_state,
    task_dirs,
    truncate_log,
)


def micro_batch_plan(n_items: int, batch_size: int, epochs: int, seed: int):
    """Fixed, seeded data order (replaces DataLoader(shuffle=True) so that the exact
    order can be saved and reproduced after a resume)."""
    batches, orders = [], []
    for e in range(epochs):
        order = np.random.RandomState(seed + e).permutation(n_items).tolist()
        orders.append(order)
        for s in range(0, n_items, batch_size):
            batches.append(order[s : s + batch_size])
    return batches, orders


def _weighted_mean(records, key):
    tot = sum(r["n"] for r in records)
    return float(sum(r[key] * r["n"] for r in records) / max(tot, 1))


def run_training(
    config_path: str,
    run_name: str,
    dataset_path: str | None = None,
    output_path: str | None = None,
    beta: float | None = None,
    max_examples: int | None = None,
    work_dir: str | None = None,
    results_dir: str | None = None,
    save_every: int = 5,
    epochs: int | None = None,
    force: bool = False,
    max_steps_this_session: int | None = None,
):
    cfg = load_yaml(config_path)
    seed = int(cfg["seed"])
    dirs = task_dirs(cfg, work_dir, results_dir)
    beta = float(cfg["beta"] if beta is None else beta)
    epochs = int(cfg.get("epochs", 1) if epochs is None else epochs)
    data_path = dataset_path or cfg["paths"]["dpo_standard_train"]
    bs = int(cfg["batch_size"])
    accum = int(cfg["grad_accum_steps"])
    lr = float(cfg["learning_rate"])
    max_len = int(cfg["max_sequence_length"])

    final_dir = repo_path(output_path) if output_path else dirs["work"] / run_name
    resume_dir = dirs["resume"] / run_name
    run_res = dirs["train_results"] / run_name
    log_path = run_res / "train_log.jsonl"
    done_path = final_dir / "DONE.json"

    if done_path.exists() and not force:
        print(f"[train:{run_name}] already finished -> {final_dir} (use --force to retrain)")
        return final_dir
    if force:
        clear_resume(resume_dir)
        if done_path.exists():
            done_path.unlink()
        if log_path.exists():
            log_path.unlink()

    set_seed(seed)
    rows = read_jsonl(data_path)
    n_rows_file = len(rows)
    if max_examples is not None:
        rows = rows[: int(max_examples)]
    tokenizer = load_tokenizer(cfg["base_model"])
    items, skipped = encode_preference_rows(tokenizer, rows, max_len)
    if not items:
        raise SystemExit("No usable training pairs.")

    run_meta = {
        "run_name": run_name,
        "data_file": rel_to_repo(data_path),
        "data_sha": file_fingerprint(data_path),
        "rows_in_file": n_rows_file,
        "rows_selected": len(rows),
        "pairs_used": len(items),
        "rows_skipped": len(skipped),
        "beta": beta,
        "learning_rate": lr,
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
        "batch_size": bs,
        "grad_accum_steps": accum,
        "effective_batch": bs * accum,
        "epochs": epochs,
        "max_sequence_length": max_len,
        "max_grad_norm": float(cfg.get("max_grad_norm", 1.0)),
        "seed": seed,
        "base_model": str(cfg["base_model"]),
        "dtype": str(cfg.get("dtype", "float16")),
        "lora": cfg["lora"],
        "lr_schedule": "constant (no warmup), as in the released starter",
    }

    ckpt = find_resume_checkpoint(resume_dir)
    state = None
    if ckpt is not None:
        state = load_json(ckpt / "state.json")
        if state["run_meta"] != run_meta:
            raise SystemExit(
                f"[train:{run_name}] resume checkpoint at {ckpt} was created with a different "
                "configuration. Re-run with --force to start this run from scratch."
            )
        print(f"[train:{run_name}] RESUMING from {ckpt} at optimizer step {state['opt_step']}")
        model = load_policy(cfg, adapter_path=str(ckpt / "adapter"), trainable=True)
    else:
        set_seed(seed)  # identical fresh LoRA initialisation for every run
        model = load_policy(cfg, trainable=True, fresh_lora=True)
    n_cast = ensure_fp32_trainable(model)
    total_p, train_p = count_parameters(model)
    print(f"[train:{run_name}] params total={total_p:,} trainable={train_p:,} (cast {n_cast} to fp32)")

    ref = ensure_ref_cache(
        model, tokenizer, items, repo_path(dirs["cache"]) / f"ref_{Path(data_path).stem}.json",
        meta=ref_cache_meta(cfg, data_path),
    )

    params = trainable_parameters(model)
    optimizer = AdamW(params, lr=lr, weight_decay=float(cfg.get("weight_decay", 0.0)))
    use_scaler = torch.cuda.is_available() and resolve_dtype(cfg.get("dtype", "float16")) == torch.float16
    # fp16 base weights + fp32 LoRA weights: dynamic loss scaling prevents fp16 gradient underflow.
    # A moderate initial scale avoids wasting the first optimizer steps on overflow skips.
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0**12, growth_interval=100, enabled=use_scaler)

    batches, orders = micro_batch_plan(len(items), bs, epochs, seed)
    total_micro = len(batches)
    total_opt = math.ceil(total_micro / accum)

    if state is not None:
        ts = torch.load(ckpt / "trainer_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(ts["optimizer"])
        scaler.load_state_dict(ts["scaler"])
        set_rng_state(ts["rng"])
        micro_start, opt_step = int(state["micro_step"]), int(state["opt_step"])
        elapsed_prev, peak_prev = float(state["elapsed_s"]), float(state.get("peak_vram_gb", 0.0))
        truncate_log(log_path, opt_step)
    else:
        micro_start, opt_step, elapsed_prev, peak_prev = 0, 0, 0.0, 0.0
        if log_path.exists():
            log_path.unlink()
        save_json(run_res / "run_config.json", run_meta)
        save_json(run_res / "skipped_rows.json", skipped)
        save_json(
            run_res / "data_order.json",
            {
                "data_file": rel_to_repo(data_path),
                "note": "orders[e] lists positions into 'rows' (file row indices) in the order used in epoch e",
                "rows": [it["row"] for it in items],
                "prompt_ids": [it["prompt_id"] for it in items],
                "orders": [[items[j]["row"] for j in o] for o in orders],
            },
        )
        n_trunc = sum(it["chosen_truncated"] or it["rejected_truncated"] for it in items)
        print(f"[train:{run_name}] pairs={len(items)} skipped={len(skipped)} pairs_with_truncated_response={n_trunc}")

    print(
        f"[train:{run_name}] beta={beta} lr={lr} micro_batches={total_micro} opt_steps={total_opt} "
        f"start_micro={micro_start} fp16_grad_scaler={use_scaler}",
        flush=True,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    device = model_device(model)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    t_start = time.time()
    window, steps_this_session = [], 0
    max_gn = float(cfg.get("max_grad_norm", 1.0))

    for i in range(micro_start, total_micro):
        g_start = (i // accum) * accum
        g_size = min(accum, total_micro - g_start)
        chunk = [items[j] for j in batches[i]]
        n = len(chunk)
        batch = collate_pairs(tokenizer, chunk, device)
        logps, _, mask = sequence_logps(model, batch)
        pc, pr = logps[:n], logps[n:]
        rc = torch.tensor([ref["chosen"][it["key"]] for it in chunk], device=device, dtype=torch.float32)
        rr = torch.tensor([ref["rejected"][it["key"]] for it in chunk], device=device, dtype=torch.float32)
        loss, diag = dpo_loss(pc, pr, rc, rr, beta)
        scaler.scale(loss / g_size).backward()
        window.append(
            {
                "n": n,
                "loss": float(loss.detach()),
                "acc": float(diag["preference_accuracy"]),
                "margin": float(diag["margin_mean"]),
                "chosen_logratio": float(diag["chosen_logratio_mean"]),
                "rejected_logratio": float(diag["rejected_logratio_mean"]),
                "logit": float(diag["logit_mean"]),
                "resp_tokens": float(mask.sum().item()) / (2 * n),
            }
        )

        if i == g_start + g_size - 1:  # end of an accumulation group -> optimizer step
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_gn))
            scale_before = scaler.get_scale() if use_scaler else 1.0
            scaler.step(optimizer)
            scaler.update()
            skipped_step = bool(use_scaler and scaler.get_scale() < scale_before)
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1
            steps_this_session += 1
            elapsed = elapsed_prev + time.time() - t_start
            peak = max(peak_prev, gpu_peak_gb())
            rec = {
                "opt_step": opt_step,
                "micro_step": i + 1,
                "pairs_seen": sum(len(b) for b in batches[: i + 1]),
                "epoch": i // math.ceil(len(items) / bs),
                "loss": _weighted_mean(window, "loss"),
                "preference_accuracy": _weighted_mean(window, "acc"),
                "margin_mean": _weighted_mean(window, "margin"),
                "chosen_logratio_mean": _weighted_mean(window, "chosen_logratio"),
                "rejected_logratio_mean": _weighted_mean(window, "rejected_logratio"),
                "logit_mean": _weighted_mean(window, "logit"),
                "mean_response_tokens": _weighted_mean(window, "resp_tokens"),
                "grad_norm": grad_norm,
                "skipped_step_nonfinite_grad": skipped_step,
                "loss_scale": float(scaler.get_scale()) if use_scaler else 1.0,
                "lr": lr,
                "elapsed_s": elapsed,
                "peak_vram_gb": peak,
            }
            append_jsonl(log_path, rec)
            rate = elapsed / max(opt_step, 1)
            print(
                f"[train:{run_name}] step {opt_step}/{total_opt} loss={rec['loss']:.4f} "
                f"acc={rec['preference_accuracy']:.3f} margin={rec['margin_mean']:+.3f} "
                f"gn={grad_norm:.3f} | {elapsed / 60:.1f} min, ETA {(total_opt - opt_step) * rate / 60:.1f} min",
                flush=True,
            )
            window = []
            stop_now = max_steps_this_session is not None and steps_this_session >= max_steps_this_session
            if opt_step < total_opt and (opt_step % save_every == 0 or stop_now):
                save_resume_checkpoint(
                    resume_dir, model, optimizer, scaler,
                    {"run_meta": run_meta, "opt_step": opt_step, "micro_step": i + 1,
                     "elapsed_s": elapsed, "peak_vram_gb": peak, "total_opt_steps": total_opt},
                )
            if stop_now and opt_step < total_opt:
                print(f"[train:{run_name}] stopping after {steps_this_session} steps this session (resumable).")
                del model, optimizer
                clear_gpu()
                return None

    elapsed = elapsed_prev + time.time() - t_start
    peak = max(peak_prev, gpu_peak_gb())
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    logs = read_jsonl_abs(log_path)
    last = logs[-10:]
    summary = {
        "run_meta": run_meta,
        "adapter_dir": str(final_dir),
        "opt_steps": opt_step,
        "micro_steps": total_micro,
        "pairs_seen": sum(len(b) for b in batches),
        "wall_clock_s": elapsed,
        "wall_clock_min": elapsed / 60,
        "peak_vram_gb": peak,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "skipped_nonfinite_steps": int(sum(r["skipped_step_nonfinite_grad"] for r in logs)),
        "final_window_last10_steps": {
            k: float(np.mean([r[k] for r in last])) for k in
            ["loss", "preference_accuracy", "margin_mean", "chosen_logratio_mean", "rejected_logratio_mean"]
        } if last else {},
    }
    save_json(run_res / "train_summary.json", summary)
    save_json(done_path, summary)
    clear_resume(resume_dir)
    print(f"[train:{run_name}] DONE in {elapsed / 60:.1f} min, peak VRAM {peak:.2f} GB -> {final_dir}")
    del model, optimizer
    clear_gpu()
    return final_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--work-dir", help="default: outputs/task1_dpo")
    ap.add_argument("--results-dir", help="default: results/task1_dpo")
    ap.add_argument("--save-every", type=int, default=5, help="optimizer steps between resume checkpoints")
    ap.add_argument("--max-steps-this-session", type=int, help="stop (resumably) after N optimizer steps")
    ap.add_argument("--force", action="store_true", help="discard previous checkpoints/results of this run")
    args = ap.parse_args()
    run_training(
        args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples,
        args.work_dir, args.results_dir, args.save_every, args.epochs, args.force, args.max_steps_this_session,
    )


if __name__ == "__main__":
    main()
