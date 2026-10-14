"""Task 3 - validate the GRPO objective helpers and the model plumbing BEFORE training.

python -m task3_grpo.validate --config configs/grpo.yaml [--math-only]
"""
from __future__ import annotations

import argparse

import torch

from common.data import load_yaml, prompt_messages
from common.generation import batch_generate, response_token_logprobs
from common.models import clear_gpu, load_tokenizer, trainable_parameters
from task3_grpo.grpo import group_relative_advantages, grpo_microbatch_loss, grpo_policy_loss, mask_truncated_sequences
from task3_grpo.utils import (
    load_policy_from, response_logps_entropy, response_logps_with_entropy, save_json, task_dirs, train_prompt_order,
)


def check(rep, name, ok, detail):
    rep.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def math_checks(rep):
    # advantages are computed within each prompt group
    r = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    gid = torch.arange(3).repeat_interleave(4)
    a = group_relative_advantages(r, gid)
    manual = torch.zeros_like(r)
    for g in range(3):
        x = r[gid == g]
        manual[gid == g] = (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-6)
    check(rep, "advantages normalised within each group", torch.allclose(a, manual), f"{[round(v, 3) for v in a.tolist()]}")
    check(rep, "identical-reward group gets zero advantage", bool((a[4:8] == 0).all()), "group 2 rewards all 10 -> A = 0")
    for g in range(3):
        if g == 1:
            continue
        x = a[gid == g]
        assert abs(float(x.mean())) < 1e-12
    check(rep, "informative groups: zero mean, unit (population) std", all(
        abs(float(a[gid == g].mean())) < 1e-9 and abs(float(a[gid == g].std(unbiased=False)) - 1) < 1e-9 for g in (0, 2)), "ok")
    starter = (r - r.mean()) / r.std(unbiased=False)
    check(rep, "starter (global batch normalisation) differs", not torch.allclose(starter, a),
          f"starter gives group 2 (all 10s) advantages {starter[4]:+.2f} instead of 0")

    # micro-batch accumulation == full-batch course loss (both normalisations, with masked sequences)
    g = torch.Generator().manual_seed(0)
    B, T = 6, 9
    new = (torch.randn(B, T, generator=g, dtype=torch.float64) * 0.3 - 2)
    old = new + torch.randn(B, T, generator=g, dtype=torch.float64) * 0.2
    ref = new + torch.randn(B, T, generator=g, dtype=torch.float64) * 0.3
    adv = torch.randn(B, generator=g, dtype=torch.float64)
    mask = torch.ones(B, T, dtype=torch.float64)
    mask[1, 4:] = 0
    mask[3, 2:] = 0
    mask = mask_truncated_sequences(mask, [False, False, True, False, False, False])
    for lt in ("grpo", "dr_grpo"):
        full, _ = grpo_policy_loss(new, old, adv, mask, ref, 0.2, 0.1, lt, max_completion_length=T)
        acc = sum(grpo_microbatch_loss(new[s:s + 2], old[s:s + 2], adv[s:s + 2], mask[s:s + 2], ref[s:s + 2], 0.2, 0.1, lt,
                                       T, B, float(mask.sum()))[0] for s in range(0, B, 2))
        check(rep, f"micro-batch accumulation == grpo_policy_loss ({lt})", torch.allclose(full, acc, atol=1e-12),
              f"full={full.item():.10f} accumulated={acc.item():.10f}")
    check(rep, "truncated completion is fully masked", float(mask[2].sum()) == 0.0, "mask row 2 all zero")

    # per-token gradient weight: GRPO ~ A/T_k, Dr.GRPO ~ A/L (ratio = 1, beta = 0)
    lp = torch.zeros(2, 8, dtype=torch.float64, requires_grad=True)
    m2 = torch.zeros(2, 8, dtype=torch.float64)
    m2[0, :2] = 1
    m2[1, :8] = 1
    a2 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    for lt, expect in (("grpo", (1 / 2 / 2, 1 / 8 / 2)), ("dr_grpo", (1 / 8 / 2, 1 / 8 / 2))):
        lp.grad = None
        loss, _ = grpo_policy_loss(lp, lp.detach(), a2, m2, lp.detach(), 0.2, 0.0, lt, max_completion_length=8)
        loss.backward()
        got = (-lp.grad[0, 0].item(), -lp.grad[1, 0].item())
        check(rep, f"per-token gradient weight ({lt})", abs(got[0] - expect[0]) < 1e-12 and abs(got[1] - expect[1]) < 1e-12,
              f"short (T=2) {got[0]:.4f}, long (T=8) {got[1]:.4f}")
    lr = torch.linspace(-3, 3, 50)
    k3 = torch.exp(lr) - lr - 1
    check(rep, "k3 KL estimator is non-negative", bool((k3 >= 0).all()), "exp(x) - x - 1 >= 0")


def model_checks(rep, cfg, dirs):
    tok = load_tokenizer(cfg["base_model"])
    rows, order = train_prompt_order(cfg, tok, dirs)
    policy = load_policy_from(cfg, "midpoint", trainable=True)
    msgs = [prompt_messages(rows[order[0]])] * 2
    torch.manual_seed(0)
    gen = batch_generate(policy, tok, msgs, max_prompt_length=int(cfg["max_prompt_length"]), max_new_tokens=32, do_sample=True)
    seq, attn, rids, mask = (gen[k].clone() for k in ("sequences", "attention_mask", "response_ids", "response_mask"))
    policy.eval()
    with torch.no_grad():
        a, _ = response_logps_entropy(policy, seq, attn, rids)
        b, _ = response_token_logprobs(policy, seq, attn, gen["prompt_width"], rids)
    d = ((a - b).abs() * mask).max().item()
    check(rep, "response log-prob helper == common.generation.response_token_logprobs", d < 0.1, f"max per-token |diff| = {d:.2e}")
    policy.train()
    new, ent = response_logps_with_entropy(policy, seq, attn, rids)
    rmax = float(((new.detach() - a).abs() * mask).max())
    check(rep, "train-mode forward == eval-mode forward (dropout disabled, ratio = 1)", rmax < 1e-2, f"max |log ratio| = {rmax:.2e}")
    check(rep, "entropy finite and non-negative", bool(torch.isfinite(ent).all() and (ent >= -1e-4).all()),
          f"mean {float((ent * mask).sum() / mask.sum()):.3f} nats")
    dts = sorted({str(p.dtype) for p in trainable_parameters(policy)})
    check(rep, "trainable LoRA parameters are fp32", dts == ["torch.float32"], str(dts))
    del policy
    clear_gpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--work-dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--math-only", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    dirs = task_dirs(cfg, args.work_dir, args.results_dir)
    rep = []
    math_checks(rep)
    if not args.math_only:
        model_checks(rep, cfg, dirs)
    save_json(dirs["results"] / "validation_report.json", rep)
    n_fail = sum(not r["ok"] for r in rep)
    print(f"\n{len(rep) - n_fail}/{len(rep)} checks passed")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
