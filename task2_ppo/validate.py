"""Task 2 - validate the PPO objective helpers and the model plumbing BEFORE training.

python -m task2_ppo.validate --config configs/ppo.yaml [--math-only]
"""
from __future__ import annotations

import argparse

import torch

from common.data import load_yaml, prompt_messages
from common.generation import batch_generate, response_token_logprobs
from common.models import clear_gpu, load_tokenizer, load_value_model, token_values
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards
from task2_ppo.utils import (
    critic_values, load_critic, load_policy_from, response_logps, response_logps_entropy, save_json, task_dirs,
    train_prompt_order,
)


def check(rep, name, ok, detail):
    rep.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def math_checks(rep):
    g = torch.Generator().manual_seed(0)
    B, T = 3, 7
    new = torch.randn(B, T, generator=g, dtype=torch.float64) * 0.4
    old = torch.randn(B, T, generator=g, dtype=torch.float64) * 0.4
    A = torch.randn(B, T, generator=g, dtype=torch.float64)
    mask = torch.ones(B, T, dtype=torch.float64)
    mask[1, 5:] = 0
    mask[2, 2:] = 0
    for eps in (0.05, 0.2, 0.5):
        loss, ratio, cf = ppo_policy_loss(new, old, A, mask, eps)
        r = torch.exp(new - old)
        obj = torch.min(r * A, torch.clamp(r, 1 - eps, 1 + eps) * A)
        manual = -(obj * mask).sum() / mask.sum()
        manual_cf = (((r < 1 - eps) | (r > 1 + eps)).double() * mask).sum() / mask.sum()
        check(rep, f"clipped surrogate matches manual (eps={eps})", torch.allclose(loss, manual, atol=1e-12),
              f"impl={loss.item():.8f} manual={manual.item():.8f}")
        check(rep, f"clip fraction = ratio outside band before clipping (eps={eps})", abs(cf.item() - manual_cf.item()) < 1e-6, f"{cf.item():.4f}")
    # starter's torch.maximum is the optimistic bound: differs whenever the clip should bind
    r = torch.tensor([[1.5, 0.5]], dtype=torch.float64)
    Ad = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    m1 = torch.ones(1, 2, dtype=torch.float64)
    lmin, _, _ = ppo_policy_loss(torch.log(r), torch.zeros_like(r), Ad, m1, 0.2)
    lmax = -(torch.maximum(r * Ad, r.clamp(0.8, 1.2) * Ad)).mean()
    check(rep, "min (pessimistic) vs starter max (optimistic)", abs(lmin.item() - (-(1.2 - 0.8) / 2)) < 1e-12 and lmax.item() < lmin.item(),
          f"corrected loss={lmin.item():+.3f}, starter loss={lmax.item():+.3f} (starter rewards ratios beyond the band)")
    # gradient vanishes exactly where the clip binds
    lnew = torch.log(torch.tensor([[1.5, 1.5, 0.5, 0.5]], dtype=torch.float64)).requires_grad_(True)
    Ab = torch.tensor([[1.0, -1.0, -1.0, 1.0]], dtype=torch.float64)
    loss, _, _ = ppo_policy_loss(lnew, torch.zeros_like(lnew), Ab, torch.ones_like(Ab), 0.2)
    loss.backward()
    gr = lnew.grad[0].tolist()
    check(rep, "zero gradient iff clip binds", gr[0] == 0 and gr[2] == 0 and gr[1] != 0 and gr[3] != 0,
          f"grads={['%.3f' % x for x in gr]} for (A>0,r>1+e) (A<0,r>1+e) (A<0,r<1-e) (A>0,r<1-e)")
    # GAE vs brute-force lambda-return definition
    rew = torch.randn(B, T, generator=g, dtype=torch.float64)
    val = torch.randn(B, T, generator=g, dtype=torch.float64)
    gamma, lam = 0.97, 0.9
    adv, ret = compute_gae(rew * mask, val * mask, mask, gamma, lam)
    bf = torch.zeros(B, T, dtype=torch.float64)
    for b in range(B):
        L = int(mask[b].sum())
        for t in range(L):
            s = 0.0
            for k in range(t, L):
                v_next = val[b, k + 1] if k + 1 < L else 0.0
                delta = rew[b, k] + gamma * v_next - val[b, k]
                s += (gamma * lam) ** (k - t) * delta
            bf[b, t] = s
    check(rep, "GAE matches brute-force sum of (gamma*lambda)^k deltas", torch.allclose(adv, bf, atol=1e-10),
          f"max |diff| = {(adv - bf).abs().max().item():.2e}")
    check(rep, "returns = advantages + values on valid tokens", torch.allclose((ret - adv - val) * mask, torch.zeros_like(ret)), "ok")
    # shaped rewards: -beta*(logpi - logref) everywhere + task reward at the LAST valid token
    lp, rf = torch.randn(B, T, dtype=torch.float64), torch.randn(B, T, dtype=torch.float64)
    sr = shaped_rewards(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), lp, rf, mask, 0.1)
    exp_ = -0.1 * (lp - rf) * mask
    for b, L in enumerate(mask.sum(-1).long().tolist()):
        exp_[b, L - 1] += [1.0, 2.0, 3.0][b]
    check(rep, "KL-shaped rewards place the task reward on the final valid token", torch.allclose(sr, exp_), "ok")
    an = normalize_advantages(adv, mask)
    v = an[mask.bool()]
    check(rep, "advantage whitening: zero mean / unit std over valid tokens", abs(v.mean().item()) < 1e-9 and abs(v.std(unbiased=False).item() - 1) < 1e-9, "ok")


def model_checks(rep, cfg, dirs):
    tok = load_tokenizer(cfg["base_model"])
    rows, order = train_prompt_order(cfg, tok, dirs)
    policy = load_policy_from(cfg, "midpoint", trainable=True)
    msgs = [prompt_messages(rows[order[0]])]
    torch.manual_seed(0)
    gen = batch_generate(policy, tok, msgs, max_prompt_length=int(cfg["max_prompt_length"]), max_new_tokens=48, do_sample=True)
    seq, attn, rids, mask = (gen[k].clone() for k in ("sequences", "attention_mask", "response_ids", "response_mask"))
    pw = gen["prompt_width"]
    policy.eval()
    with torch.no_grad():
        a, ent = response_logps_entropy(policy, seq, attn, rids)
        b, _ = response_token_logprobs(policy, seq, attn, pw, rids)
    d = ((a - b).abs() * mask).max().item()
    check(rep, "response log-prob helper == common.generation.response_token_logprobs", d < 0.1, f"max per-token |diff| = {d:.2e}")
    check(rep, "entropy is non-negative and finite", bool(torch.isfinite(ent).all() and (ent >= -1e-4).all()),
          f"mean entropy {float((ent * mask).sum() / mask.sum()):.3f} nats")
    policy.train()
    new = response_logps(policy, seq, attn, rids)
    rmax = float(((new.detach() - a).abs() * mask).max())
    check(rep, "first-epoch ratio == 1 (dropout disabled, train vs eval forward identical)", rmax < 1e-2,
          f"max |log ratio| = {rmax:.2e}")
    del policy
    clear_gpu()

    critic = load_critic(cfg)
    critic.eval()
    ref_critic = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode="frozen")
    with torch.no_grad():
        v_ours = critic_values(critic, seq, attn, pw, rids.shape[1])
        v_course = token_values(ref_critic, seq, attn)[:, pw - 1 : pw - 1 + rids.shape[1]].float()
    d2 = ((v_ours - v_course).abs() * mask).max().item()
    check(rep, "critic values (fp32 head, fresh zero LoRA) == course token_values of the supplied critic", d2 < 0.05,
          f"max |diff| = {d2:.2e}; mean value {float((v_ours * mask).sum() / mask.sum()):+.3f}")
    dts = sorted({str(p.dtype) for p in critic.parameters() if p.requires_grad})
    check(rep, "critic trainable parameters are fp32", dts == ["torch.float32"], str(dts))
    del critic, ref_critic
    clear_gpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
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
