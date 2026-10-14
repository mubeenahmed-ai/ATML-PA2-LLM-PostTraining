"""Task 1 - validate the DPO objective and the log-prob plumbing BEFORE training.

python -m task1_dpo.validate --config configs/dpo.yaml

(1) analytic checks of task1_dpo.dpo.dpo_loss against the manual's equation
    (independent float64 implementation, pi == ref => log 2, gradient signs, and a
    demonstration of what the released starter formula did);
(2) model checks: fast sequence log-probs == course helper, fresh LoRA => pi == ref,
    cached reference log-probs == on-the-fly reference, one optimisation step on a
    batch increases its margin, generation log-prob helper == course helper.
Writes <results-dir>/validation_report.json and exits non-zero on failure.
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from common.data import load_yaml, read_jsonl
from common.generation import batch_generate, response_sequence_logprobs, response_token_logprobs
from common.logging_utils import set_seed
from common.models import clear_gpu, load_policy, load_tokenizer, reference_mode, resolve_dtype, trainable_parameters
from task1_dpo.dpo import dpo_loss
from task1_dpo.utils import (
    collate_pairs, encode_preference_rows, ensure_fp32_trainable, ensure_ref_cache, generated_response_logps,
    model_device, ref_cache_meta, save_json, sequence_logps, task_dirs,
)


def check(report, name, ok, detail):
    report.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def math_checks(report):
    g = torch.Generator().manual_seed(0)
    pc, pr, rc, rr = (torch.randn(64, generator=g, dtype=torch.float64) * 30 - 150 for _ in range(4))
    for beta in (0.03, 0.1, 0.3):
        loss, diag = dpo_loss(pc, pr, rc, rr, beta)
        ref_loss = -torch.log(torch.sigmoid(beta * ((pc - rc) - (pr - rr)))).mean()
        check(report, f"loss matches manual equation (beta={beta})", torch.allclose(loss, ref_loss, atol=1e-10),
              f"impl={loss.item():.8f} manual={ref_loss.item():.8f}")
    acc = ((pc - rc) - (pr - rr) > 0).double().mean()
    check(report, "diagnostic accuracy uses m > 0", abs(diag["preference_accuracy"].item() - acc.item()) < 1e-12,
          f"{diag['preference_accuracy'].item():.4f}")

    # pi == ref (fresh adapter): logits must be exactly 0 and the loss log(2), whatever the reference margin
    loss0, d0 = dpo_loss(rc, rr, rc, rr, 0.1)
    check(report, "pi == ref gives loss log(2) and zero margin", abs(loss0.item() - math.log(2)) < 1e-12
          and abs(d0["margin_mean"].item()) < 1e-12, f"loss={loss0.item():.6f} (log2={math.log(2):.6f})")
    starter = -F.logsigmoid(0.1 * ((rc - rr) + (rc - rr))).mean()
    check(report, "starter formula (policy_margin + ref_margin) was wrong at pi == ref", abs(starter.item() - math.log(2)) > 1e-3,
          f"starter loss at pi==ref = {starter.item():.4f} (should be log 2 = 0.6931)")

    # gradient directions: decreasing the loss must raise log pi(y+) and lower log pi(y-)
    pc_, pr_ = pc.clone().requires_grad_(True), pr.clone().requires_grad_(True)
    loss, _ = dpo_loss(pc_, pr_, rc, rr, 0.1)
    loss.backward()
    check(report, "gradient signs", bool((pc_.grad < 0).all() and (pr_.grad > 0).all()),
          "dL/dlogpi(y+) < 0 and dL/dlogpi(y-) > 0 for every pair")
    # translation invariance in the reference: shifting both reference log-probs equally changes nothing
    l1, _ = dpo_loss(pc, pr, rc + 5, rr + 5, 0.1)
    l2, _ = dpo_loss(pc, pr, rc, rr, 0.1)
    check(report, "invariant to a common shift of reference log-probs", torch.allclose(l1, l2), "ok")


def model_checks(report, cfg, dirs, n_pairs=2):
    set_seed(int(cfg["seed"]))
    tok = load_tokenizer(cfg["base_model"])
    rows = read_jsonl(cfg["paths"]["dpo_standard_train"])[: n_pairs * 4]
    items, _ = encode_preference_rows(tok, rows, int(cfg["max_sequence_length"]))
    items = sorted(items, key=lambda it: len(it["chosen"][0]) + len(it["rejected"][0]))[:n_pairs]
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    ensure_fp32_trainable(model)
    dev = model_device(model)
    dtypes = sorted({str(p.dtype) for p in trainable_parameters(model)})
    check(report, "trainable LoRA parameters are fp32", dtypes == ["torch.float32"], str(dtypes))

    batch = collate_pairs(tok, items, dev)
    model.eval()
    with torch.no_grad():
        fast, _, _ = sequence_logps(model, batch)
        slow, _, _ = response_sequence_logprobs(model, batch)
        with reference_mode(model):
            refv, _, _ = sequence_logps(model, batch)
    diff = (fast - slow).abs().max().item()
    tol = max(1.0, 2e-3 * slow.abs().max().item())  # fp16 GEMM-order noise; an indexing bug gives O(100) nats
    check(report, "fast sequence log-probs == common.generation.response_sequence_logprobs", diff < tol,
          f"max |diff| = {diff:.2e} nats (values ~{slow.abs().mean().item():.1f})")
    d2 = (fast - refv).abs().max().item()
    check(report, "fresh LoRA: policy == reference", d2 < 1e-3, f"max |logpi - logref| = {d2:.2e}")

    meta = dict(ref_cache_meta(cfg, cfg["paths"]["dpo_standard_train"]), note="validation")
    cache = ensure_ref_cache(model, tok, items, dirs["cache"] / "ref_validation_tmp.json", meta=meta)
    n = len(items)
    cached = torch.tensor([cache["chosen"][it["key"]] for it in items] + [cache["rejected"][it["key"]] for it in items])
    d3 = (cached - refv.float().cpu()).abs().max().item()
    check(report, "cached reference log-probs == on-the-fly reference", d3 < 0.5,
          f"max |diff| = {d3:.2e} nats (fp16 padding noise is expected to be tiny)")
    (dirs["cache"] / "ref_validation_tmp.json").unlink(missing_ok=True)

    # one exaggerated step on the same batch must increase the margin
    model.train()
    opt = torch.optim.AdamW(trainable_parameters(model), lr=1e-3)
    rc, rr = cached[:n].to(dev), cached[n:].to(dev)
    use_scaler = torch.cuda.is_available() and resolve_dtype(cfg.get("dtype", "float16")) == torch.float16
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0**12, enabled=use_scaler)
    for attempt in range(6):  # retry if the fp16 loss scaler skipped the step (overflow)
        opt.zero_grad(set_to_none=True)
        lp, _, _ = sequence_logps(model, batch)
        loss_before, d_before = dpo_loss(lp[:n], lp[n:], rc, rr, 0.1)
        scaler.scale(loss_before).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0)
        scale0 = scaler.get_scale() if use_scaler else 1.0
        scaler.step(opt)
        scaler.update()
        if not use_scaler or scaler.get_scale() >= scale0:
            break
    model.eval()
    with torch.no_grad():
        lp2, _, _ = sequence_logps(model, batch)
        loss_after, d_after = dpo_loss(lp2[:n], lp2[n:], rc, rr, 0.1)
    check(report, "one step increases the margin on its batch", d_after["margin_mean"].item() > d_before["margin_mean"].item(),
          f"margin {d_before['margin_mean'].item():+.4f} -> {d_after['margin_mean'].item():+.4f}, "
          f"loss {loss_before.item():.4f} -> {loss_after.item():.4f}")

    prompts = [[{"role": "user", "content": "Name three primary colors."}], [{"role": "user", "content": "What is 2+2?"}]]
    torch.manual_seed(0)
    gen = batch_generate(model, tok, prompts, max_prompt_length=64, max_new_tokens=16, do_sample=False)
    with torch.no_grad():
        a = generated_response_logps(model, gen) * gen["response_mask"]
        b, _ = response_token_logprobs(model, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"])
        b = b * gen["response_mask"]
    d4 = (a - b).abs().max().item()
    check(report, "generation log-prob helper == common.generation.response_token_logprobs", d4 < 0.1,
          f"max per-token |diff| = {d4:.2e} (fp16 rounding only)")
    del model, opt
    clear_gpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--work-dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--math-only", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    dirs = task_dirs(cfg, args.work_dir, args.results_dir)
    report = []
    math_checks(report)
    if not args.math_only:
        model_checks(report, cfg, dirs)
    save_json(dirs["results"] / "validation_report.json", report)
    n_fail = sum(not r["ok"] for r in report)
    print(f"\n{len(report) - n_fail}/{len(report)} checks passed -> {dirs['results'] / 'validation_report.json'}")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
