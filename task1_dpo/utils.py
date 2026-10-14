"""Shared Task 1 (DPO) helpers.

encoding of preference pairs, memory-efficient response log-probabilities, a
persistent reference-log-prob cache, resumable training checkpoints, generation
helpers, and small atomic I/O utilities. Every long-running piece writes its
progress to disk so a crashed / disconnected session can continue.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from common.data import (
    encode_prompt_response,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    repo_path,
)
from common.generation import batch_generate
from common.metrics import word_count
from common.models import reference_mode

COMMON_LOSS_BETA = 0.10  # beta used to report a comparable held-out loss across beta forks


# ----------------------------------------------------------------------------- paths / io
def task_dirs(cfg: dict, work_dir: str | None = None, results_dir: str | None = None) -> dict:
    """work_dir   : adapters, resume checkpoints, reference cache (large-ish, not committed)
    results_dir: small JSON/CSV/JSONL logs used in the report (committed to Git)."""
    work = repo_path(work_dir) if work_dir else repo_path(Path(cfg["standard_output"]).parent)
    results = repo_path(results_dir) if results_dir else repo_path(cfg["results_dir"])
    d = {
        "work": work,
        "results": results,
        "cache": work / "cache",
        "resume": work / "resume",
        "train_results": results / "train",
        "eval_results": results / "eval",
        "tables": results / "tables",
        "figures": results / "figures",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def atomic_write_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_json(path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path, rows) -> None:
    atomic_write_text(
        path, "".join(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n" for r in rows)
    )


def read_jsonl_abs(path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def file_fingerprint(path) -> str:
    h = hashlib.sha256()
    with open(repo_path(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def rel_to_repo(path) -> str:
    p = repo_path(path)
    try:
        return str(p.relative_to(repo_path(".")))
    except ValueError:
        return str(p)


def describe(values) -> dict:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return {"n": 0}
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(med),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def softplus_np(x):
    """-log(sigmoid(z)) == softplus(-z); numerically stable."""
    return np.logaddexp(0.0, x)


def gpu_peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / 2**30)


def model_device(model):
    return next(model.parameters()).device


# ----------------------------------------------------------------------------- encoding
def response_token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def encode_preference_rows(tokenizer, rows: list[dict], max_length: int):
    """Encode (prompt, chosen) and (prompt, rejected) with the released course encoder.

    Rows are always a PREFIX of a course file, so the list index is the file row index.
    Rows whose prompt alone does not fit in max_length are skipped and reported.
    """
    items, skipped = [], []
    for i, row in enumerate(rows):
        prompt = prompt_messages_from_preference(row)
        yc, yr = preference_responses(row)
        try:
            c_ids, c_mask = encode_prompt_response(tokenizer, prompt, yc, max_length)
            r_ids, r_mask = encode_prompt_response(tokenizer, prompt, yr, max_length)
        except ValueError as exc:
            skipped.append({"row": i, "prompt_id": row.get("prompt_id"), "reason": str(exc)})
            continue
        c_raw = response_token_count(tokenizer, yc) + 1  # + EOS
        r_raw = response_token_count(tokenizer, yr) + 1
        items.append(
            {
                "key": str(i),
                "row": i,
                "prompt_id": row.get("prompt_id"),
                "source_index": row.get("source_index"),
                "length_stratum": row.get("length_stratum"),
                "chosen": (c_ids, c_mask),
                "rejected": (r_ids, r_mask),
                "prompt_tokens": len(c_ids) - int(sum(c_mask)),
                "chosen_resp_tokens": int(sum(c_mask)),
                "rejected_resp_tokens": int(sum(r_mask)),
                "chosen_truncated": int(sum(c_mask)) < c_raw,
                "rejected_truncated": int(sum(r_mask)) < r_raw,
                "chosen_tokens_full": c_raw - 1,
                "rejected_tokens_full": r_raw - 1,
            }
        )
    return items, skipped


def collate_pairs(tokenizer, items: list[dict], device=None) -> dict:
    """One left-padded batch: rows [0, n) are chosen, rows [n, 2n) are rejected."""
    examples = [it["chosen"] for it in items] + [it["rejected"] for it in items]
    batch = pad_batch(tokenizer, examples)
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch


def sequence_logps(model, batch: dict):
    """Summed response-token log-probs, numerically identical to
    common.generation.response_sequence_logprobs but far cheaper in memory.

    pad_batch left-pads, so every response occupies the LAST positions of its row.
    With R = longest response in the batch we only need logits at the last R+1
    positions (`logits_to_keep`), instead of a [B, L, |V|] tensor.
    """
    rmask = batch["response_mask"]
    R = int(rmask.sum(-1).max().item())
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
        logits_to_keep=R + 1,
    )
    logits = out.logits[:, :-1, :]  # predicts the last R tokens
    labels = batch["input_ids"][:, -R:]
    mask = rmask[:, -R:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok = torch.gather(logp, -1, labels.unsqueeze(-1)).squeeze(-1)
    return (tok * mask).sum(-1), tok, mask


@torch.no_grad()
def policy_logps_for_items(model, tokenizer, items, batch_size=4, desc="policy"):
    """Eval-mode (no dropout) sequence log-probs for chosen/rejected of every item."""
    was_training = model.training
    model.eval()
    device = model_device(model)
    out_c, out_r = {}, {}
    order = sorted(items, key=lambda it: max(len(it["chosen"][0]), len(it["rejected"][0])))
    t0 = time.time()
    for b, s in enumerate(range(0, len(order), batch_size)):
        chunk = order[s : s + batch_size]
        batch = collate_pairs(tokenizer, chunk, device)
        logps, _, _ = sequence_logps(model, batch)
        logps = logps.float().cpu().tolist()
        n = len(chunk)
        for j, it in enumerate(chunk):
            out_c[it["key"]] = logps[j]
            out_r[it["key"]] = logps[n + j]
        if (b + 1) % 25 == 0:
            print(f"  [{desc}] {s + n}/{len(order)} pairs  ({time.time() - t0:.0f}s)", flush=True)
    if was_training:
        model.train()
    return out_c, out_r


def ref_cache_meta(cfg: dict, data_path: str) -> dict:
    return {
        "file": rel_to_repo(data_path),
        "sha": file_fingerprint(data_path),
        "max_length": int(cfg["max_sequence_length"]),
        "base_model": str(cfg["base_model"]),
        "dtype": str(cfg.get("dtype", "float16")),
    }


@torch.no_grad()
def ensure_ref_cache(model, tokenizer, items, cache_path, meta: dict, batch_size=4, save_every=20):
    """Reference log-probs pi_ref(y|x) for chosen/rejected, computed ONCE per data file.

    The reference policy is the frozen base model (LoRA adapter disabled, eval mode),
    so its log-probs never change during training; caching them removes one forward
    pass per training step and makes resuming cheap. The cache is keyed by file row
    index and is written incrementally (resumable).
    """
    cache_path = Path(cache_path)
    store = {"meta": meta, "chosen": {}, "rejected": {}}
    if cache_path.exists():
        try:
            old = load_json(cache_path)
        except Exception:
            old = None
        if old and old.get("meta") == meta:
            store = old
        else:
            print(f"[ref-cache] {cache_path.name}: metadata changed/unreadable -> recomputing", flush=True)
    todo = [it for it in items if it["key"] not in store["chosen"]]
    if not todo:
        return store
    print(f"[ref-cache] computing reference log-probs for {len(todo)} pairs -> {cache_path}", flush=True)
    todo = sorted(todo, key=lambda it: max(len(it["chosen"][0]), len(it["rejected"][0])))
    device = model_device(model)
    t0 = time.time()
    with reference_mode(model):
        for b, s in enumerate(range(0, len(todo), batch_size)):
            chunk = todo[s : s + batch_size]
            batch = collate_pairs(tokenizer, chunk, device)
            logps, _, _ = sequence_logps(model, batch)
            logps = logps.float().cpu().tolist()
            n = len(chunk)
            for j, it in enumerate(chunk):
                store["chosen"][it["key"]] = logps[j]
                store["rejected"][it["key"]] = logps[n + j]
            if (b + 1) % save_every == 0:
                save_json(cache_path, store)
                print(f"  [ref-cache] {s + n}/{len(todo)}  ({time.time() - t0:.0f}s)", flush=True)
    save_json(cache_path, store)
    print(f"[ref-cache] done in {time.time() - t0:.0f}s", flush=True)
    return store


def ensure_fp32_trainable(model) -> int:
    """LoRA weights must be fp32 for a stable optimizer / GradScaler (PEFT normally
    up-casts them automatically; this is a guard)."""
    n = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            n += 1
    return n


# ----------------------------------------------------------------------------- resume checkpoints
def rng_state() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(st["cuda"])
        except Exception as exc:  # different GPU count after reconnect
            print("WARNING: could not restore CUDA RNG state:", exc)


def save_resume_checkpoint(resume_dir, model, optimizer, scaler, state: dict) -> None:
    """Write to <dir>.tmp, then swap into place, so a disconnect mid-save never
    corrupts the last good checkpoint. state.json is written last = completeness marker."""
    resume_dir = Path(resume_dir)
    tmp = resume_dir.with_name(resume_dir.name + ".tmp")
    old = resume_dir.with_name(resume_dir.name + ".old")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(str(tmp / "adapter"))
    torch.save(
        {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state()},
        tmp / "trainer_state.pt",
    )
    save_json(tmp / "state.json", state)
    if old.exists():
        shutil.rmtree(old)
    if resume_dir.exists():
        resume_dir.rename(old)
    tmp.rename(resume_dir)
    if old.exists():
        shutil.rmtree(old)


def find_resume_checkpoint(resume_dir):
    """Most advanced COMPLETE checkpoint among <dir>, <dir>.old, <dir>.tmp (or None)."""
    resume_dir = Path(resume_dir)
    best, best_step = None, -1
    for cand in [resume_dir, resume_dir.with_name(resume_dir.name + ".old"), resume_dir.with_name(resume_dir.name + ".tmp")]:
        sj = cand / "state.json"
        if sj.exists() and (cand / "trainer_state.pt").exists() and (cand / "adapter").exists():
            try:
                step = int(load_json(sj)["opt_step"])
            except Exception:
                continue
            if step > best_step:
                best, best_step = cand, step
    return best


def clear_resume(resume_dir) -> None:
    resume_dir = Path(resume_dir)
    for cand in [resume_dir, resume_dir.with_name(resume_dir.name + ".old"), resume_dir.with_name(resume_dir.name + ".tmp")]:
        if cand.exists():
            shutil.rmtree(cand)


def truncate_log(path, max_opt_step: int) -> None:
    path = Path(path)
    if not path.exists():
        return
    keep = [r for r in read_jsonl_abs(path) if int(r.get("opt_step", 0)) <= max_opt_step]
    write_jsonl(path, keep)


# ----------------------------------------------------------------------------- generation
@torch.no_grad()
def generated_response_logps(model, gen: dict, sub_batch: int = 4) -> torch.Tensor:
    """Per-token log-probs of generated response tokens (unmasked [B, T]); equivalent to
    common.generation.response_token_logprobs but only materialises response logits."""
    seq, attn, rids = gen["sequences"], gen["attention_mask"], gen["response_ids"]
    T = rids.shape[1]
    outs = []
    for s in range(0, seq.shape[0], sub_batch):
        o = model(
            input_ids=seq[s : s + sub_batch],
            attention_mask=attn[s : s + sub_batch],
            use_cache=False,
            return_dict=True,
            logits_to_keep=T + 1,
        )
        lp = torch.log_softmax(o.logits[:, :-1, :].float(), dim=-1)
        outs.append(torch.gather(lp, -1, rids[s : s + sub_batch].unsqueeze(-1)).squeeze(-1))
    return torch.cat(outs, 0)


def generate_records(
    model,
    tokenizer,
    prompt_rows: list[dict],
    gen_cfg: dict,
    max_new_tokens: int,
    max_prompt_length: int,
    batch_size: int,
    seed: int,
    compute_ref: bool,
    log_every: int = 1,
):
    """Generate one sampled response per entry in prompt_rows (each has 'messages'),
    with a fixed seed per batch, and record policy/reference log-probs of the sampled
    tokens for the sampled-response KL estimator."""
    records = []
    t0 = time.time()
    n_batches = math.ceil(len(prompt_rows) / batch_size)
    for b, s in enumerate(range(0, len(prompt_rows), batch_size)):
        chunk = prompt_rows[s : s + batch_size]
        torch.manual_seed(seed + b)  # also seeds CUDA
        gen = batch_generate(
            model,
            tokenizer,
            [c["messages"] for c in chunk],
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            temperature=float(gen_cfg.get("temperature", 0.7)),
            top_p=float(gen_cfg.get("top_p", 0.9)),
            do_sample=bool(gen_cfg.get("do_sample", True)),
        )
        mask = gen["response_mask"]
        pol = generated_response_logps(model, gen) * mask
        if compute_ref:
            with reference_mode(model):
                ref = generated_response_logps(model, gen) * mask
        else:
            ref = pol
        diff = (pol - ref).sum(-1).float().cpu().tolist()
        pol_sum = pol.sum(-1).float().cpu().tolist()
        ref_sum = ref.sum(-1).float().cpu().tolist()
        for j, c in enumerate(chunk):
            resp = gen["responses"][j]
            rec = dict(c)
            rec.update(
                {
                    "response": resp,
                    "response_tokens": int(gen["response_lengths"][j]),
                    "terminated_with_eos": bool(gen["terminated_with_eos"][j]),
                    "truncated": bool(gen["truncated"][j]),
                    "policy_logp_sum": float(pol_sum[j]),
                    "ref_logp_sum": float(ref_sum[j]),
                    "kl_seq": float(diff[j]),
                    "word_count": int(word_count(resp)),
                    "gen_batch": b,
                    "gen_seed": seed + b,
                }
            )
            records.append(rec)
        if (b + 1) % log_every == 0:
            print(f"  [gen] batch {b + 1}/{n_batches}  ({time.time() - t0:.0f}s)", flush=True)
    return records
