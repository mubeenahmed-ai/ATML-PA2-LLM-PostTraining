"""Shared Task 3 (GRPO) helpers.

atomic I/O, fixed prompt sets, policy loading from the supplied GRPO midpoint,
memory-efficient response log-probs + entropy, Drive-safe resumable checkpoints and
generation records for held-out evaluation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from common.data import prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate
from common.metrics import word_count
from common.models import load_policy, reference_mode


# ----------------------------------------------------------------------------- paths / io
def task_dirs(cfg: dict, work_dir: str | None = None, results_dir: str | None = None) -> dict:
    """work_dir: adapters / resume checkpoints (not committed).
    results_dir: small JSON/CSV/JSONL logs used in the report (committed)."""
    work = repo_path(work_dir) if work_dir else repo_path(Path(cfg["output"]).parent)
    results = repo_path(results_dir) if results_dir else repo_path(cfg["results_dir"])
    d = {
        "work": work,
        "results": results,
        "resume": work / "resume",
        "train_results": results / "train",
        "eval_results": results / "eval",
        "tables": results / "tables",
        "figures": results / "figures",
        "qualitative": results / "qualitative",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
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
    atomic_write_text(path, "".join(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n" for r in rows))


def read_jsonl_abs(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def truncate_log(path, max_update: int, key: str = "update") -> None:
    path = Path(path)
    if path.exists():
        write_jsonl(path, [r for r in read_jsonl_abs(path) if int(r.get(key, 0)) <= max_update])


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
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
            "min": float(a.min()), "max": float(a.max())}


def gpu_peak_gb() -> float:
    return float(torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0


def model_device(model):
    return next(model.parameters()).device


def masked_stats(x: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    v = x[mask.bool()].float()
    if v.numel() == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(unbiased=False)) if v.numel() > 1 else 0.0


def explained_variance(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask.bool()
    y, p = target[m].float(), pred[m].float()
    if y.numel() < 2 or float(y.var(unbiased=False)) == 0.0:
        return float("nan")
    return float(1.0 - (y - p).var(unbiased=False) / y.var(unbiased=False))


# ----------------------------------------------------------------------------- fixed prompt sets
def rendered_len(tokenizer, messages) -> int:
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))


def train_prompt_order(cfg, tokenizer, dirs) -> tuple[list[dict], list[int]]:
    """Fixed training-prompt sequence shared by the standard run and every fork.

    Only prompts whose chat-rendered length is <= max_prompt_length are eligible
    (batch_generate truncates longer prompts from the right, which would cut off the
    assistant turn). Eligible rows are shuffled once with the config seed and saved.
    Update u uses order[u*P:(u+1)*P]."""
    path = dirs["results"] / "train_prompt_order.json"
    src = cfg["paths"]["rl_prompt_train"]
    rows = read_jsonl(src)
    maxp = int(cfg["max_prompt_length"])
    if path.exists():
        sel = load_json(path)
        if sel["source_sha"] != file_fingerprint(src) or sel["max_prompt_length"] != maxp:
            raise SystemExit(f"{path} does not match the current prompt file / max_prompt_length.")
        return rows, sel["order"]
    eligible = [i for i, r in enumerate(rows) if rendered_len(tokenizer, prompt_messages(r)) <= maxp]
    order = np.random.RandomState(int(cfg["seed"])).permutation(eligible).tolist()
    save_json(path, {
        "source_file": rel_to_repo(src), "source_sha": file_fingerprint(src), "max_prompt_length": maxp,
        "rule": "rows whose rendered prompt <= max_prompt_length, permuted with the config seed",
        "n_rows": len(rows), "n_eligible": len(eligible), "order": order,
        "first_prompt_ids": [rows[i].get("prompt_id") for i in order[:64]],
    })
    return rows, order


def select_eval_prompts(cfg, tokenizer, dirs, n: int) -> list[dict]:
    """Fixed held-out evaluation prompts: first n rows of rl_prompt_eval (file order) whose
    rendered prompt fits max_prompt_length. Saved once; identical for every model."""
    path = dirs["results"] / "eval_prompt_ids.json"
    src = cfg["paths"]["rl_prompt_eval"]
    rows = read_jsonl(src)
    maxp = int(cfg["max_prompt_length"])
    if path.exists():
        sel = load_json(path)
        if sel["n_requested"] != n or sel["max_prompt_length"] != maxp:
            raise SystemExit(f"{path} was created with n={sel['n_requested']}; keep the eval set fixed "
                             "(or delete it AND every eval/*/generations.jsonl).")
    else:
        chosen = []
        for i, r in enumerate(rows):
            if rendered_len(tokenizer, prompt_messages(r)) <= maxp:
                chosen.append(i)
            if len(chosen) == n:
                break
        sel = {"source_file": rel_to_repo(src), "source_sha": file_fingerprint(src), "n_requested": n,
               "max_prompt_length": maxp, "rule": "first n rows (file order) with rendered prompt <= max_prompt_length",
               "rows": chosen, "prompt_ids": [rows[i].get("prompt_id") for i in chosen]}
        save_json(path, sel)
    return [{"row": i, "prompt_id": rows[i].get("prompt_id"), "messages": prompt_messages(rows[i])} for i in sel["rows"]]


# ----------------------------------------------------------------------------- models
def disable_dropout(model) -> int:
    """The clipped ratio needs pi_old and pi_theta evaluated with the same function; LoRA
    dropout noise would make the ratio != 1 with identical weights, so dropout is disabled."""
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout) and m.p > 0:
            m.p = 0.0
            n += 1
    return n


def ensure_fp32_trainable(model) -> int:
    n = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            n += 1
    return n


def load_policy_from(cfg, adapter, trainable: bool):
    """adapter: path, 'midpoint' (supplied GRPO checkpoint) or None (untouched SFT policy)."""
    if adapter == "midpoint":
        adapter = cfg["paths"]["grpo_midpoint_policy"]
    model = load_policy(cfg, adapter_path=adapter, trainable=trainable)
    disable_dropout(model)
    if trainable:
        ensure_fp32_trainable(model)
    return model


def trainable_state(model) -> dict:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model, sd: dict) -> None:
    params = dict(model.named_parameters())
    missing = [n for n, p in params.items() if p.requires_grad and n not in sd]
    unexpected = [n for n in sd if n not in params]
    if missing or unexpected:
        raise RuntimeError(f"trainable-state mismatch: missing={missing[:3]} unexpected={unexpected[:3]}")
    with torch.no_grad():
        for n, t in sd.items():
            params[n].copy_(t.to(device=params[n].device, dtype=params[n].dtype))


# ----------------------------------------------------------------------------- log-probs / entropy
def response_logps(model, sequences, attention_mask, response_ids):
    """Per-token log pi(a_t|s_t) of the response tokens [B, T] (with grad if enabled).
    Equivalent to common.generation.response_token_logprobs; only response logits are built."""
    T = response_ids.shape[1]
    out = model(input_ids=sequences, attention_mask=attention_mask, use_cache=False, return_dict=True,
                logits_to_keep=T + 1)
    logp = torch.log_softmax(out.logits[:, :-1, :].float(), dim=-1)
    return torch.gather(logp, -1, response_ids.unsqueeze(-1)).squeeze(-1)


def response_logps_with_entropy(model, sequences, attention_mask, response_ids):
    """Per-token log-probs WITH grad and the full-distribution entropy (detached) [B, T]."""
    T = response_ids.shape[1]
    out = model(input_ids=sequences, attention_mask=attention_mask, use_cache=False, return_dict=True,
                logits_to_keep=T + 1)
    logp = torch.log_softmax(out.logits[:, :-1, :].float(), dim=-1)
    tok = torch.gather(logp, -1, response_ids.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        ent = -(logp.detach().exp() * logp.detach()).sum(-1)
    return tok, ent


@torch.no_grad()
def response_logps_entropy(model, sequences, attention_mask, response_ids, sub_batch: int = 2, entropy: bool = True,
                           temperature: float = 1.0):
    """No-grad per-token log-probs and FULL-distribution entropy H(pi(.|s_t)) [B, T].
    temperature != 1 is only used to identify the convention of externally cached log-probs."""
    T = response_ids.shape[1]
    lps, ents = [], []
    for s in range(0, sequences.shape[0], sub_batch):
        out = model(input_ids=sequences[s : s + sub_batch], attention_mask=attention_mask[s : s + sub_batch],
                    use_cache=False, return_dict=True, logits_to_keep=T + 1)
        logp = torch.log_softmax(out.logits[:, :-1, :].float() / float(temperature), dim=-1)
        lps.append(torch.gather(logp, -1, response_ids[s : s + sub_batch].unsqueeze(-1)).squeeze(-1))
        if entropy:
            ents.append(-(logp.exp() * logp).sum(-1))
        del out, logp
    return torch.cat(lps, 0), (torch.cat(ents, 0) if entropy else None)


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
        except Exception as exc:
            print("WARNING: could not restore CUDA RNG state:", exc)


# Google-Drive-safe checkpoints: two alternating single-file slots (<resume_dir>/slot_A.pt,
# slot_B.pt + .json sidecars), written locally, copied under a temporary FILE name and atomically
# replaced. No directory is renamed or deleted on Drive; the previous slot survives a failed save.
def _slot_step(resume_dir: Path, slot: str):
    j = resume_dir / f"slot_{slot}.json"
    if not (j.exists() and (resume_dir / f"slot_{slot}.pt").exists()):
        return None
    try:
        return int(load_json(j)["step"])
    except Exception:
        return None


def save_slot_checkpoint(resume_dir, payload: dict, step: int) -> None:
    resume_dir = Path(resume_dir)
    resume_dir.mkdir(parents=True, exist_ok=True)
    steps = {s: _slot_step(resume_dir, s) for s in ("A", "B")}
    target = min(("A", "B"), key=lambda s: -1 if steps[s] is None else steps[s])
    local = Path(tempfile.gettempdir()) / f"pa2_ckpt_{os.getpid()}_{target}.pt"
    torch.save(payload, local)
    dst_tmp = resume_dir / f"slot_{target}.pt.tmp"
    shutil.copyfile(local, dst_tmp)
    os.replace(dst_tmp, resume_dir / f"slot_{target}.pt")
    local.unlink(missing_ok=True)
    save_json(resume_dir / f"slot_{target}.json", {"step": int(step)})


def load_latest_slot(resume_dir):
    resume_dir = Path(resume_dir)
    cands = sorted([(st, s) for s in ("A", "B") if (st := _slot_step(resume_dir, s)) is not None], reverse=True)
    for st, s in cands:
        try:
            return torch.load(resume_dir / f"slot_{s}.pt", map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"WARNING: checkpoint slot {s} (update {st}) unreadable ({exc}); trying the other slot")
    return None


def clear_resume(resume_dir) -> None:
    resume_dir = Path(resume_dir)
    for s in ("A", "B"):
        for suffix in (".pt", ".json", ".pt.tmp"):
            (resume_dir / f"slot_{s}{suffix}").unlink(missing_ok=True)


# ----------------------------------------------------------------------------- held-out generation
@torch.no_grad()
def generate_eval_records(model, tokenizer, prompt_rows, gen_cfg, max_new_tokens, max_prompt_length,
                          batch_size, seed, compute_ref: bool):
    """One sampled response per prompt (fixed seed per batch) with per-token policy and
    reference log-probs and full-distribution entropy."""
    records = []
    t0 = time.time()
    nb = math.ceil(len(prompt_rows) / batch_size)
    for b, s in enumerate(range(0, len(prompt_rows), batch_size)):
        chunk = prompt_rows[s : s + batch_size]
        torch.manual_seed(seed + b)
        gen = batch_generate(model, tokenizer, [c["messages"] for c in chunk], max_prompt_length=max_prompt_length,
                             max_new_tokens=max_new_tokens, temperature=float(gen_cfg.get("temperature", 0.7)),
                             top_p=float(gen_cfg.get("top_p", 0.9)), do_sample=bool(gen_cfg.get("do_sample", True)))
        mask = gen["response_mask"]
        pol, ent = response_logps_entropy(model, gen["sequences"], gen["attention_mask"], gen["response_ids"])
        if compute_ref:
            with reference_mode(model):
                ref, _ = response_logps_entropy(model, gen["sequences"], gen["attention_mask"], gen["response_ids"], entropy=False)
        else:
            ref = pol
        for j, c in enumerate(chunk):
            n = int(mask[j].sum().item())
            lr = (pol[j, :n] - ref[j, :n]).float().cpu()
            rec = dict(c)
            rec.update({
                "response": gen["responses"][j],
                "response_tokens": n,
                "response_ids": gen["response_ids"][j, :n].cpu().tolist(),
                "terminated_with_eos": bool(gen["terminated_with_eos"][j]),
                "truncated": bool(gen["truncated"][j]),
                "policy_logp_sum": float(pol[j, :n].sum()),
                "ref_logp_sum": float(ref[j, :n].sum()),
                "kl_seq": float(lr.sum()),
                "logratio_tokens": [round(float(x), 5) for x in lr.tolist()],
                "entropy_sum": float(ent[j, :n].sum()),
                "word_count": int(word_count(gen["responses"][j])),
                "gen_seed": seed + b,
            })
            records.append(rec)
        print(f"  [gen] batch {b + 1}/{nb} ({time.time() - t0:.0f}s)", flush=True)
    return records
