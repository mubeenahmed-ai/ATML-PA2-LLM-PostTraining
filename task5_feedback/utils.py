"""Shared Task 5 (RLVR vs RLAIF) helpers: atomic I/O, statistics, response failure typing, verifier
variants for the specification analysis, a crash-safe wrapper around the released judge cache, and a
fixed-presentation judge call used ONLY to measure position bias (the protocol itself uses the released
PairwiseAIJudge.compare unchanged)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from common.data import repo_path
from task5_feedback.rlaif import PAIRWISE_RUBRIC, PairwiseAIJudge
from task5_feedback.rlvr import _FINAL_RE, exact_reward, extract_designated_final, numerically_equal

POLICIES = ["sft", "rlvr", "rlaif"]


# ----------------------------------------------------------------------------- paths / io
def task_dirs(cfg: dict, results_dir: str | None = None) -> dict:
    results = repo_path(results_dir) if results_dir else repo_path(cfg["results_dir"]) / "task5_feedback"
    d = {"results": results, "gsm": results / "gsm", "transfer": results / "transfer", "diagnostics": results / "diagnostics",
         "judge": results / "judge", "tables": results / "tables", "figures": results / "figures", "qualitative": results / "qualitative"}
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
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


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


def append_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def describe(values) -> dict:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return {"n": 0}
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1)}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, c - h), min(1.0, c + h))


def bootstrap_ci(values, n_boot=2000, seed=0):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, v.size, v.size)].mean() for _ in range(n_boot)]
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


# ----------------------------------------------------------------------------- verifier variants
_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def last_number(text: str):
    m = _NUM_RE.findall(str(text))
    return m[-1].replace(",", "") if m else None


def verifier_variants(response: str, gold) -> dict:
    """strict  = released exact_reward (last designated `#### <number>` only);
    last_num  = last number anywhere in the response (a common lenient parser);
    any_num   = gold appears ANYWHERE as a number (deliberately naive: rewards discarded candidates)."""
    nums = [n.replace(",", "") for n in _NUM_RE.findall(str(response))]
    return {"strict": exact_reward(response, str(gold)),
            "last_num": float(last_number(response) is not None and numerically_equal(last_number(response), gold)),
            "any_num": float(any(numerically_equal(n, gold) for n in nums))}


def failure_type(response: str, gold, truncated: bool) -> dict:
    """Mutually exclusive outcome type plus diagnostic flags."""
    text = str(response)
    pred = extract_designated_final(text)
    has_marker = "####" in text
    nums = [n.replace(",", "") for n in _NUM_RE.findall(text)]
    gold_mentioned = any(numerically_equal(n, gold) for n in nums)
    if pred is not None and numerically_equal(pred, gold):
        t = "correct"
    elif pred is not None:
        t = "wrong_final"
    elif has_marker:
        t = "unparseable_final"           # e.g. '#### $18', '#### \\boxed{18}'
    elif truncated:
        t = "no_final_truncated"
    else:
        t = "no_final_marker"
    return {"type": t, "format_ok": pred is not None, "n_designated_finals": len(_FINAL_RE.findall(text)),
            "gold_mentioned": gold_mentioned,
            "gold_mentioned_but_wrong": t != "correct" and gold_mentioned}


# ----------------------------------------------------------------------------- judge helpers
def make_judge(cfg, cache_path):
    """Released PairwiseAIJudge with crash protection for its JSON cache: the released class rewrites the
    whole cache after every comparison (non-atomically); if a disconnect corrupted it, restore the last
    atomic backup written by `backup_judge_cache`."""
    cache_path = Path(cache_path)
    bak = cache_path.with_name(cache_path.name + ".bak")
    if cache_path.exists():
        try:
            json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            print(f"[judge] cache {cache_path.name} unreadable - restoring backup", flush=True)
            if bak.exists():
                shutil.copyfile(bak, cache_path)
            else:
                cache_path.unlink()
    return PairwiseAIJudge(cfg, cache_path)


def backup_judge_cache(judge) -> None:
    p = Path(judge.cache_path)
    save_json(p.with_name(p.name + ".bak"), judge.cache)


def timed_compare(judge, problem, a, b, stats: dict):
    """Released compare() (cached, deterministic orientation balancing). Identical candidates are a TIE
    without a judge call (a call could only reflect position bias)."""
    if a == b:
        stats["identical_short_circuit"] = stats.get("identical_short_circuit", 0) + 1
        return "TIE"
    cached = judge._key(problem, a, b) in judge.cache
    t0 = time.time()
    r = judge.compare(problem, a, b)
    if not cached:
        stats["calls"] = stats.get("calls", 0) + 1
        stats["seconds"] = stats.get("seconds", 0.0) + time.time() - t0
    return r


@torch.no_grad()
def presented_compare(judge, problem, first, second, cache: dict):
    """Same rubric, chat template, greedy 4-token decoding and parser as the released compare, but with a
    FIXED presentation order (no hash swap). Used only to measure position (order) consistency."""
    key = hashlib.sha256(json.dumps({"m": judge.cfg["ai_judge_model"], "p": problem, "first": first, "second": second},
                                    sort_keys=True).encode()).hexdigest()
    if key in cache:
        return cache[key]
    text = PAIRWISE_RUBRIC.format(problem=problem, a=first, b=second)
    ids = judge.tokenizer.apply_chat_template([{"role": "user", "content": text}], return_tensors="pt",
                                              add_generation_prompt=True).to(next(judge.model.parameters()).device)
    out = judge.model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=judge.tokenizer.eos_token_id,
                               eos_token_id=judge.tokenizer.eos_token_id)
    decoded = judge.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().upper()
    m = re.search(r"\b(A|B|TIE)\b", decoded)
    cache[key] = m.group(1) if m else "TIE"
    return cache[key]


@torch.no_grad()
def released_raw_output(judge, problem, a, b, cache: dict) -> str:
    """Raw decoded text of the released compare() for (a, b): SAME key, SAME hash-based orientation swap,
    prompt and decoding - used only to tell an explicit 'TIE' from an unparseable output (which the released
    parser silently maps to TIE)."""
    key = judge._key(problem, a, b)
    if key in cache:
        return cache[key]
    swap = int(key[:8], 16) % 2 == 1
    aa, bb = (b, a) if swap else (a, b)
    text = PAIRWISE_RUBRIC.format(problem=problem, a=aa, b=bb)
    ids = judge.tokenizer.apply_chat_template([{"role": "user", "content": text}], return_tensors="pt",
                                              add_generation_prompt=True).to(next(judge.model.parameters()).device)
    out = judge.model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=judge.tokenizer.eos_token_id,
                               eos_token_id=judge.tokenizer.eos_token_id)
    cache[key] = judge.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().upper()
    return cache[key]


def tie_kind(raw: str) -> str:
    m = re.search(r"\b(A|B|TIE)\b", str(raw))
    return "explicit_tie" if (m and m.group(1) == "TIE") else ("ambiguous_output" if not m else "mismatch")


def pref_score(result: str) -> float:
    """A (first argument preferred) = 1, TIE = 0.5, B = 0 - the manual's pairwise win-rate convention."""
    return {"A": 1.0, "TIE": 0.5, "B": 0.0}[result]


def verifier_pref(r_a: float, r_b: float) -> str:
    return "A" if r_a > r_b else ("B" if r_b > r_a else "TIE")
