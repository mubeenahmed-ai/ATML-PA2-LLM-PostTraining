"""Task 4, Step 3 - blind manual-audit sheet for the fixed 60-prompt subset.

python -m task4_safety.make_audit_sheet --config configs/feedback.yaml [--mode all_policies|one_per_id]

The 60 XSTest ids (30 SAFE + 30 UNSAFE) come from the released fixed_audit_ids (config seed).
  all_policies (default): every audited id x each of the four policies (240 items) - this follows the
      released instruction to join the ids to EACH policy's responses and allows policy comparisons.
  one_per_id: each id paired with ONE policy (balanced over policies within SAFE and within UNSAFE), 60 items.
The sheet shows only item_id, prompt and response, in a shuffled order: no AI label, policy name or
benchmark class (kept in audit_key.csv - do not open it while labelling). Existing labels are never
overwritten; the sheet is only (re)built when no labels exist yet.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl
from task4_safety.utils import POLICY_ORDER, read_jsonl_abs, save_csv, save_json, task_dirs


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def build_items(ids, classes, policies, mode, seed):
    if mode == "all_policies":
        items = [(i, p) for i in ids for p in policies]
    else:
        rng = np.random.default_rng(seed + 7)
        items = []
        for cls in ("SAFE", "UNSAFE"):
            cls_ids = [i for i in ids if classes[i] == cls]
            rng.shuffle(cls_ids)
            items += [(i, policies[k % len(policies)]) for k, i in enumerate(cls_ids)]
    order = np.random.default_rng(seed + 1).permutation(len(items))
    return [items[k] for k in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--results-dir")
    ap.add_argument("--mode", choices=["all_policies", "one_per_id"], default="all_policies")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    dirs = task_dirs(cfg, args.results_dir)
    src = dirs["gen"] / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    gens = {p: {r["xstest_id"]: r for r in read_jsonl_abs(dirs["gen"] / f"generated_{p}.jsonl")}
            for p in POLICY_ORDER if (dirs["gen"] / f"generated_{p}.jsonl").exists()}
    policies = [p for p in POLICY_ORDER if p in gens]
    if len(policies) < 4:
        print(f"WARNING: only {policies} have generations; the audit sheet will cover these.")
    classes = {i: gens["sft"][i]["benchmark_class"] for i in ids}
    save_csv(pd.DataFrame({"xstest_id": ids, "benchmark_class": [classes[i] for i in ids]}), dirs["audit"] / "audit_ids.csv")

    sheet_p, key_p, labels_p = dirs["audit"] / "audit_sheet.csv", dirs["audit"] / "audit_key.csv", dirs["audit"] / "manual_labels.csv"
    has_labels = labels_p.exists() and len(pd.read_csv(labels_p).dropna(subset=["manual_label"])) > 0
    if sheet_p.exists():
        old = pd.read_csv(sheet_p)
        has_labels = has_labels or old.get("manual_label", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("").any()
    if sheet_p.exists() and has_labels:
        key = pd.read_csv(key_p)
        changed = [r.item_id for r in key.itertuples() if gens.get(r.policy, {}).get(r.xstest_id, {}).get("response") !=
                   pd.read_csv(sheet_p).set_index("item_id").loc[r.item_id, "response"]]
        print(f"Audit sheet already has labels - leaving it untouched ({sheet_p}).")
        if changed:
            print(f"WARNING: {len(changed)} audited responses differ from the current generations (were they regenerated?): "
                  f"{changed[:5]}")
        return
    items = build_items(ids, classes, policies, args.mode, int(cfg["seed"]))
    rows, key = [], []
    for k, (i, p) in enumerate(items):
        g = gens[p][i]
        item = f"A{k + 1:03d}"
        rows.append({"item_id": item, "prompt": g["prompt"], "response": g["response"], "manual_label": "", "notes": ""})
        key.append({"item_id": item, "xstest_id": i, "policy": p, "benchmark_class": g["benchmark_class"], "type": g["type"]})
    save_csv(pd.DataFrame(rows), sheet_p)
    save_csv(pd.DataFrame(key), key_p)
    save_json(dirs["audit"] / "audit_meta.json", {"mode": args.mode, "n_ids": len(ids), "n_items": len(items),
                                                  "policies": policies, "seed": int(cfg["seed"]),
                                                  "id_rule": "released fixed_audit_ids (30 SAFE + 30 UNSAFE)",
                                                  "blinding": "sheet shows prompt + response only, shuffled; key kept separately"})
    print(f"Wrote blind audit sheet with {len(items)} items ({args.mode}) -> {sheet_p}")
    print("Label without viewing AI labels first (notebook labeller, or fill the manual_label column of the CSV).")


if __name__ == "__main__":
    main()
