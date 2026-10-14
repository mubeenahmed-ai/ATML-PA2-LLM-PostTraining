# Task 4 - Safety calibration (XSTest)

Run from the repository root after the Task 1-3 standard runs. Every step is resumable.
Default results folder: `results/task4_safety/` (override with `--results-dir`). The DPO / PPO / GRPO
adapters default to `configs/feedback.yaml: policies` (override with `--policies "dpo=...,ppo=...,grpo=..."`).

```bash
python -m task4_safety.validate --config configs/feedback.yaml             # parser, statistics, audit ids, pairing, policy provenance
python -m task4_safety.generate_responses --config configs/feedback.yaml   # Step 1: greedy responses, 4 fixed policies x 450 prompts
python -m task4_safety.judge_responses --config configs/feedback.yaml      # Step 2: fixed AI judge (released prompt + parser)
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml     # Step 3: blind audit sheet for the fixed 60 ids
#   ... label manual_audit/audit_sheet.csv (manual_label column) or with the notebook labeller ...
python -m task4_safety.evaluate_safety --config configs/feedback.yaml \
       --task-results "task1=results/task1_dpo,task2=results/task2_ppo,task3=results/task3_grpo"
```

## Implementation notes
- Policies are fixed by the manual: SFT = untouched Qwen2.5-1.5B-Instruct; DPO / PPO / GRPO = the
  standard Task 1-3 adapters. `generate_responses` and `validate` read each adapter's `DONE.json` and
  refuse forks / ablations (wrong run name, beta, update count, loss type or a training subset).
- Generation: one user turn with the base chat template, greedy decoding (`do_sample=False`),
  `safety_max_new_tokens` (256), identical batch size for every policy, fixed `xstest_id` order.
- Judge: the released `load_judge`, `JUDGE_PROMPT`, `parse_json` and `judge_one` are unchanged. Items are
  judged in left-padded batches with the same prompt, greedy decoding and parser; `batch_consistency.json`
  records agreement with the single-example `judge_one` on the first items (`--batch-size 1` uses
  `judge_one` for everything). Parse failures map to AMBIGUOUS, as in the released parser, and are
  counted separately.
- Rates use the benchmark class as prompt-level truth (see `evaluate_safety.py` docstring), with
  Wilson 95% intervals. Labels contradicting the benchmark class are reported as cross-class.
- Contrast pairs: each safe XSTest prompt is paired with its unsafe contrast (same sensitive word,
  harmful intent) by the shared `focus` word when unique, otherwise by position in the category block;
  categories without a one-to-one contrast are not paired. `contrast_pairs_used.csv` lists every pair.
- A keyword refusal heuristic (refusal phrases at the start of the response) is a judge-independent
  cross-check only.
- Manual audit: the released `fixed_audit_ids` (30 SAFE + 30 UNSAFE). Default mode `all_policies` labels
  each audited prompt for all four policies (240 items); `--mode one_per_id` gives 60 items. The sheet
  shows prompt + response only, shuffled; the key (policy, class) is kept separately. Labels were
  assigned by hand without viewing the AI labels.

## Attribution
Uses the course starter utilities in `common/`, the released Task 4 judge code, Hugging Face `transformers` / `peft`.
<!-- State any other materially reused code / coding assistance here, as required by the course policy. -->
