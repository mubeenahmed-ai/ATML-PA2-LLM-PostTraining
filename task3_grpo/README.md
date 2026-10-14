# Task 3 - Group Relative Policy Optimization

Run from the repository root after `python -m scripts.download_assets` and
`python -m scripts.validate_assets`. Every command is resumable; finished runs (`DONE.json`)
and finished evaluation stages are skipped.

Default locations (override with `--work-dir` / `--results-dir`):
- `outputs/task3_grpo/` - policy adapters, resume checkpoints (not committed)
- `results/task3_grpo/` - per-update and per-completion logs, held-out generations/metrics, tables, figures (committed)

```bash
python -m task3_grpo.validate --config configs/grpo.yaml                      # objective + plumbing checks

# Step 1: 20-update continuation from the supplied midpoint (K = 4) + held-out evaluation
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter midpoint --name midpoint
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard

# Step 2: equal-generation group-size study on the supplied K=8 cache (no training)
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml

# Step 3: canonical GRPO vs Dr. GRPO-style normalisation (8-update forks) + evaluation
python -m task3_grpo.compare_normalization --config configs/grpo.yaml

python -m task3_grpo.summarize --config configs/grpo.yaml                     # tables / figures / qualitative
```

## Implementation notes
- **Objective correction** (`task3_grpo/grpo.py`): the starter normalised rewards with the mean / std
  of the whole batch; the group-relative advantage uses the mean / std of each prompt's own group
  (`group_ids`). Identical-reward groups get advantage 0 (uninformative). Population std, clamped
  at 1e-6, as in the released helper; a group is informative when its std > 1e-6.
- One update = `prompts_per_update` prompt x `num_generations` sampled completions (cap
  `max_completion_length`, released sampling config, fixed seed per update index) -> reward-model
  scores -> group-relative advantages -> completions that hit the cap are masked from the loss
  (`mask_truncated_completions`; they still count in the group statistics) -> `policy_epochs`
  optimizer step(s) on the clipped GRPO objective + beta x k3 KL to the frozen reference.
  With `policy_epochs: 1` the old log-probs equal the detached new ones (ratio = 1, as in TRL).
- The group's completions are processed in micro-batches with exact gradient accumulation
  (`grpo_microbatch_loss`; `validate` checks equality with `grpo_policy_loss`).
- Step 2 keeps the total number of generations fixed: each prompt's 8 cached completions are
  split into 8/K disjoint groups (ordered by `generation_index`); the exact probability over all
  C(8, K) groups is reported as a robustness check. Difficulty = tertiles of the 8-sample mean reward.
- Step 3 changes ONLY the sequence normaliser (1/T_k vs 1/max_completion_length); advantages keep
  the std scaling so the normalisation effect is isolated. Per-completion length, advantage, loss
  weight and measured per-completion policy-gradient norm are logged (`completions_log.jsonl`).
- Held-out evaluation: fixed `rl_prompt_eval` subset (`results/task3_grpo/eval_prompt_ids.json`),
  released sampling config, cap = `cache_generation_cap` (768), fixed seed per batch.
- Checkpoints are single files written locally and copied to Drive (two alternating slots).

## Attribution
Uses the course starter utilities in `common/` and Hugging Face `transformers` / `peft`.
<!-- State any other materially reused code / coding assistance here, as required by the course policy. -->
