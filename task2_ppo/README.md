# Task 2 - Proximal Policy Optimization

Run from the repository root after `python -m scripts.download_assets` and
`python -m scripts.validate_assets`. Every command is resumable (re-run it after a crash);
finished runs (`DONE.json`) and finished evaluation stages are skipped.

Default locations (override with `--work-dir` / `--results-dir`):
- `outputs/task2_ppo/` - policy adapters, critic states (`value_trainable.pt`), resume checkpoints (not committed)
- `results/task2_ppo/` - per-update logs, held-out generations/metrics, tables, figures (committed)

```bash
python -m task2_ppo.validate --config configs/ppo.yaml                       # objective + plumbing checks

# Step 1: 20-update continuation from the supplied midpoint + held-out evaluation
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter midpoint --name midpoint
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard

# Step 2: cached-rollout clipping geometry + matched eps forks (8 updates each) + evaluation
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml

# Step 3: KL-pressure forks (8 updates each) + evaluation
python -m task2_ppo.ablate_kl --config configs/ppo.yaml

python -m task2_ppo.summarize --config configs/ppo.yaml                      # tables / figures / qualitative
```

## Implementation notes
- **Objective correction** (`task2_ppo/ppo.py`): the starter combined the two surrogates with
  `torch.maximum` (optimistic bound, no effective clipping); the clipped surrogate is the
  elementwise `torch.minimum`. `validate` checks it against the manual's equation and checks that
  the gradient vanishes exactly where the clip binds.
- One update = `prompts_per_update` prompt(s) -> sampled response(s) with the released sampling
  config (cap `max_response_length`) -> old / reference log-probs, reward-model score minus
  `missing_eos_penalty` for responses without EOS, KL-shaped token rewards
  (`shaped_rewards`, beta_KL), critic values, GAE (gamma, lambda), advantage whitening ->
  `ppo_epochs` passes of the clipped policy loss and `value_coef` x critic MSE.
- Every run (standard and all forks) starts from the identical supplied policy and critic,
  uses the same fixed prompt sequence (`results/task2_ppo/train_prompt_order.json`: prompts whose
  rendered length fits `max_prompt_length`, permuted with the config seed) and the same sampling
  seed per update index. The (eps 0.20, beta 0.10) fork is shared by the clipping and KL studies.
- Dropout is disabled in policy and critic so that the first-epoch ratio is exactly 1.
- The critic is the supplied checkpoint wrapped with a fresh zero-initialised LoRA plus a
  trainable score head (`common.models.load_value_model(train_mode="lora_head")`); trainable
  parameters and the score head are kept in fp32. `V(s_t)` is read at the last prefix token.
- Log-probs use raw logits (temperature 1), as `common.generation.response_token_logprobs`.
- Logged per update: learned reward, KL (token mean via `common.metrics.sampled_kl`, and sequence
  sum), policy / value loss, full-distribution entropy, policy / value gradient norms, clip fraction
  (ratio outside the band before clipping), clip-binding fraction, response length, critic
  explained variance, per-update policy-shift KL, wall-clock and peak VRAM.
- Held-out evaluation: fixed `rl_prompt_eval` subset (`results/task2_ppo/eval_prompt_ids.json`),
  `eval_max_response_length` cap, released sampling config, fixed seed per batch.

## Attribution
Uses the course starter utilities in `common/` and Hugging Face `transformers` / `peft`.
<!-- State any other materially reused code / coding assistance here, as required by the course policy. -->
