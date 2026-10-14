# Task 1 - Direct Preference Optimization

All commands are run from the repository root after `python -m scripts.download_assets`
and `python -m scripts.validate_assets`. Every script is resumable: re-running a command
continues from the last checkpoint / skips finished stages.

Default locations (override with `--work-dir` / `--results-dir`):
- `outputs/task1_dpo/` - LoRA adapters, resume checkpoints, reference log-prob cache (not committed)
- `results/task1_dpo/` - machine-readable logs, metrics, generations, tables, figures (committed)

```bash
# 0. validate the corrected objective and log-prob plumbing
python -m task1_dpo.validate --config configs/dpo.yaml

# 1. standard DPO (one epoch, beta = 0.10) + common evaluation; SFT reference row
python -m task1_dpo.train    --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter none --name sft

# 2. beta in {0.03, 0.10, 0.30}: short forks (first 600 pairs) from the original initialisation
python -m task1_dpo.ablate_beta --config configs/dpo.yaml

# 3. length-confounding study (dataset statistics, length-balanced model, stratified eval)
python -m task1_dpo.analyze_length --config configs/dpo.yaml

# 4. tables / figures / qualitative candidates from the saved results
python -m task1_dpo.summarize --config configs/dpo.yaml
```

## Implementation notes
- **Objective correction** (`task1_dpo/dpo.py`): the released starter used
  `beta * (policy_margin + ref_margin)`; the DPO objective needs
  `beta * (policy_margin - ref_margin)`. `task1_dpo.validate` checks the implementation
  against an independent float64 version of the manual's equation.
- Sequence log-probabilities are summed over response tokens only (course encoder
  `common.data.encode_prompt_response`). `utils.sequence_logps` is numerically equivalent
  to `common.generation.response_sequence_logprobs` but only materialises logits for the
  response positions (`logits_to_keep`); equivalence is checked in `validate`.
- The reference policy is the frozen base model (adapter disabled). Its log-probs are
  computed once per data file and cached (`outputs/task1_dpo/cache/`).
- Data order: a seeded permutation (seed from `configs/base.yaml`) saved to
  `results/task1_dpo/train/<run>/data_order.json`; constant learning rate, AdamW,
  batch 2 x 8 accumulation, grad-norm clip 1.0, fp16 base with fp32 LoRA weights and
  dynamic loss scaling.
- Generation metrics use a fixed held-out prompt subset (`results/task1_dpo/gen_prompt_ids.json`:
  first 200 held-out prompts whose rendered prompt is <= 512 tokens), the released
  sampling configuration, 256 new tokens and a fixed seed per batch. KL is the sampled-response
  estimator (token-averaged as in `common.metrics.sampled_kl`; the sequence-summed convention is
  also reported). Reward = fixed course reward model.
- Word-limit compliance: 10 common prompts x 4 samples, `common.metrics.word_limit_compliance`.

## Attribution
Uses the course starter utilities in `common/` and Hugging Face `transformers` / `peft`.
<!-- State any other materially reused code / coding assistance here, as required by the course policy. -->
