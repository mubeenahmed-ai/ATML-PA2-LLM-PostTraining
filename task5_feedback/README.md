# Task 5 - Feedback source: RLVR versus RLAIF

Run from the repository root after `python -m scripts.download_assets` / `python -m scripts.validate_assets`.
Default results folder: `results/task5_feedback/` (override with `--results-dir`). Every step is resumable.

```bash
python -m task5_feedback.validate --config configs/feedback.yaml                        # verifier, data, conventions, adapters
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm      # Step 1: in-domain (generate + judge + metrics)
python -m task5_feedback.score_perturbations --config configs/feedback.yaml             # Step 2: controlled reward diagnostics
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer # Step 3: SVAMP transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml                # tables / figures / qualitative
```

## Implementation notes
- The released verifier (`rlvr.py`) and pairwise judge (`rlaif.py`) are used unchanged. `validate` confirms
  the verifier reproduces `expected_exact_reward` on every diagnostic row.
- Policies: SFT = untouched base; RLVR / RLAIF = the supplied adapters (`checkpoints/rlvr_policy`,
  `checkpoints/rlaif_policy`). Greedy decoding, `math_max_new_tokens` (512), the dataset's own prompt
  (`messages`), identical batch size for all three policies.
- Metrics: exact accuracy (Wilson 95% CI), format compliance (designated final parsed), response tokens,
  truncation, failure types (correct / wrong final / unparseable `####` / no marker / truncated without final),
  and accuracy under two looser parsers (last number; any number equal to gold) for the verifier-specification
  analysis.
- AI pairwise protocol: the released `PairwiseAIJudge.compare` (deterministic hash-based A/B orientation
  balancing, persistent cache) on the problem text and the two responses; win 1 / tie 0.5 / loss 0.
  Identical responses are scored as a tie without a judge call (a call could only reflect position bias).
  Ties are split into explicit `TIE` outputs and unparseable outputs (which the released parser maps to TIE)
  by re-running the identical presentation. Verifier-judge agreement compares the judge's choice with the
  verifier's preference on the same pair.
- Diagnostics: RLVR reward = exact verifier; RLAIF reward = released `group_rewards` over the five variants
  (direct-RLAIF group reward, K = 5) and, for the controlled pairs, the direct pairwise judgment. Pairs
  (better first): reasoning (clean vs corrupted reasoning), outcome (clean vs wrong final), filler (clean vs
  clean + persuasive filler), distractor (clean vs gold-distractor wrong final). S_reason / S_outcome follow the
  manual. Judge position consistency: every controlled pair is also presented in both fixed orders.
- The judge cache is backed up atomically; a cache corrupted by a disconnect is restored automatically.

## Attribution
Uses the course starter utilities in `common/`, the released Task 5 verifier and judge, Hugging Face `transformers` / `peft`.
<!-- State any other materially reused code / coding assistance here, as required by the course policy. -->
