<!-- PA2_REPRODUCE_START -->
## Reproducing every result
All commands run from the repository root after `python -m pip install -r requirements.txt`, `python -m scripts.download_assets` and `python -m scripts.validate_assets`. Every script is resumable; results are written to `results/<task>/` (committed) and adapters / checkpoints to `outputs/<task>/` (ignored). Each task folder's README documents implementation details.

### Task 1 - DPO  ([details](task1_dpo/README.md))
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

### Task 2 - PPO  ([details](task2_ppo/README.md))
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

### Task 3 - GRPO  ([details](task3_grpo/README.md))
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

### Task 4 - Safety calibration  ([details](task4_safety/README.md))
```bash
python -m task4_safety.validate --config configs/feedback.yaml             # parser, statistics, audit ids, pairing, policy provenance
python -m task4_safety.generate_responses --config configs/feedback.yaml   # Step 1: greedy responses, 4 fixed policies x 450 prompts
python -m task4_safety.judge_responses --config configs/feedback.yaml      # Step 2: fixed AI judge (released prompt + parser)
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml     # Step 3: blind audit sheet for the fixed 60 ids
#   ... label manual_audit/audit_sheet.csv (manual_label column) or with the notebook labeller ...
python -m task4_safety.evaluate_safety --config configs/feedback.yaml \
       --task-results "task1=results/task1_dpo,task2=results/task2_ppo,task3=results/task3_grpo"
```

### Task 5 - RLVR vs RLAIF  ([details](task5_feedback/README.md))
```bash
python -m task5_feedback.validate --config configs/feedback.yaml                        # verifier, data, conventions, adapters
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm      # Step 1: in-domain (generate + judge + metrics)
python -m task5_feedback.score_perturbations --config configs/feedback.yaml             # Step 2: controlled reward diagnostics
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer # Step 3: SVAMP transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml                # tables / figures / qualitative
```

### Task 6 - Synthesis
No new runs; `results/task6_synthesis/` holds the compact cross-task summary built from the files above.

<!-- PA2_REPRODUCE_END -->

---

# ATML PA2 - LLM Post-Training

<!-- FINAL_STUDENT_SETUP -->

## Quick start

```bash
git clone https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
```

The fixed datasets, cached diagnostics, and supplied
continuation checkpoints are downloaded from:

https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets

Pinned release revision:

`0b350481fb03f5525a35bcdec4131bd4fe487f98`

---
# ATML PA2 - LLM Post-Training

This is the **student starter repository** for ATML PA2. The released code is intentionally incomplete: Tasks 1-3 provide model/data loading, objective helpers, checkpoint restoration, and experiment entry points, but **you must implement the training loops and ablation orchestration yourself**. Each of Tasks 1-3 also contains one deliberate algorithmic defect in its core objective code; identifying and correcting these defects is part of validating your implementation.

Task 4 supplies the fixed AI safety judge and response-generation utilities, but you must write the evaluation/aggregation code. Task 5 supplies the exact RLVR verifier, the fixed pairwise AI judge used for RLAIF evaluation, and data/model loaders; you must implement the requested evaluation and analysis.

## 1. Clone and install

```bash
git clone https://github.com/COURSE_ORG/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
```

## 2. Download the course assets

The large course-created checkpoints and fixed data are distributed as a GitHub Release asset rather than normal Git files. After cloning, run:

```bash
python -m scripts.download_assets
python -m scripts.validate_assets
```

If your instructor provides a direct asset URL separately, use:

```bash
python -m scripts.download_assets --url '<ASSET_URL>'
```

Public base/reward/judge models are downloaded from Hugging Face at runtime and are **not** included in the course asset archive.

The installer also materializes the fixed 100-example Task 5 transfer set from the official SVAMP challenge-set source if it is not already present. The tiny Task 1 word-limit prompt set is tracked directly in this repository.

## 3. Environment check

```bash
python -m scripts.check_environment
```

Run commands from the repository root. The reference environment used to prepare the release pins Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, and Tokenizers 0.22.1.

## 4. Supplied course checkpoints

After `download_assets`, these directories should exist:

```text
checkpoints/ppo_midpoint_policy/
checkpoints/ppo_midpoint_value/
checkpoints/grpo_midpoint_policy/
checkpoints/rlvr_policy/
checkpoints/rlaif_policy/
```

PPO and GRPO begin from the supplied continuation checkpoints. RLVR and RLAIF are supplied frozen evaluation policies; students do not retrain them.

The PPO value checkpoint is intentionally released as the exact staff midpoint state, including its imperfect held-out value calibration. Treat critic behavior as an analysis variable rather than assuming a perfect baseline, and start every PPO fork from the identical supplied policy/value state. The default continuation generation cap is 512 tokens for feasibility; frozen evaluation uses the larger cap specified in `configs/ppo.yaml`.

## 5. Task entry points

### Task 1 - DPO

```bash
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml
python -m task1_dpo.analyze_length --config configs/dpo.yaml
```

### Task 2 - PPO

```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
python -m task2_ppo.ablate_kl --config configs/ppo.yaml
```

### Task 3 - GRPO

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml
```

### Task 4 - Safety calibration

The judge loader/parser are supplied. You must implement the requested generation aggregation and evaluation.

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

### Task 5 - RLVR vs RLAIF

The exact verifier and pairwise AI judge are supplied; you implement the evaluation/analysis.

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

## 6. Reproducibility rules

- Do not alter course-provided data, cached rollouts, or supplied checkpoints.
- Start every short fork from the **same supplied midpoint checkpoint**.
- Keep prompt IDs, generated-token/update budgets, seed, and evaluation procedure matched across ablations.
- Commit your code, configs, small JSON/CSV logs, and figures. Do not commit downloaded checkpoints, raw course assets, or model caches.
- Record peak VRAM and wall-clock time for the standard PPO and GRPO continuations.

See the assignment manual for the required experiments, metrics, and report questions.
