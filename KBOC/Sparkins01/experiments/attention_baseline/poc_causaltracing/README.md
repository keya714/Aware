# Causal-tracing proof of concept

This directory is intentionally separate from the original Week 1 attention
baseline. It will contain the ROME-inspired causal-localization and targeted
editing experiment for one user.

## Step 1: target user

The target is fixed to participant `1`, with every other participant assigned
label `0`. This agrees with the existing baseline checkpoint in
`../outputs/config.json`.

Run the selector from the repository root:

```powershell
python experiments/attention_baseline/poc_causaltracing/select_target_user.py
```

The Week 8 file is downloaded from the public dataset repository:
<https://github.com/sparkins01/keystroke>. Run:

```powershell
python experiments/attention_baseline/poc_causaltracing/select_target_user.py `
  --week8-data data/week_8.csv.gz
```

The selection record is written to `outputs/target_user.json` in this folder.

## Current evidence

- Selected user: `1`
- Week 1 usable attempts: `159`
- Week 8 usable attempts: `160`
- Existing Week 1 attention baseline test ROC-AUC: `0.9216`
- Existing Week 1 attention baseline test recall: `0.6250`
- Week 8 degradation and false-negative counts: evaluated in the next PoC step

Step 2 reproduces the saved baseline metrics on the held-out Week 1 test
split: EER `0.1636`, FAR `0.0727`, FRR `0.3750`, recall `0.6250`, and F1
`0.1695`. The 24 genuine-user probabilities are stored in
`outputs/week1_baseline.json`.

The selector does not fabricate Week 8 results. Later steps should consume the
recorded target ID and add Week 8 evaluation, causal localization, targeted
editing, and unrestricted-fine-tuning comparisons.

## Step 2: Week 1 baseline

Evaluate the frozen checkpoint and reconstruct its original held-out split:

```powershell
python experiments/attention_baseline/poc_causaltracing/evaluate_week1_baseline.py
```

Results, including the genuine-user probability distribution, are written to
`outputs/week1_baseline.json`.

## Step 3: frozen Week 8 evaluation

Run the unchanged Week 1 checkpoint directly on Week 8:

```powershell
python experiments/attention_baseline/poc_causaltracing/evaluate_week8_frozen.py
```

The comparison is written to `outputs/week8_frozen.json`. Target-user genuine
samples rejected at the 0.5 threshold are written to
`outputs/week8_false_rejections.csv`.

## All-user Week 8 diagnostic archive

Because this checkpoint is trained only for user `1` versus all others, the
all-user run reports cohort score distributions rather than pretending there
is a separate authentication model for every participant:

```powershell
python experiments/attention_baseline/poc_causaltracing/evaluate_week8_all_users.py
```

Results are stored separately in `outputs/week8_all_users/`, including all
sample scores, a per-user summary, and target-user false rejections.

The frozen run produced 56 false rejections among 160 Week 8 genuine samples.
ROC-AUC changed from `0.9216` to `0.9188`, and EER changed from `0.1636` to
`0.1747`. At the fixed 0.5 threshold, recall changed from `0.6250` to
`0.6500`; this is not a broad recall degradation, but the false-rejection
samples are available for the causal investigation.
