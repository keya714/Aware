# Clarkson II Checkpoint 1 — Documentation Handoff

This file is a report-ready summary of the work completed so far. It is intended to help the team convert the implementation, tables, and figures into the Checkpoint 1 project report.

## 1. Research objective

The project studies whether a behavioral biometric authenticator remains reliable when a user's typing behavior changes over time. The working hypothesis is that a model trained on an initial 30-day period may experience longitudinal drift: genuine samples from the same user become farther from the user's enrollment representation, increasing false rejection rate (FRR).

The current experiment has two stages:

1. Build and evaluate a frozen attention-based biometric model.
2. Localize and attempt to correct user-specific drift using causal tracing and ROME-style rank-one model editing.

The key security requirement is that any recovery of genuine-user authentication must not significantly increase false acceptance rate (FAR).

## 2. Data preprocessing and quality analysis

### 2.1 Raw-data cleaning

The raw keystroke data was processed at the press/release-event level. The cleaning pipeline:

- matched press and release events using the available event identifiers and ordering;
- removed invalid or negative hold-time values;
- removed unmatched press/release events;
- investigated long-hold anomalies rather than silently treating them as normal typing;
- applied a fixed 5-second hold-time threshold;
- retained valid nonnegative press/release pairs;
- generated fixed-size windows containing 50 event pairs;
- preserved user, window, segment, burst, and chronological split information;
- rejected missing and non-finite model features.

The final cleaned dataset retains 5,498,480 valid press/release pairs out of 10,425,149 candidate pairs under the fixed 5-second hold-time configuration. The preprocessing configuration must remain fixed for the checkpoint experiment so that later model comparisons are fair.

### 2.2 Features used by the attention model

The model uses timing-only features and does not use key identity. This isolates typing rhythm from the actual text typed. Each 50-event window contains four features per event pair:

- hold time;
- press-to-press time;
- release-to-press time;
- inter-event time.

Timing values are transformed using `log1p(max(seconds, 0))`, then standardized using training-only statistics. The resulting sequence tensor has shape:

```text
[number of windows, 50 event pairs, 4 timing features]
```

### 2.3 Preprocessing artifacts

- [Cleaned-dataset directory](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_preprocessing/outputs/final_clean_dataset>)
- [Normalized windows](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_preprocessing/outputs/final_clean_dataset/windows_normalized.csv>)
- [Verification trials](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_preprocessing/outputs/final_clean_dataset/verification_trials.csv>)
- [Event-cleaning summary](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_preprocessing/outputs/final_clean_dataset/event_cleaning_summary.json>)
- [Final quality report](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_preprocessing/outputs/final_clean_dataset/final_quality_report.json>)
- [Pipeline configuration](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_preprocessing/outputs/final_clean_dataset/pipeline_config.json>)
- [Training-only scaler](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_preprocessing/outputs/final_clean_dataset/scaler_train_only.json>)
- [Anomaly summary CSV](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_preprocessing/outputs/anomaly_investigation/anomaly_summary.csv>)
- [Anomaly summary JSON](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_preprocessing/outputs/anomaly_investigation/anomaly_summary.json>)
- [Hold-time sensitivity results](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_preprocessing/outputs/hold_threshold_sensitivity>)

The preprocessing notebooks contain the detailed exploratory work:

- `clarkson2_full_cleaning_pipeline.ipynb`
- `clarkson2_anomaly_investigation.ipynb`
- `clarkson2_hold_threshold_sensitivity.ipynb`
- `clarkson2_hold_threshold_visualization.ipynb`
- `clarkson2_event_features.ipynb`

## 3. First-30-day baseline model

### 3.1 Enrollment and model architecture

The first 30 days are used as the enrollment/training period. The attention model is a lightweight Transformer-style sequence encoder:

```text
4 timing features
→ linear projection to d_model=64
→ positional encoding
→ Transformer encoder layers with four attention heads
→ pooling
→ 32-dimensional user embedding
```

The model is trained with a metric-learning objective so that windows from the same user are closer in embedding space than windows from different users. Each user receives an enrollment prototype calculated from first-30-day training windows.

Verification uses embedding distance to the enrollment prototype. A fixed threshold is calibrated without using the later held-out stages. A query is accepted when its distance is at or below the threshold and rejected otherwise.

### 3.2 Baseline model results

The saved attention model contains 89 users and 41,895 sequence windows with shape `(41895, 50, 4)`. The best saved model was selected at epoch 32 based on validation EER; the epoch number describes ordinary model training and is separate from the later ROME value-vector optimization steps.

Saved baseline artifacts:

- [Best attention model](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/attention_model_best.pt>)
- [Attention metrics and training history](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/attention_metrics.json>)
- [Sequence build report](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/sequence_build_report.json>)
- [Sequence metadata](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/sequence_metadata.csv>)
- [Saved sequence array](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/sequences.npz>)

The best validation EER was approximately 13.70%. On the saved verification evaluation, the attention model produced approximately 13.98% FRR and 13.97% FAR on the test split at the fixed threshold.

## 4. The `clarkson2_30day_model` longitudinal experiment

This folder is the main working directory for the longitudinal analysis. It uses the first 30 days to train/enroll the frozen model, then evaluates later behavior without updating the model.

### 4.1 Later-stage construction

All data after the first 30 days is divided chronologically into three equal per-user windows:

- Stage 1: earliest later behavior;
- Stage 2: middle later behavior and primary held-out test;
- Stage 3: latest later behavior and secondary held-out test.

The remaining-stage data contains 37,391 later windows from 82 users, while 89 users have usable first-30-day enrollment data. Users are retained in the longitudinal analysis only when they have enough windows for the relevant analysis.

Artifacts:

- [Remaining-stage metadata](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/remaining_stage_metadata.csv>)
- [Remaining-stage metrics](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/remaining_stage_metrics.csv>)
- [Remaining-stage summary](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/remaining_stage_summary.json>)
- [Remaining-stage sequences](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/remaining_stage_sequences.npz>)

### 4.2 Drift definition

For each user and later-stage window, the embedding distance to the user's first-30-day enrollment prototype is measured. Drift is the increase in mean later-stage distance relative to that user's first-30-day test distance:

```text
drift = later-stage mean distance − first-30-day test mean distance
```

The ranking uses Stage 1 information for targeted selection so that Stage 2 and Stage 3 remain held out. The saved drift analysis ranked 74 users and identified user `163748` as having the largest mean absolute drift. The maximum mean absolute drift was approximately 0.730 embedding-distance units.

Drift artifacts:

- [Per-user drift metrics](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/user_drift_metrics.csv>)
- [Overall drift ranking](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/user_drift_overall_ranking.csv>)
- [Drift summary](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/user_drift_summary.json>)

## 5. Existing longitudinal graphs

These figures are useful for the checkpoint report because they show data coverage, drift, and authentication degradation:

- [Training versus later-stage error rates](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/figures/training_vs_later_stage_error_rates.png>)
- [Later-stage EER change](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/figures/later_stage_eer_change.png>)
- [Remaining-stage error rates](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/figures/remaining_stage_error_rates.png>)
- [Remaining-stage window counts](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/figures/remaining_stage_window_counts.png>)
- [Top-user drift](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/figures/top_user_drift.png>)
- [User drift by stage boxplot](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/figures/user_drift_by_stage_boxplot.png>)
- [User drift heatmap](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/figures/user_drift_heatmap.png>)
- [Causal-tracing plot for user 163748](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/figures/causal_trace_user_163748.png>)

Interpretation: the frozen model can perform acceptably during first-30-day evaluation but later behavioral changes increase genuine-user distances and can increase FRR. This supports the AWARe motivation for studying adaptation, but the result must be reported as longitudinal degradation rather than as evidence that every user drifts identically.

## 6. Causal tracing

Causal tracing tests which parts of the attention model carry information that can restore the target user's representation. The method patches one component at a time and measures the resulting embedding distance.

Interventions include:

- input projection;
- Transformer layer 1;
- Transformer layer 2;
- post-encoder normalization;
- embedding projection.

For each intervention, the CSV records baseline distance, patched distance, distance recovery, recovery fraction, and wrong-user control distance. The wrong-user control is important: a patch that lowers all users' distances may indicate generic representation collapse rather than target-specific recovery.

Artifacts:

- [Causal-tracing CSV](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/causal_trace_stage1/causal_trace_stage1.csv>)
- [Causal-tracing summary](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/causal_trace_stage1/summary.json>)
- [Stage-1 drift ranking used by tracing](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/causal_trace_stage1/stage1_drift_ranking.csv>)
- [Recovery heatmap](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/causal_trace_stage1/figures/causal_trace_recovery_heatmap.png>)
- [Patched-distance heatmap](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/causal_trace_stage1/figures/causal_trace_distance_heatmap.png>)
- [Wrong-user control heatmap](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/causal_trace_stage1/figures/causal_trace_control_heatmap.png>)

The causal-tracing conclusion is localization, not final recovery. It identifies a candidate internal location for editing. It does not by itself prove that editing that layer will improve authentication safely.

## 7. ROME-style rank-one model editing

The editing method follows the ROME idea from Meng et al., adapted to the biometric Transformer. The edit is not ordinary full-model fine-tuning.

For each selected user:

1. Compute a key vector `k` from the mean layer-2 feed-forward input activation over that user's Stage 1 adaptation windows.
2. Estimate the key covariance from first-30-day training windows.
3. Optimize only a desired value vector while keeping the original model frozen and preserving a replay buffer.
4. Apply a closed-form rank-one update to the selected layer:

```text
W_new = W + (v − (Wk + b)) (C⁻¹k)^T / ((C⁻¹k)^T k)
```

5. Evaluate the edited model on early first-30-day test data, Stage 2, and Stage 3 using the original enrollment prototypes and fixed threshold.

The edit targets the second Transformer layer's feed-forward `linear2` projection. The reported `v_steps` parameter is the number of optimization steps used to find the value vector; it is not the number of model-training epochs. Each final weight update has measured rank 1.

Success criteria for an individual user:

- Stage 2 FRR reduction of at least 5 percentage points;
- no Stage 3 FRR increase;
- FAR increase no greater than 1 percentage point;
- early-test FRR increase no greater than 2 percentage points;
- measured update rank equal to 1.

## 8. Population ROME results

The generalized population experiment requires at least five windows in each of Stage 1, Stage 2, and Stage 3. Stage 1 is used for drift ranking and edit construction; Stage 2 is the primary held-out test; Stage 3 is secondary confirmation.

Main population artifacts:

- [Eligible users](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/model_editing/eligible_users.csv>)
- [Stage-1 drift ranking](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/model_editing/stage1_drift_ranking.csv>)
- [Per-user ROME metrics](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/model_editing/rome_per_user_metrics.csv>)
- [Population summary](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/model_editing/rome_population_summary.csv>)
- [Bootstrap confidence intervals](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/model_editing/rome_bootstrap_confidence_intervals.csv>)
- [Statistical tests](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/model_editing/rome_statistical_tests.json>)
- [ROME protocol](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/model_editing/protocol.json>)

Main population figures:

- [FRR before/after](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/model_editing/figures/rome_frr_before_after.png>)
- [FAR before/after](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/model_editing/figures/rome_far_before_after.png>)
- [Distance recovery](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/model_editing/figures/rome_distance_recovery.png>)
- [Recovery/security trade-off](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/model_editing/figures/rome_recovery_security_tradeoff.png>)
- [Edit cost and rank](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/model_editing/figures/rome_edit_cost.png>)
- [Population FRR distribution](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/model_editing/figures/rome_population_frr_distribution.png>)
- [Population FAR distribution](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/model_editing/figures/rome_population_far_distribution.png>)
- [Macro-average comparison](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/model_editing/figures/rome_population_macro_frr.png>)
- [Success by user](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/model_editing/figures/rome_success_by_user.png>)
- [Stage-1 drift versus recovery](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/model_editing/figures/rome_stage1_drift_vs_recovery.png>)

## 9. Five-user reproduction comparison

To check whether the optimization-step behavior was reproducible beyond a single-user pilot, the same five Stage-1-ranked users were evaluated with `v_steps=1`, `5`, and `20`, keeping `adaptation_size=64` and `replay_size=64` fixed.

Users: `163748`, `132109`, `562519`, `756382`, and `748549`.

- [Five-user v_steps=1 directory](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/phase1/reproduction_users5_vsteps1>)
- [Five-user v_steps=5 directory](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/phase1/reproduction_users5_vsteps5>)
- [Five-user v_steps=20 directory](<C:/Users/keyar/Documents/GradSem3/Biometric/Term Project/clarkson2_30day_model/results/phase1/reproduction_users5_vsteps20>)

Each directory contains `rome_per_user_metrics.csv`, `rome_population_summary.csv`, `protocol.json`, bootstrap/statistical outputs, figures, and individual ROME checkpoints.

Five-user summary:

| Setting | Stage 2 macro FRR reduction | Stage 2 macro FAR increase | Success rate |
|---|---:|---:|---:|
| `v_steps=1` | 0.78 percentage points | +0.24 percentage points | 0/5 |
| `v_steps=5` | 3.61 percentage points | +1.32 percentage points | 0/5 |
| `v_steps=20` | 6.07 percentage points | +4.25 percentage points | 0/5 |

Interpretation: increasing value-vector optimization improves genuine-user recovery, but the FAR increase also grows and violates the one-percentage-point security constraint. Even when the average Stage 2 FRR reduction exceeds the 5-point target at `v_steps=20`, the method is not considered successful because the security constraint fails. This is the central result to emphasize in the checkpoint discussion.

## 10. Current conclusion for Checkpoint 1

The completed work supports the following cautious conclusion:

> The frozen attention-based biometric model exhibits longitudinal behavioral drift after the first 30 days. Causal tracing localizes a potentially useful internal representation in the Transformer. ROME-style rank-one editing can reduce genuine-user embedding distance and sometimes reduce FRR, but stronger edits also increase impostor acceptance. Under the predefined security constraints, the current ROME configuration does not safely solve the drift problem. Broader adaptation methods, such as constrained fine-tuning or continual metric-learning adaptation, should therefore be investigated next.

Do not claim that ROME never works. The defensible claim is that this implementation and operating point did not satisfy the combined recovery-and-security criteria across the evaluated users. The five-user reproduction strengthens the observation, while the full eligible-user population results should be used for the main population-level statement.

## 11. Suggested report structure

1. **Problem and motivation:** behavioral biometrics drift over time.
2. **Data and preprocessing:** event matching, anomaly handling, 5-second threshold, fixed 50-pair windows, timing-only features.
3. **Baseline attention model:** architecture, training-only normalization, enrollment prototype, threshold calibration.
4. **Longitudinal evaluation:** first 30 days versus Stages 1–3, frozen-model FRR/FAR and embedding-distance drift.
5. **Causal tracing:** intervention design, recovery fraction, target specificity, heatmaps.
6. **ROME editing:** rank-one update equation, selected layer, per-user procedure, held-out evaluation.
7. **Results:** FRR recovery, FAR/security trade-off, rank-one edit cost, five-user reproduction, population results.
8. **Conclusion:** ROME is promising for representation recovery but unsafe under the current FAR constraint.
9. **Next step:** evaluate constrained broader fine-tuning or continual adaptation while preserving held-out Stage 2/3 evaluation.

## 12. Reproducibility commands

From `clarkson2_30day_model`:

```powershell
python causal_trace_stage1.py
python model_editing.py
python model_editing.py --users 5 --v-steps 1 --adaptation-size 64 --replay-size 64 --output-dir results/phase1/reproduction_users5_vsteps1
python model_editing.py --users 5 --v-steps 5 --adaptation-size 64 --replay-size 64 --output-dir results/phase1/reproduction_users5_vsteps5
python model_editing.py --users 5 --v-steps 20 --adaptation-size 64 --replay-size 64 --output-dir results/phase1/reproduction_users5_vsteps20
```

The CSV files are the source of truth for numerical values. Figures should be used to make the results easier to interpret, but every plotted value should be traceable to the corresponding CSV row.
