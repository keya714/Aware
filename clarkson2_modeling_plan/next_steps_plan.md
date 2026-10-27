# Clarkson II Checkpoint 1 Attention-Model Coding and Results Plan

## 1. Dataset to use

Use:

`clarkson2_preprocessing/outputs/final_clean_dataset`

This contains **94 of 100 users**, or **94% user coverage**, with 79,285 fixed-size windows and 322,969 generated verification trials. Five users produced no usable windows and user `214875` had too few windows; exclude them from the primary experiment.

The current cleaned subset uses a 5-second hold-time threshold and retains 5,498,480 of 10,425,149 candidate nonnegative press/release pairs. Keep this preprocessing configuration fixed for Checkpoint 1.

## 2. Modeling choice

Use a lightweight attention-based sequence model as the primary model.

The current `windows_normalized.csv` contains summary features. A self-attention model should instead receive the 50 paired events in each window as a sequence. Create sequence inputs from:

`clarkson2_preprocessing/outputs/final_clean_dataset/clean_events`

Use timing-only event features initially:

- hold time;
- press-to-press time;
- release-to-press time;
- inter-event time.

Do not use key identity in the first model so that the model learns typing rhythm rather than text content.

Each input should have shape:

```text
[batch_size, 50 event pairs, 4 timing features]
```

Use masking for incomplete sequences. Keep the fixed 50-pair window size consistent with the current preprocessing.

The attention encoder should be:

```text
event features
-> Linear projection to d_model=64
-> positional encoding
-> 1-2 Transformer encoder layers
-> 4 attention heads
-> masked mean or attention pooling
-> 32-dimensional user embedding
```

Train the encoder using a metric-learning objective so that windows from the same user are close and windows from different users are separated. Use the resulting embedding distance for verification.

Retain the nearest-centroid model as a simple comparison baseline. The attention model is the primary model, but the comparison baseline is important for showing whether attention provides a measurable improvement.

## 3. Code structure

Create:

```text
src/
  load_windows.py
  make_trials.py
  metrics.py
  baseline.py
  attention_model.py
  evaluate.py

experiments/
  checkpoint1_baseline.py

results/
figures/
```

The code must load the generated files, train the baseline, evaluate the trials, and save metrics and figures automatically.

## 4. Data loader

Load:

`clarkson2_preprocessing/outputs/final_clean_dataset/windows_normalized.csv`

The loader must:

- select the normalized `z_` timing features;
- preserve `user_id`, `window_id`, and `split`;
- reject missing or non-finite features;
- print users and windows loaded;
- print train, adaptation, and test counts;
- use a fixed feature-column list.

Load verification trials from:

`clarkson2_preprocessing/outputs/final_clean_dataset/verification_trials.csv`

Join trials to windows using `window_id`.

## 5. Sequence construction and exploratory analysis

Create a sequence loader that:

- reads clean event files;
- groups valid pairs into the existing 50-pair windows;
- extracts the four timing features per event pair;
- applies the training-only feature scaler;
- preserves `user_id`, `window_id`, `segment_id`, `burst_id`, and `split`;
- returns tensors and masks;
- verifies that no sequence crosses a burst, repair segment, or chronological split.

Save a sequence-quality table containing sequence counts, missing values, and sequence lengths.

Generate and save figures for:

1. Windows per user.
2. Windows in train, adaptation, and test splits.
3. Histograms of timing features.
4. Train/adaptation/test feature box plots.
5. Correlation matrix of normalized features.
6. Genuine versus impostor trial counts.
7. Genuine versus impostor score distributions after scoring.

Each figure needs a title, labeled axes, and deterministic output.

## 6. Comparison baseline

For each user:

1. Select that user's training windows.
2. Calculate the mean feature vector.
3. Store it as the enrollment prototype.
4. Score a sample by Euclidean or cosine distance to the prototype.

Decision rule:

```text
distance <= threshold -> accept
distance > threshold  -> reject
```

This is the non-attention comparison required before interpreting the attention model's results.

## 7. Attention-model training

Train the attention encoder only on training windows.

Use user-balanced batches so users with many windows do not dominate training. A practical batch should contain several users and multiple windows per user.

Recommended training objective:

1. Encode two training windows from the same user as a positive pair.
2. Encode windows from different users as negative pairs.
3. Use supervised contrastive loss, triplet loss, or a prototypical loss.
4. Save the best model using a validation split created only from the training period.

Save:

- model weights;
- scaler;
- feature list;
- model configuration;
- training seed;
- loss curve;
- validation metrics.

Do not use adaptation or final-test windows during model training.

## 8. Attention-model verification and threshold calibration

Do not choose the threshold using the final test set.

Use training data only:

1. Split training trials into fit and calibration subsets using a fixed seed.
2. Generate genuine and impostor calibration scores.
3. Search thresholds across the score range.
4. Select the threshold closest to equal FAR and FRR, or choose a low-FAR operating point.
5. Save the threshold in `results/baseline_config.json`.

For each user, calculate an enrollment embedding prototype from training windows. Score a query window using cosine or Euclidean distance to the enrolled user's prototype.

Do not choose the threshold using the final test set.

## 9. Required results

Evaluate both the nearest-centroid comparison model and the attention model separately on training/calibration, adaptation, and final test trials without updating the model.

Save in `results/baseline_metrics.csv`:

- total trials;
- genuine and impostor counts;
- true accepts and true rejects;
- false accepts and false rejects;
- FAR;
- FRR;
- EER;
- ROC-AUC;
- selected threshold.

Also save:

- ROC curve;
- FAR/FRR versus threshold curve;
- genuine/impostor score histogram;
- confusion matrix.

## 10. Error analysis

Save incorrectly classified trials with:

- enrolled user;
- sample user;
- window ID;
- split;
- label;
- distance/score;
- predicted decision.

Calculate FAR and FRR per user, identify the worst users, and compare errors across chronological splits.

## 11. Minimal longitudinal result

Run this fixed protocol:

```text
Enroll on training windows
Score adaptation windows without updating
Score test windows without updating
```

Produce a table and plot of FAR, FRR, and EER for each split. This is the first direct test of the AWARe motivation: whether a static authenticator degrades on later behavior.

## 12. Reproducibility requirements

Save the:

- random seed;
- feature list;
- data paths;
- split rules;
- threshold-calibration method;
- model hyperparameters;
- attention architecture and loss;
- impostor-sampling policy.

The complete baseline should run from one command:

```text
python experiments/checkpoint1_baseline.py
```

## 13. Completion criteria

Checkpoint 1 coding is complete when:

- the loader runs from a clean environment;
- EDA figures are generated automatically;
- the nearest-centroid comparison runs end to end;
- the attention sequence loader runs end to end;
- the attention model trains only on training windows;
- the threshold is calibrated without final-test data;
- FAR, FRR, EER, and ROC-AUC are computed;
- genuine/impostor score distributions are plotted;
- errors are saved for analysis;
- train, adaptation, and test results are reported separately;
- both models are reproducible from the repository.

## 14. Work order

1. Build event-level sequences from the cleaned event files.
2. Verify sequence counts, masks, and chronological boundaries.
3. Generate EDA tables and figures.
4. Implement the nearest-centroid comparison model.
5. Implement the attention encoder and metric-learning loss.
6. Train using training windows only.
7. Calibrate the embedding-distance threshold on training data.
8. Evaluate both models on train, adaptation, and test splits.
9. Save metrics, curves, score plots, and error CSVs.
10. Inspect longitudinal degradation.
