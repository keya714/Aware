# Clarkson II 30-Day Model and User-Level Drift Methodology

## Dataset and temporal setup

The first 30 days of each user's recorded activity were used as the model-development period. Events were represented as non-overlapping windows of 50 valid press/release pairs. The model-development dataset contains 41,895 windows from 89 users. All 100 source user files were retained in the preprocessing output; users with no usable 50-pair windows are documented in `user_30day_report.csv`.

For the later evaluation, every timestamp after day 30 was retained. For each user, the remaining time interval was divided into three equal chronological intervals:

- `stage_1`: earliest remaining interval;
- `stage_2`: middle remaining interval;
- `stage_3`: latest remaining interval.

These stages are user-specific time intervals, not common calendar periods. The frozen model was evaluated on 37,391 later-stage windows from 82 users.

## Attention model and event representation

Each 50-pair window contains four timing features for every event pair:

1. hold time;
2. press-to-press time;
3. release-to-press time;
4. inter-event time.

The timing values were transformed using:

```text
log1p(max(timing_seconds, 0))
```

The attention model is a small Transformer encoder:

```text
4 timing features
→ linear projection to 64 dimensions
→ learned positional encoding
→ two Transformer encoder layers with four attention heads
→ mean pooling over the 50 events
→ 32-dimensional normalized embedding
```

The model was trained as a user-classification model on the first-30-day training split. The classifier was used during training, but drift analysis uses the 32-dimensional normalized embeddings rather than the classifier logits.

Feature normalization was fitted only on the first-30-day training windows. The same mean and standard deviation were applied to first-30-day test windows and all later-stage windows. This prevents later data from influencing preprocessing.

The best checkpoint was selected using adaptation-set EER with early stopping. It is stored as:

```text
clarkson2_30day_model/results/attention_model_best.pt
```

## Enrollment embedding

For each user, all embeddings from that user's first-30-day training windows were averaged:

```text
user_prototype = mean(training_window_embeddings)
```

This prototype represents the user's initial typing profile. It is fixed during later-stage evaluation; the model is not retrained or adapted before measuring drift.

## User-level drift calculation

For a window embedding `z` and the user's enrollment prototype `p`, the distance is the Euclidean distance:

```text
distance(z, p) = ||z - p||₂
```

For each user, the baseline distance was calculated from that user's first-30-day test windows:

```text
baseline_mean_distance
```

For every later stage, the mean distance from that stage's windows to the same enrollment prototype was calculated:

```text
stage_mean_distance
```

Absolute drift was defined as:

```text
absolute_drift = stage_mean_distance - baseline_mean_distance
```

Relative drift was defined as:

```text
relative_drift = (stage_mean_distance / baseline_mean_distance) - 1
```

Positive drift means that later windows moved farther from the user's original typing profile. Larger values indicate greater embedding-space degradation relative to that user's own first-30-day baseline.

Users were included in the ranking only when a stage contained at least five windows. The overall user ranking used the mean absolute drift across all available later stages for that user.

## Threshold-based user error measure

The first-30-day test threshold was retained as a fixed reference. It was approximately:

```text
1.1886 embedding-distance units
```

For each later-stage window, a rejection was counted when its distance from the user's prototype exceeded this threshold:

```text
stage_FRR = number of rejected genuine windows / number of stage windows
```

This provides a user-level operational measure in addition to the continuous embedding-distance drift.

## Visualization calculations

### Top-user drift bar chart

The 20 users with the largest mean absolute drift were selected after applying the five-window minimum. Each bar represents:

```text
mean absolute drift across available later stages
```

### User-stage heatmap

The heatmap contains one row per high-drift user and one column per later stage. Each cell is the user's absolute drift for that stage. Red indicates larger positive drift; blue indicates lower or negative change.

### Stage drift boxplot

The boxplot shows the distribution of user-level absolute drift separately for `stage_1`, `stage_2`, and `stage_3`. It summarizes variation across users rather than treating every window as an independent user.

### Training-versus-later error-rate chart

The comparison chart uses EER, FAR, and FRR from:

- first-30-day adaptation evaluation;
- first-30-day test evaluation;
- later Stage 1;
- later Stage 2;
- later Stage 3.

All rates are displayed as percentages. The same frozen model and embedding system are used for the later stages.

## Main result

The largest overall mean drift was observed for user `163748`:

- mean absolute drift: `+0.7299` distance units;
- mean relative drift: approximately `+236.8%`;
- Stage 2 drift: `+0.8672`;
- Stage 3 drift: `+0.8223`.

This ranking identifies users for causal tracing. It does not establish why their embeddings drifted. Causal tracing must next test which internal Transformer representations can be intervened on to recover the original verification behavior.

## Causal tracing for user 163748

Activation patching was performed on the highest-drift user, `163748`, using its Stage 2 windows. The unpatched Stage 2 mean distance from the user's first-30-day enrollment prototype was `1.1755`.

For each intervention, the activation at one model component was replaced with the mean activation produced by user `163748`'s first-30-day training windows. The patched Stage 2 embedding was then compared with the same enrollment prototype. A wrong-user activation control used user `118630`'s first-30-day activations.

| Intervention | Patched distance | Distance recovery |
|---|---:|---:|
| Input projection | 0.9548 | 18.8% |
| Transformer layer 1 | 0.1669 | 85.8% |
| Transformer layer 2 | 0.1311 | 88.8% |
| Post-encoder normalization | 0.1229 | 89.5% |
| Embedding projection | 0.1224 | 89.6% |

The most meaningful internal result is that patching Transformer layer 2 substantially restores the target user's early embedding behavior. The post-encoder and embedding-projection interventions are upper-level interventions and should not be interpreted as evidence that those layers alone store the user's identity. The wrong-user control did not produce the same consistent recovery, supporting user-specificity of the intervention.

The causal-tracing result is an intervention-based localization result, not proof that a single layer contains the complete user profile. The next experiment should patch individual attention heads or feed-forward subcomponents in Transformer layer 2, then test whether a small edit to those components improves later-stage verification without damaging first-30-day performance.

Causal-tracing outputs:

- `results/causal_trace_user_163748.csv`;
- `results/causal_trace_user_163748_summary.json`;
- `figures/causal_trace_user_163748.png`.

## Output files

- `results/user_drift_metrics.csv`: user-stage metrics;
- `results/user_drift_overall_ranking.csv`: ranked users;
- `results/user_drift_summary.json`: analysis definition and maximum-drift summary;
- `figures/top_user_drift.png`: highest-drift users;
- `figures/user_drift_heatmap.png`: drift by user and stage;
- `figures/user_drift_by_stage_boxplot.png`: drift distributions;
- `figures/training_vs_later_stage_error_rates.png`: first-30-day versus later-stage error rates;
- `results/training_vs_later_stage_comparison.csv`: numeric comparison table.
