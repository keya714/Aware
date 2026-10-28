# 30-day longitudinal drift experiments

The frozen baseline is `results/attention_model_best.pt`. The follow-up
experiments use Stage 1 as the adaptation/tracing window and keep Stage 2 and
Stage 3 held out:

```powershell
python causal_trace_stage1.py
python model_editing.py
```

Outputs are written under `results/causal_trace_stage1/` and
`results/model_editing/`. The editing script applies ROME-style rank-one edits
to the second Transformer layer's feed-forward projection, saves edited
checkpoints, and reports fixed-threshold FRR/FAR, early retention, and
parameter displacement. It also saves PNG visualizations:

- `causal_trace_stage1/figures/causal_trace_recovery_heatmap.png`
- `causal_trace_stage1/figures/causal_trace_distance_heatmap.png`
- `causal_trace_stage1/figures/causal_trace_control_heatmap.png`
- `model_editing/figures/rome_frr_before_after.png`
- `model_editing/figures/rome_far_before_after.png`
- `model_editing/figures/rome_distance_recovery.png`
- `model_editing/figures/rome_recovery_security_tradeoff.png`
- `model_editing/figures/rome_edit_cost.png`

The causal-tracing heatmaps use rows for model interventions and columns for
users. The ROME plots separate embedding recovery from FRR recovery and FAR
security degradation.

For a quick deterministic pilot:

```powershell
python model_editing.py --users 1 --v-steps 5 --replay-size 64
```

By default, `model_editing.py` evaluates every eligible user. Use `--users N`
only for a pilot. Eligibility requires at least five windows in each later
stage. Stage 1 alone is used for drift ranking and edit construction; Stage 2
and Stage 3 remain held out.

Population outputs include:

- `results/model_editing/eligible_users.csv`
- `results/model_editing/stage1_drift_ranking.csv`
- `results/model_editing/rome_per_user_metrics.csv`
- `results/model_editing/rome_population_summary.csv`
- `results/model_editing/rome_bootstrap_confidence_intervals.csv`
- `results/model_editing/rome_statistical_tests.json`

Additional population figures include per-user FRR/FAR distributions, success
by user, macro-average FRR, and Stage 1 drift versus Stage 2 recovery.

Phase 1 value-vector comparisons are isolated by output directory:

```powershell
python model_editing.py --v-steps 1 --adaptation-size 64 --replay-size 64 --output-dir results/phase1/v_steps_1
python model_editing.py --v-steps 5 --adaptation-size 64 --replay-size 64 --output-dir results/phase1/v_steps_5
python model_editing.py --v-steps 20 --adaptation-size 64 --replay-size 64 --output-dir results/phase1/v_steps_20
```

Keeping adaptation and replay sizes fixed isolates the effect of
`v_optimization_steps`.
