# FNN Baseline Experiment

This folder contains the first one-vs-all Week 1 baseline using a feed-forward
neural network. It is intentionally separate from the future attention-model
experiment.

The experiment will use the shared dataset at:

```text
data/week_1.csv.gz
```

Shared data loading utilities remain in `src/`. Experiment-specific feature
construction, training, evaluation, plots, and saved artifacts belong here.

Planned layout:

```text
experiments/fnn_baseline/
├── README.md
├── train.py
├── features.py
├── model.py
└── outputs/
```

The FNN will operate on one fixed-length feature vector per typing sample. The
sample boundary will remain the complete typing attempt identified by
`participant_id`, `session_date`, `session_num`, `phrase_index`, and
`repetition`.
