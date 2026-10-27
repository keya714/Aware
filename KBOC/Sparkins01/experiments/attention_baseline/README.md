# Attention Baseline Experiment

This folder contains the Week 1 one-vs-all attention baseline. Each complete
typing attempt becomes a padded event sequence. Padding is masked in the
Transformer and in sequence pooling.

The experiment is intentionally separate from \`experiments/fnn_baseline/\` so
the two models can be compared using the same target user and split protocol.
