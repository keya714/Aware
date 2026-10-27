# Clarkson II attention model

This folder contains the Checkpoint 1 attention-model experiment for the clean Clarkson II subset.

The experiment uses 94 of the 100 local user files (94% user coverage), 79,285 fixed 50-pair windows, and timing-only event sequences. The five-second preprocessing threshold is kept fixed. The model is a small Transformer encoder trained to classify the enrolled users; its 32-dimensional embeddings are evaluated as a verification system.

## Run

From the project root:

```powershell
python clarkson2_attention_model\prepare_sequences.py
python clarkson2_attention_model\train_attention.py --epochs 8
```

Outputs are written to `clarkson2_attention_model/results/`.

`prepare_sequences.py` reconstructs the same non-overlapping 50-pair windows from `clean_events` and joins them to `windows_normalized.csv`, so the preprocessing split remains the authoritative split. It saves timing sequences as `sequences.npz` and metadata as `sequence_metadata.csv`.
