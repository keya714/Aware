"""Step 2: evaluate the frozen Week 1 attention baseline.

The original training script did not persist split indices, so this evaluator
reconstructs the held-out test split from the same seed and stratification
protocol used by ``attention_baseline/train.py``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

BASELINE_DIR = Path(__file__).resolve().parents[1]
ROOT = BASELINE_DIR.parents[1]
sys.path.insert(0, str(BASELINE_DIR))

from KBOC.Sparkins01.experiments.attention_baseline.train import AttentionModel, EVENT_FEATURES, make_sequences  # noqa: E402


def build_arrays(samples: pd.DataFrame, key_vocab: list[str], scaler, max_len: int):
    timing = np.zeros((len(samples), max_len, len(EVENT_FEATURES)), dtype=np.float32)
    key_ids = np.zeros((len(samples), max_len), dtype=np.int64)
    mask = np.zeros((len(samples), max_len), dtype=bool)
    for i, row in samples.iterrows():
        n = int(row["length"])
        timing[i, :n] = row["timing"]
        # make_sequences assigns IDs from the sorted vocabulary, matching the
        # checkpoint's embedding rows.
        key_ids[i, :n] = row["key_ids"]
        mask[i, :n] = True
    timing = scaler.transform(timing.reshape(-1, len(EVENT_FEATURES))).reshape(timing.shape).astype(np.float32)
    timing[~mask] = 0.0
    return key_ids, timing, mask


def score(model, key_ids, timing, mask, batch_size: int = 256) -> np.ndarray:
    scores: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(key_ids), batch_size):
            stop = start + batch_size
            logits = model(
                torch.from_numpy(key_ids[start:stop]),
                torch.from_numpy(timing[start:stop]),
                torch.from_numpy(mask[start:stop]),
            )
            scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float64)


def calculate_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    predictions = (scores >= 0.5).astype(int)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "eer": float((fpr[eer_index] + fnr[eer_index]) / 2.0),
        "eer_threshold": float(thresholds[eer_index]),
        "far": float(fp / max(fp + tn, 1)),
        "frr": float(fn / max(fn + tp, 1)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen Week 1 attention baseline")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "week_1.csv.gz")
    parser.add_argument("--model-path", type=Path, default=BASELINE_DIR / "outputs" / "model.pt")
    parser.add_argument("--scaler-path", type=Path, default=BASELINE_DIR / "outputs" / "scaler.pkl")
    parser.add_argument("--target-user", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    # A single CPU thread is substantially faster for this small checkpoint
    # in constrained environments and keeps the evaluator reproducible.
    torch.set_num_threads(1)

    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    with args.scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)
    model = AttentionModel(
        checkpoint["timing_dim"], len(checkpoint["key_vocab"]), checkpoint["max_len"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    events = pd.read_csv(args.data_path, compression="gzip")
    if sorted(events["week"].dropna().unique().tolist()) != [1]:
        raise ValueError("Step 2 requires a Week 1-only data file")
    samples, discovered_vocab, report = make_sequences(events, max_len=checkpoint["max_len"])
    if discovered_vocab != checkpoint["key_vocab"]:
        raise ValueError("Week 1 key vocabulary differs from the trained checkpoint")
    samples["label"] = (samples["participant_id"] == args.target_user).astype(np.int64)

    indices = np.arange(len(samples))
    train_idx, other_idx = train_test_split(
        indices, test_size=0.30, random_state=args.seed, stratify=samples["label"]
    )
    _, test_idx = train_test_split(
        other_idx,
        test_size=0.50,
        random_state=args.seed,
        stratify=samples.iloc[other_idx]["label"],
    )
    del train_idx

    key_ids, timing, mask = build_arrays(samples, checkpoint["key_vocab"], scaler, checkpoint["max_len"])
    labels = samples["label"].to_numpy(np.int64)[test_idx]
    scores = score(model, key_ids[test_idx], timing[test_idx], mask[test_idx], batch_size=1024)
    result = calculate_metrics(labels, scores)
    genuine_scores = scores[labels == 1]
    result["target_user"] = args.target_user
    result["split"] = "reconstructed held-out Week 1 test split"
    result["split_seed"] = args.seed
    result["sequence_report"] = report
    result["genuine_probability_distribution"] = {
        "n": int(len(genuine_scores)),
        "min": float(np.min(genuine_scores)),
        "max": float(np.max(genuine_scores)),
        "mean": float(np.mean(genuine_scores)),
        "std": float(np.std(genuine_scores)),
        "quantiles": {
            str(q): float(np.quantile(genuine_scores, q)) for q in [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
        },
        "scores": [float(value) for value in genuine_scores],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "week1_baseline.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
