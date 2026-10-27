"""Train a Week 1 one-vs-all feed-forward keystroke baseline.

This baseline represents each complete typing attempt by aggregate timing
features. It is intentionally separate from the later sequence/attention
experiment.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_KEYS = ["participant_id", "session_date", "session_num", "phrase_index", "repetition"]
TIMING_COLUMNS = [
    "hold_time",
    "down_down_latency",
    "up_down_latency",
    "elapsed_seconds",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_features(events: pd.DataFrame, max_events: int = 64) -> tuple[pd.DataFrame, list[str], dict]:
    """Aggregate each attempt into one fixed-length timing feature vector."""
    frame = events.copy()
    frame["press_time"] = pd.to_datetime(frame["press_time"], errors="coerce")
    frame["release_time"] = pd.to_datetime(frame["release_time"], errors="coerce")
    frame = frame.dropna(subset=["press_time", "release_time"])
    frame = frame.sort_values(SAMPLE_KEYS + ["key_seq"])

    frame["hold_time"] = (frame["release_time"] - frame["press_time"]).dt.total_seconds()
    previous_press = frame.groupby(SAMPLE_KEYS, sort=False)["press_time"].shift(1)
    previous_release = frame.groupby(SAMPLE_KEYS, sort=False)["release_time"].shift(1)
    frame["down_down_latency"] = (frame["press_time"] - previous_press).dt.total_seconds()
    frame["up_down_latency"] = (frame["press_time"] - previous_release).dt.total_seconds()
    frame["hold_time"] = frame["hold_time"].clip(lower=0)
    frame[TIMING_COLUMNS] = frame[TIMING_COLUMNS].replace([np.inf, -np.inf], np.nan)

    records = []
    dropped = 0
    for sample_id, group in frame.groupby(SAMPLE_KEYS, sort=False):
        group = group.sort_values("key_seq")
        n_events = len(group)
        # A few released rows contain extremely long malformed attempts. Do not
        # let truncation turn them into plausible samples.
        if n_events < 2 or n_events > max_events:
            dropped += 1
            continue

        record = dict(zip(SAMPLE_KEYS, sample_id))
        record["n_events"] = float(n_events)
        for column in TIMING_COLUMNS:
            values = group[column].dropna().to_numpy(dtype=np.float64)
            if len(values) == 0:
                values = np.array([0.0])
            record[f"{column}_mean"] = float(np.mean(values))
            record[f"{column}_std"] = float(np.std(values))
            record[f"{column}_median"] = float(np.median(values))
            record[f"{column}_min"] = float(np.min(values))
            record[f"{column}_max"] = float(np.max(values))
        records.append(record)

    features = pd.DataFrame(records)
    feature_names = [c for c in features.columns if c not in SAMPLE_KEYS]
    report = {
        "event_rows": int(len(events)),
        "attempts_kept": int(len(features)),
        "attempts_dropped_for_length_or_missing_timestamps": int(dropped),
        "max_events": max_events,
    }
    return features, feature_names, report


class FNN(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    scores, labels, total_loss, count = [], [], 0.0, 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            total_loss += float(criterion(logits, y)) * len(y)
            count += len(y)
            scores.extend(torch.sigmoid(logits).numpy())
            labels.extend(y.numpy())
    return np.asarray(labels), np.asarray(scores), total_loss / max(count, 1)


def calculate_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    predictions = (scores >= 0.5).astype(int)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fpr - fnr)))
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "far": float(fp / max(fp + tn, 1)),
        "frr": float(fn / max(fn + tp, 1)),
        "eer": float((fpr[eer_index] + fnr[eer_index]) / 2.0),
        "eer_threshold": float(thresholds[eer_index]),
        "n": int(len(labels)),
        "positive": int(labels.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/week_1.csv.gz")
    parser.add_argument("--target-user", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.data_path, compression="gzip")
    if sorted(events["week"].dropna().unique().tolist()) != [1]:
        raise ValueError("This experiment accepts Week 1 data only.")
    attempt_features, feature_names, feature_report = build_features(events)
    if args.target_user not in set(attempt_features["participant_id"]):
        raise ValueError(f"Target user {args.target_user} has no usable Week 1 attempts.")
    attempt_features["label"] = (attempt_features["participant_id"] == args.target_user).astype(np.float32)

    indices = np.arange(len(attempt_features))
    train_idx, other_idx = train_test_split(indices, test_size=0.30, random_state=args.seed, stratify=attempt_features["label"])
    val_idx, test_idx = train_test_split(other_idx, test_size=0.50, random_state=args.seed, stratify=attempt_features.iloc[other_idx]["label"])
    x = attempt_features[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    y = attempt_features["label"].to_numpy(dtype=np.float32)
    scaler = StandardScaler().fit(x[train_idx])
    x = scaler.transform(x).astype(np.float32)

    def loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        return DataLoader(TensorDataset(torch.from_numpy(x[idx]), torch.from_numpy(y[idx])), batch_size=args.batch_size, shuffle=shuffle)

    train_loader, val_loader, test_loader = loader(train_idx, True), loader(val_idx, False), loader(test_idx, False)
    positives = max(float(y[train_idx].sum()), 1.0)
    negatives = max(float(len(train_idx) - y[train_idx].sum()), 1.0)
    model = FNN(len(feature_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives]))
    history = {"train_loss": [], "val_loss": []}
    best_state, best_val = None, float("inf")
    for _ in range(args.epochs):
        model.train()
        total, count = 0.0, 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(batch_y)
            count += len(batch_y)
        _, _, val_loss = predict(model, val_loader)
        history["train_loss"].append(total / count)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    train_labels, train_scores, _ = predict(model, loader(train_idx, False))
    val_labels, val_scores, _ = predict(model, val_loader)
    test_labels, test_scores, _ = predict(model, test_loader)
    metrics = {"train": calculate_metrics(train_labels, train_scores), "validation": calculate_metrics(val_labels, val_scores), "test": calculate_metrics(test_labels, test_scores)}

    torch.save({"model_state_dict": model.state_dict(), "input_dim": len(feature_names)}, args.output_dir / "model.pt")
    with open(args.output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    config = {"target_user": args.target_user, "feature_names": feature_names, "sample_keys": SAMPLE_KEYS, "architecture": "64-32 FNN", "seed": args.seed, "feature_report": feature_report, "split_sizes": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)}}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="validation")
    plt.xlabel("Epoch"); plt.ylabel("Weighted BCE loss"); plt.legend(); plt.tight_layout()
    plt.savefig(args.output_dir / "loss_curve.png", dpi=150); plt.close()
    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={metrics['test']['roc_auc']:.3f}"); plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(); plt.tight_layout()
    plt.savefig(args.output_dir / "roc_curve.png", dpi=150); plt.close()
    print(json.dumps({"feature_report": feature_report, "metrics": metrics, "outputs": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
