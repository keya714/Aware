"""Train a Week 1 one-vs-all Transformer keystroke baseline."""

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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_KEYS = ["participant_id", "session_date", "session_num", "phrase_index", "repetition"]
EVENT_FEATURES = ["hold_time", "down_down_latency", "up_down_latency", "elapsed_seconds"]


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def make_sequences(events: pd.DataFrame, max_len: int = 32) -> tuple[pd.DataFrame, list[str], dict]:
    frame = events.copy()
    frame["press_time"] = pd.to_datetime(frame["press_time"], errors="coerce")
    frame["release_time"] = pd.to_datetime(frame["release_time"], errors="coerce")
    frame = frame.dropna(subset=["press_time", "release_time"]).sort_values(SAMPLE_KEYS + ["key_seq"])
    frame["hold_time"] = (frame["release_time"] - frame["press_time"]).dt.total_seconds().clip(lower=0)
    frame["previous_press"] = frame.groupby(SAMPLE_KEYS, sort=False)["press_time"].shift(1)
    frame["previous_release"] = frame.groupby(SAMPLE_KEYS, sort=False)["release_time"].shift(1)
    frame["down_down_latency"] = (frame["press_time"] - frame["previous_press"]).dt.total_seconds()
    frame["up_down_latency"] = (frame["press_time"] - frame["previous_release"]).dt.total_seconds()
    frame[EVENT_FEATURES] = frame[EVENT_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    keys = sorted(frame["key"].astype(str).unique())
    key_to_id = {key: i + 1 for i, key in enumerate(keys)}
    frame["key_id"] = frame["key"].astype(str).map(key_to_id).astype(np.int64)

    records, dropped = [], 0
    for sample_id, group in frame.groupby(SAMPLE_KEYS, sort=False):
        group = group.sort_values("key_seq")
        if len(group) < 2 or len(group) > max_len:
            dropped += 1
            continue
        record = dict(zip(SAMPLE_KEYS, sample_id))
        record["key_ids"] = group["key_id"].to_numpy(np.int64)
        record["timing"] = group[EVENT_FEATURES].to_numpy(np.float32)
        record["length"] = len(group)
        records.append(record)
    result = pd.DataFrame(records)
    report = {"event_rows": int(len(events)), "attempts_kept": int(len(result)), "attempts_dropped": int(dropped), "max_sequence_length": max_len, "key_vocabulary_size": len(keys)}
    return result, keys, report


class AttentionModel(nn.Module):
    def __init__(self, timing_dim: int, key_vocab_size: int, max_len: int, d_model: int = 64) -> None:
        super().__init__()
        self.key_embedding = nn.Embedding(key_vocab_size + 1, 16, padding_idx=0)
        self.input_projection = nn.Linear(timing_dim + 16, d_model)
        self.position = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, key_ids: torch.Tensor, timing: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.key_embedding(key_ids), timing], dim=-1)
        x = self.input_projection(x) + self.position[:, :x.shape[1]]
        x = self.encoder(x, src_key_padding_mask=~mask)
        weights = mask.unsqueeze(-1).float()
        pooled = (x * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        return self.head(pooled).squeeze(1)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval(); labels, scores, total, count = [], [], 0.0, 0
    with torch.no_grad():
        for key_ids, timing, mask, y in loader:
            logits = model(key_ids, timing, mask)
            total += criterion(logits, y).item() * len(y); count += len(y)
            labels.extend(y.numpy()); scores.extend(torch.sigmoid(logits).numpy())
    return np.asarray(labels), np.asarray(scores), total / max(count, 1)


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    pred = (scores >= 0.5).astype(int)
    fpr, tpr, thresholds = roc_curve(labels, scores); fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(labels, scores)), "accuracy": float(accuracy_score(labels, pred)), "precision": float(precision_score(labels, pred, zero_division=0)), "recall": float(recall_score(labels, pred, zero_division=0)), "f1": float(f1_score(labels, pred, zero_division=0)), "far": float(fp / max(fp + tn, 1)), "frr": float(fn / max(fn + tp, 1)), "eer": float((fpr[i] + fnr[i]) / 2), "eer_threshold": float(thresholds[i]), "n": int(len(labels)), "positive": int(labels.sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/week_1.csv.gz")
    parser.add_argument("--target-user", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); seed_everything(args.seed); args.output_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.data_path, compression="gzip")
    if sorted(events["week"].dropna().unique().tolist()) != [1]: raise ValueError("This experiment accepts Week 1 only.")
    samples, key_vocab, report = make_sequences(events)
    samples["label"] = (samples["participant_id"] == args.target_user).astype(np.float32)
    if samples["label"].sum() == 0: raise ValueError(f"Target user {args.target_user} has no usable samples.")
    indices = np.arange(len(samples)); train_idx, other_idx = train_test_split(indices, test_size=0.30, random_state=args.seed, stratify=samples["label"])
    val_idx, test_idx = train_test_split(other_idx, test_size=0.50, random_state=args.seed, stratify=samples.iloc[other_idx]["label"])

    max_len = report["max_sequence_length"]
    timing = np.zeros((len(samples), max_len, len(EVENT_FEATURES)), dtype=np.float32)
    key_ids = np.zeros((len(samples), max_len), dtype=np.int64)
    mask = np.zeros((len(samples), max_len), dtype=bool)
    for i, row in samples.iterrows():
        n = int(row["length"]); timing[i, :n] = row["timing"]; key_ids[i, :n] = row["key_ids"]; mask[i, :n] = True
    scaler = StandardScaler().fit(timing[train_idx][mask[train_idx]])
    timing = scaler.transform(timing.reshape(-1, len(EVENT_FEATURES))).reshape(timing.shape).astype(np.float32)
    timing[~mask] = 0.0
    labels = samples["label"].to_numpy(np.float32)

    def loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(key_ids[idx]), torch.from_numpy(timing[idx]), torch.from_numpy(mask[idx]), torch.from_numpy(labels[idx]))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)
    train_loader, val_loader, test_loader = loader(train_idx, True), loader(val_idx, False), loader(test_idx, False)
    positives = max(float(labels[train_idx].sum()), 1.0); negatives = max(float(len(train_idx) - labels[train_idx].sum()), 1.0)
    model = AttentionModel(len(EVENT_FEATURES), len(key_vocab), max_len); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives])); history = {"train_loss": [], "val_loss": []}; best_state, best_val = None, float("inf")
    for _ in range(args.epochs):
        model.train(); total, count = 0.0, 0
        for batch_keys, batch_timing, batch_mask, batch_y in train_loader:
            optimizer.zero_grad(); loss = criterion(model(batch_keys, batch_timing, batch_mask), batch_y); loss.backward(); optimizer.step()
            total += loss.detach().item() * len(batch_y); count += len(batch_y)
        _, _, val_loss = evaluate(model, val_loader, nn.BCEWithLogitsLoss())
        history["train_loss"].append(total / count); history["val_loss"].append(val_loss)
        if val_loss < best_val: best_val = val_loss; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    train_y, train_s, _ = evaluate(model, loader(train_idx, False), nn.BCEWithLogitsLoss()); val_y, val_s, _ = evaluate(model, val_loader, nn.BCEWithLogitsLoss()); test_y, test_s, _ = evaluate(model, test_loader, nn.BCEWithLogitsLoss())
    all_metrics = {"train": metrics(train_y, train_s), "validation": metrics(val_y, val_s), "test": metrics(test_y, test_s)}

    torch.save({"model_state_dict": model.state_dict(), "timing_dim": len(EVENT_FEATURES), "key_vocab": key_vocab, "max_len": max_len}, args.output_dir / "model.pt")
    with open(args.output_dir / "scaler.pkl", "wb") as f: pickle.dump(scaler, f)
    config = {"target_user": args.target_user, "sample_keys": SAMPLE_KEYS, "event_features": EVENT_FEATURES, "max_sequence_length": max_len, "d_model": 64, "encoder_layers": 2, "attention_heads": 4, "seed": args.seed, "feature_report": report, "split_sizes": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)}}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plt.figure(); plt.plot(history["train_loss"], label="train"); plt.plot(history["val_loss"], label="validation"); plt.xlabel("Epoch"); plt.ylabel("Weighted BCE loss"); plt.legend(); plt.tight_layout(); plt.savefig(args.output_dir / "loss_curve.png", dpi=150); plt.close()
    fpr, tpr, _ = roc_curve(test_y, test_s); plt.figure(); plt.plot(fpr, tpr, label=f"AUC={all_metrics['test']['roc_auc']:.3f}"); plt.plot([0, 1], [0, 1], "--", color="gray"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(); plt.tight_layout(); plt.savefig(args.output_dir / "roc_curve.png", dpi=150); plt.close()
    print(json.dumps({"sequence_report": report, "metrics": all_metrics, "outputs": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
