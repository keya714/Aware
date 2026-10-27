"""Compute diagnostic metrics for one participant across Week 1 and Week 8."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASELINE_DIR = Path(__file__).resolve().parents[1]
ROOT = BASELINE_DIR.parents[1]
sys.path.insert(0, str(BASELINE_DIR))
from KBOC.Sparkins01.experiments.attention_baseline.poc_causaltracing.evaluate_week1_baseline import build_arrays, calculate_metrics, score  # noqa: E402
from KBOC.Sparkins01.experiments.attention_baseline.train import AttentionModel, make_sequences  # noqa: E402


def summarize(path: Path, user: int, model, scaler, checkpoint) -> dict:
    events = pd.read_csv(path, compression="gzip")
    samples, vocab, report = make_sequences(events, max_len=checkpoint["max_len"])
    frozen_ids = {key: i + 1 for i, key in enumerate(checkpoint["key_vocab"])}
    for index, row in samples.iterrows():
        samples.at[index, "key_ids"] = np.asarray(
            [frozen_ids.get(vocab[int(key_id) - 1], 0) for key_id in row["key_ids"]], dtype=np.int64
        )
    key_ids, timing, mask = build_arrays(samples, checkpoint["key_vocab"], scaler, checkpoint["max_len"])
    probabilities = score(model, key_ids, timing, mask, batch_size=1024)
    labels = (samples["participant_id"] == user).astype(np.int64).to_numpy()
    metrics = calculate_metrics(labels, probabilities)
    genuine = probabilities[labels == 1]
    metrics["genuine_probability_distribution"] = {
        "n": int(len(genuine)), "min": float(genuine.min()), "max": float(genuine.max()),
        "mean": float(genuine.mean()), "std": float(genuine.std()),
        "median": float(np.median(genuine)),
        "q05": float(np.quantile(genuine, 0.05)), "q25": float(np.quantile(genuine, 0.25)),
        "q75": float(np.quantile(genuine, 0.75)), "q95": float(np.quantile(genuine, 0.95)),
    }
    metrics["week"] = int(events["week"].iloc[0])
    metrics["sequence_report"] = report
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", type=int, default=13)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "drift_analysis")
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(BASELINE_DIR / "outputs" / "model.pt", map_location="cpu", weights_only=False)
    with (BASELINE_DIR / "outputs" / "scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    model = AttentionModel(checkpoint["timing_dim"], len(checkpoint["key_vocab"]), checkpoint["max_len"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    result = {"participant_id": args.participant, "interpretation": "diagnostic cohort metrics from the frozen user-1-vs-all checkpoint"}
    result["week1"] = summarize(ROOT / "data" / "week_1.csv.gz", args.participant, model, scaler, checkpoint)
    cached = Path(__file__).parent / "outputs" / "week8_all_users" / "all_week8_sample_scores.csv"
    if cached.exists():
        scores = pd.read_csv(cached)
        labels = (scores["participant_id"] == args.participant).astype(int).to_numpy()
        probabilities = scores["genuine_probability"].to_numpy()
        result["week8"] = calculate_metrics(labels, probabilities)
        genuine = probabilities[labels == 1]
        result["week8"]["genuine_probability_distribution"] = {"n": int(len(genuine)), "min": float(genuine.min()), "max": float(genuine.max()), "mean": float(genuine.mean()), "std": float(genuine.std()), "median": float(np.median(genuine)), "q05": float(np.quantile(genuine, .05)), "q25": float(np.quantile(genuine, .25)), "q75": float(np.quantile(genuine, .75)), "q95": float(np.quantile(genuine, .95))}
    result["metric_change_week8_minus_week1"] = {m: result["week8"][m] - result["week1"][m] for m in ["roc_auc", "eer", "far", "frr", "recall", "f1"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"participant_{args.participant}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
