"""Evaluate the frozen user-1 model and archive Week 8 diagnostics for all users."""

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

SAMPLE_KEYS = ["participant_id", "session_date", "session_num", "phrase_index", "repetition"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive frozen Week 8 scores for every participant")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "week_8.csv.gz")
    parser.add_argument("--model-path", type=Path, default=BASELINE_DIR / "outputs" / "model.pt")
    parser.add_argument("--scaler-path", type=Path, default=BASELINE_DIR / "outputs" / "scaler.pkl")
    parser.add_argument("--target-user", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "week8_all_users")
    args = parser.parse_args()
    torch.set_num_threads(1)

    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    with args.scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)
    model = AttentionModel(checkpoint["timing_dim"], len(checkpoint["key_vocab"]), checkpoint["max_len"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    events = pd.read_csv(args.data_path, compression="gzip")
    if sorted(events["week"].dropna().unique().tolist()) != [8]:
        raise ValueError("This evaluator requires a Week 8-only data file")
    samples, discovered_vocab, sequence_report = make_sequences(events, max_len=checkpoint["max_len"])
    frozen_ids = {key: i + 1 for i, key in enumerate(checkpoint["key_vocab"])}
    for index, row in samples.iterrows():
        samples.at[index, "key_ids"] = np.asarray(
            [frozen_ids.get(discovered_vocab[int(key_id) - 1], 0) for key_id in row["key_ids"]],
            dtype=np.int64,
        )
    key_ids, timing, mask = build_arrays(samples, checkpoint["key_vocab"], scaler, checkpoint["max_len"])
    probabilities = score(model, key_ids, timing, mask, batch_size=1024)

    labels = (samples["participant_id"] == args.target_user).astype(np.int64).to_numpy()
    predictions = (probabilities >= 0.5).astype(np.int64)
    samples["genuine_probability"] = probabilities
    samples["prediction_at_0.5"] = predictions
    samples["is_target_user"] = labels

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    score_columns = SAMPLE_KEYS + ["genuine_probability", "prediction_at_0.5", "is_target_user"]
    samples[score_columns].to_csv(output / "all_week8_sample_scores.csv", index=False)

    summaries = []
    for participant_id, group in samples.groupby("participant_id", sort=True):
        participant_scores = group["genuine_probability"].to_numpy()
        participant_predictions = group["prediction_at_0.5"].to_numpy()
        summaries.append({
            "participant_id": int(participant_id),
            "is_model_target": bool(int(participant_id) == args.target_user),
            "n_samples": int(len(group)),
            "accepted_at_0.5": int(participant_predictions.sum()),
            "rejected_at_0.5": int((participant_predictions == 0).sum()),
            "rejection_rate_at_0.5": float((participant_predictions == 0).mean()),
            "probability_min": float(participant_scores.min()),
            "probability_max": float(participant_scores.max()),
            "probability_mean": float(participant_scores.mean()),
            "probability_std": float(participant_scores.std()),
            "probability_median": float(np.median(participant_scores)),
        })
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output / "per_user_week8_summary.csv", index=False)

    target_samples = samples[samples["participant_id"] == args.target_user]
    target_false_rejections = target_samples[target_samples["prediction_at_0.5"] == 0]
    target_false_rejections[score_columns].to_csv(output / "target_user_false_rejections.csv", index=False)

    result = {
        "target_user": args.target_user,
        "week": 8,
        "model_update": "none",
        "interpretation": "user-1-vs-all checkpoint; per-user rows are cohort diagnostics, not separate user authentication models",
        "n_users": int(samples["participant_id"].nunique()),
        "n_samples": int(len(samples)),
        "sequence_report": sequence_report,
        "target_user_metrics": calculate_metrics(labels, probabilities),
        "target_user_false_rejections": int(len(target_false_rejections)),
        "output_files": {
            "all_sample_scores": "all_week8_sample_scores.csv",
            "per_user_summary": "per_user_week8_summary.csv",
            "target_false_rejections": "target_user_false_rejections.csv",
        },
    }
    (output / "all_users_week8.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
