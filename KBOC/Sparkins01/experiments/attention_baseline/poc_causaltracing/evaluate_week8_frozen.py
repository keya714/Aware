"""Step 3: evaluate the frozen Week 1 model on Week 8."""

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
    parser = argparse.ArgumentParser(description="Run the frozen Week 1 model on Week 8")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "week_8.csv.gz")
    parser.add_argument("--model-path", type=Path, default=BASELINE_DIR / "outputs" / "model.pt")
    parser.add_argument("--scaler-path", type=Path, default=BASELINE_DIR / "outputs" / "scaler.pkl")
    parser.add_argument("--week1-results", type=Path, default=Path(__file__).parent / "outputs" / "week1_baseline.json")
    parser.add_argument("--target-user", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
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
        raise ValueError("Step 3 requires a Week 8-only data file")
    samples, discovered_vocab, report = make_sequences(events, max_len=checkpoint["max_len"])
    # make_sequences numbers each file's observed vocabulary independently.
    # Remap those IDs to the frozen Week 1 embedding rows; an unseen Week 8
    # key is represented by the embedding's padding/unknown row 0.
    frozen_ids = {key: i + 1 for i, key in enumerate(checkpoint["key_vocab"])}
    for index, row in samples.iterrows():
        samples.at[index, "key_ids"] = np.asarray(
            [frozen_ids.get(discovered_vocab[int(key_id) - 1], 0) for key_id in row["key_ids"]],
            dtype=np.int64,
        )

    labels = (samples["participant_id"] == args.target_user).astype(np.int64).to_numpy()
    key_ids, timing, mask = build_arrays(samples, checkpoint["key_vocab"], scaler, checkpoint["max_len"])
    probabilities = score(model, key_ids, timing, mask, batch_size=1024)
    metrics = calculate_metrics(labels, probabilities)
    samples["genuine_probability"] = probabilities
    samples["label"] = labels
    samples["prediction"] = (probabilities >= 0.5).astype(np.int64)
    false_rejections = samples[(samples["label"] == 1) & (samples["prediction"] == 0)].copy()
    false_rejections["sample_index"] = false_rejections.index.astype(int)
    false_rejections = false_rejections[SAMPLE_KEYS + ["sample_index", "genuine_probability", "prediction"]]

    week1 = json.loads(args.week1_results.read_text(encoding="utf-8"))
    comparison = {}
    for metric in ["roc_auc", "eer", "far", "frr", "recall", "f1"]:
        comparison[metric] = {
            "week1": week1.get(metric),
            "week8": metrics.get(metric),
            "week8_minus_week1": metrics[metric] - week1[metric],
        }
    result = {
        "target_user": args.target_user,
        "week": 8,
        "model_update": "none",
        "data_path": str(args.data_path),
        "sequence_report": report,
        "metrics": metrics,
        "comparison": comparison,
        "false_rejections": int(len(false_rejections)),
        "genuine_user_probability_distribution": {
            "n": int(labels.sum()),
            "scores": [float(value) for value in probabilities[labels == 1]],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "week8_frozen.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    false_rejections.to_csv(args.output_dir / "week8_false_rejections.csv", index=False)
    print(json.dumps({"week8_metrics": metrics, "false_rejections": len(false_rejections)}, indent=2))


if __name__ == "__main__":
    main()
