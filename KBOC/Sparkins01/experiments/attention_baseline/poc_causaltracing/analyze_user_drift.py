"""Find participant-level score-distribution drift from Week 1 to Week 8."""

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
from KBOC.Sparkins01.experiments.attention_baseline.poc_causaltracing.evaluate_week1_baseline import build_arrays, score  # noqa: E402
from KBOC.Sparkins01.experiments.attention_baseline.train import AttentionModel, make_sequences  # noqa: E402


def ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    values = np.sort(np.unique(np.concatenate([left, right])))
    left_sorted, right_sorted = np.sort(left), np.sort(right)
    left_cdf = np.searchsorted(left_sorted, values, side="right") / len(left)
    right_cdf = np.searchsorted(right_sorted, values, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def frozen_scores(path: Path, model, scaler, checkpoint):
    events = pd.read_csv(path, compression="gzip")
    samples, vocab, report = make_sequences(events, max_len=checkpoint["max_len"])
    frozen_ids = {key: i + 1 for i, key in enumerate(checkpoint["key_vocab"])}
    for index, row in samples.iterrows():
        samples.at[index, "key_ids"] = np.asarray(
            [frozen_ids.get(vocab[int(key_id) - 1], 0) for key_id in row["key_ids"]], dtype=np.int64
        )
    key_ids, timing, mask = build_arrays(samples, checkpoint["key_vocab"], scaler, checkpoint["max_len"])
    samples["probability"] = score(model, key_ids, timing, mask, batch_size=1024)
    return samples[["participant_id", "probability"]], report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week1-data", type=Path, default=ROOT / "data" / "week_1.csv.gz")
    parser.add_argument("--week8-data", type=Path, default=ROOT / "data" / "week_8.csv.gz")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "drift_analysis")
    args = parser.parse_args()
    torch.set_num_threads(1)

    checkpoint = torch.load(BASELINE_DIR / "outputs" / "model.pt", map_location="cpu", weights_only=False)
    with (BASELINE_DIR / "outputs" / "scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    model = AttentionModel(checkpoint["timing_dim"], len(checkpoint["key_vocab"]), checkpoint["max_len"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    week1, report1 = frozen_scores(args.week1_data, model, scaler, checkpoint)
    cached_week8 = Path(__file__).parent / "outputs" / "week8_all_users" / "all_week8_sample_scores.csv"
    if cached_week8.exists():
        cached = pd.read_csv(cached_week8, usecols=["participant_id", "genuine_probability"])
        week8 = cached.rename(columns={"genuine_probability": "probability"})
        report8 = {"source": str(cached_week8), "cached": True}
    else:
        week8, report8 = frozen_scores(args.week8_data, model, scaler, checkpoint)
    rows = []
    for user in sorted(set(week1.participant_id) & set(week8.participant_id)):
        s1 = week1.loc[week1.participant_id == user, "probability"].to_numpy()
        s8 = week8.loc[week8.participant_id == user, "probability"].to_numpy()
        rows.append({
            "participant_id": int(user),
            "n_week1": len(s1), "n_week8": len(s8),
            "mean_week1": float(s1.mean()), "mean_week8": float(s8.mean()),
            "mean_delta_week8_minus_week1": float(s8.mean() - s1.mean()),
            "median_delta_week8_minus_week1": float(np.median(s8) - np.median(s1)),
            "ks_statistic": ks_statistic(s1, s8),
            "week1_rejection_rate_at_0.5": float((s1 < 0.5).mean()),
            "week8_rejection_rate_at_0.5": float((s8 < 0.5).mean()),
            "rejection_rate_delta": float((s8 < 0.5).mean() - (s1 < 0.5).mean()),
        })
    result = pd.DataFrame(rows).sort_values(["ks_statistic", "mean_delta_week8_minus_week1"], ascending=[False, True])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "user_drift_ranked.csv", index=False)
    summary = {
        "ranking_definition": "KS distribution shift first; mean score decrease breaks ties",
        "week1_sequence_report": report1,
        "week8_sequence_report": report8,
        "most_drift_ks": result.iloc[0].to_dict(),
        "largest_mean_decrease": result.sort_values("mean_delta_week8_minus_week1").iloc[0].to_dict(),
        "largest_rejection_rate_increase": result.sort_values("rejection_rate_delta", ascending=False).iloc[0].to_dict(),
    }
    (args.output_dir / "drift_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
