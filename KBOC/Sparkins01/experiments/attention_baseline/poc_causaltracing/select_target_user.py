"""Step 1: record and validate the causal-tracing target user.

The current proof of concept intentionally uses participant 1. This script
keeps the choice reproducible and checks the available data before later
causal-tracing steps are run. Week 8 is optional while the repository only
contains the released Week 1 file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEEK1 = ROOT / "data" / "week_1.csv.gz"
DEFAULT_WEEK8 = ROOT / "data" / "week_8.csv.gz"
DEFAULT_OUTPUT = Path(__file__).parent / "outputs" / "target_user.json"
TARGET_USER = 1


def load_attempt_counts(path: Path, expected_week: int) -> dict[int, int]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, compression="gzip")
    required = {"week", "participant_id", "session_date", "session_num", "phrase_index", "repetition"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    weeks = sorted(frame["week"].dropna().unique().tolist())
    if weeks != [expected_week]:
        raise ValueError(f"Expected only week {expected_week} in {path}, found {weeks}")
    keys = ["participant_id", "session_date", "session_num", "phrase_index", "repetition"]
    grouped = frame.groupby(keys, dropna=False)
    # Match the baseline's usable-attempt bounds (at least two events and no
    # more than the model's 32-token maximum).
    usable = grouped.size().loc[lambda sizes: sizes.between(2, 32)]
    return usable.reset_index().groupby("participant_id").size().astype(int).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the causal-tracing target user")
    parser.add_argument("--week1-data", type=Path, default=DEFAULT_WEEK1)
    parser.add_argument("--week8-data", type=Path, default=DEFAULT_WEEK8)
    parser.add_argument("--target-user", type=int, default=TARGET_USER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    week1 = load_attempt_counts(args.week1_data, 1)
    if args.target_user not in week1:
        raise ValueError(f"Target user {args.target_user} has no Week 1 samples")

    result = {
        "target_user": args.target_user,
        "dataset_source": "https://github.com/sparkins01/keystroke",
        "negative_label": 0,
        "positive_label": 1,
        "week1": {
            "data_path": str(args.week1_data),
            "usable_attempts_approx": week1[args.target_user],
            "baseline_metrics_path": str(ROOT / "experiments/attention_baseline/outputs/metrics.json"),
        },
        "week8": {"status": "pending", "data_path": str(args.week8_data)},
        "selection_criteria": {
            "good_week1_performance": "confirmed by existing baseline metrics",
            "enough_week8_samples": "pending Week 8 data",
            "week8_degradation": "pending Week 8 data",
            "false_negatives": "pending Week 8 evaluation",
        },
    }
    if args.week8_data.exists():
        week8 = load_attempt_counts(args.week8_data, 8)
        result["week8"] = {
            "status": "available",
            "data_path": str(args.week8_data),
            "usable_attempts_approx": week8.get(args.target_user, 0),
        }
        result["selection_criteria"]["enough_week8_samples"] = (
            f"confirmed: {week8.get(args.target_user, 0)} usable attempts"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
