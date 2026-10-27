"""Step 1: load the released Week 1 keystroke data only.

This module deliberately accepts one explicit file path instead of scanning the
data directory. That keeps later longitudinal experiments from accidentally
mixing Week 2--8 data into the Week 1 baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_WEEK = 1


def load_week1(path: str | Path) -> pd.DataFrame:
    """Load and validate one compressed Week 1 CSV file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Week 1 data was not found at {path}. "
            "Download data/week_1.csv.gz from the project repository."
        )

    frame = pd.read_csv(path, compression="gzip")
    if "week" not in frame.columns:
        raise ValueError("The input does not contain the required 'week' column.")

    weeks = sorted(frame["week"].dropna().unique().tolist())
    if weeks != [EXPECTED_WEEK]:
        raise ValueError(f"Expected only week 1, but found week values: {weeks}")

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and inspect Week 1 keystrokes")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/week_1.csv.gz"),
        help="Path to the Week 1 gzip-compressed CSV",
    )
    args = parser.parse_args()

    frame = load_week1(args.data_path)
    print(f"Loaded Week 1: {len(frame):,} event rows")
    print(f"Shape: {frame.shape}")
    print("Columns:")
    for column in frame.columns:
        print(f"  - {column}: {frame[column].dtype}")
    print("\nFirst five rows:")
    print(frame.head().to_string(index=False))


if __name__ == "__main__":
    main()
