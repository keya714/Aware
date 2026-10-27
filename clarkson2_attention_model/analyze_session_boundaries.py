from pathlib import Path
import gzip
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "clarkson2_preprocessing" / "outputs" / "final_clean_dataset" / "clean_events"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

def main():
    rows = []
    for path in sorted(DATA.glob("*.clean_events.tsv.gz")):
        user = path.name.split(".", 1)[0]
        with gzip.open(path, "rt") as fh:
            d = pd.read_csv(fh, sep="\t", usecols=["press_timestamp_100ns"])
        ts = np.sort(pd.to_numeric(d["press_timestamp_100ns"], errors="coerce").dropna().astype(np.int64).unique())
        if len(ts) < 2:
            continue
        diffs = np.diff(ts)
        positive = diffs > 0
        if not positive.any():
            continue
        candidates = np.where(positive)[0]
        order = candidates[np.argsort(diffs[candidates])[::-1]]
        top = []
        for idx in order[:5]:
            gap_seconds = float(diffs[idx]) / 1e7
            top.append({"rank": len(top) + 1, "gap_seconds": gap_seconds, "gap_days": gap_seconds / 86400.0, "before_timestamp_100ns": int(ts[idx]), "after_timestamp_100ns": int(ts[idx + 1])})
        best = top[0]
        boundary = (best["before_timestamp_100ns"] + best["after_timestamp_100ns"]) // 2
        before = int(np.searchsorted(ts, boundary, side="right"))
        rows.append({
            "user_id": user,
            "event_count": len(ts),
            "first_timestamp_100ns": int(ts[0]),
            "last_timestamp_100ns": int(ts[-1]),
            "span_days": (int(ts[-1]) - int(ts[0])) / 1e7 / 86400.0,
            "largest_gap_days": best["gap_days"],
            "second_gap_days": top[1]["gap_days"] if len(top) > 1 else np.nan,
            "third_gap_days": top[2]["gap_days"] if len(top) > 2 else np.nan,
            "boundary_timestamp_100ns": boundary,
            "events_before_boundary": before,
            "events_after_boundary": len(ts) - before,
            "before_fraction": before / len(ts),
            "after_fraction": (len(ts) - before) / len(ts),
            "likely_two_sessions": bool(best["gap_days"] >= 180 and before > 0 and before < len(ts)),
            "top_gaps_json": json.dumps(top),
        })
    report = pd.DataFrame(rows).sort_values("user_id")
    report.to_csv(OUT / "session_boundary_by_user.csv", index=False)
    summary = {
        "users_analyzed": int(len(report)),
        "users_with_gap_at_least_180_days": int(report.likely_two_sessions.sum()),
        "largest_gap_days_median": float(report.largest_gap_days.median()),
        "largest_gap_days_min": float(report.largest_gap_days.min()),
        "largest_gap_days_max": float(report.largest_gap_days.max()),
        "span_days_median": float(report.span_days.median()),
        "caution": "A large timestamp gap is a candidate session boundary; confirm against the dataset's session metadata or raw release documentation before treating it as definitive.",
    }
    (OUT / "session_boundary_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(report[["user_id", "span_days", "largest_gap_days", "second_gap_days", "before_fraction", "likely_two_sessions"]].sort_values("largest_gap_days", ascending=False).head(15).to_string(index=False))

if __name__ == "__main__":
    main()
