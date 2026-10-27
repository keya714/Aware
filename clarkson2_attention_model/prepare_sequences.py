from pathlib import Path
import gzip
import json
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "clarkson2_preprocessing" / "outputs" / "final_clean_dataset"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

EVENT_COLUMNS = ["hold_seconds", "press_to_press_seconds", "release_to_press_seconds", "inter_event_seconds"]

def main():
    global DATA, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    DATA, OUT = args.data_dir.resolve(), args.out_dir.resolve()
    OUT.mkdir(exist_ok=True, parents=True)
    windows = pd.read_csv(DATA / "windows_normalized.csv", dtype={"user_id": str})
    windows["user_id"] = windows["user_id"].astype(str)
    windows["window_index"] = windows["window_index"].astype(int)
    wanted = set(zip(windows.user_id, windows.segment_id.astype(int), windows.burst_id.astype(int), windows.window_index))

    sequences, metadata = [], []
    event_dir = DATA / "clean_events"
    for path in sorted(event_dir.glob("*.clean_events.tsv.gz")):
        user = path.name.split(".", 1)[0]
        user_rows = windows[windows.user_id == user]
        if user_rows.empty:
            continue
        with gzip.open(path, "rt") as fh:
            events = pd.read_csv(fh, sep="\t")
        user_window_count = 0
        for (segment, burst), group in events.groupby(["segment_id", "burst_id"], sort=True):
            group = group.sort_values("pair_index").reset_index(drop=True)
            n_windows = len(group) // 50
            for wi in range(n_windows):
                # The preprocessing pipeline numbers windows continuously per user;
                # window_index therefore does not reset at each burst.
                global_wi = user_window_count + wi
                key = (user, int(segment), int(burst), int(global_wi))
                if key not in wanted:
                    continue
                row = group.iloc[wi * 50:(wi + 1) * 50]
                x = row[EVENT_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
                # Timing values are positive and heavy-tailed; log1p preserves order and reduces outlier dominance.
                x = np.log1p(np.maximum(x, 0.0)).astype(np.float32)
                sequences.append(x)
                metadata.append(user_rows[(user_rows.segment_id == int(segment)) & (user_rows.burst_id == int(burst)) & (user_rows.window_index == global_wi)].iloc[0].to_dict())
            user_window_count += n_windows

    meta = pd.DataFrame(metadata)
    if len(sequences) != len(windows):
        missing = len(windows) - len(sequences)
        raise RuntimeError(f"Sequence reconstruction mismatch: {missing} windows missing (built {len(sequences)} of {len(windows)})")
    order = {str(w): i for i, w in enumerate(meta.window_id)}
    target_order = [order[w] for w in windows.window_id]
    arr = np.stack(sequences)[target_order]
    meta = meta.iloc[target_order].reset_index(drop=True)
    np.savez_compressed(OUT / "sequences.npz", X=arr)
    meta.to_csv(OUT / "sequence_metadata.csv", index=False)
    report = {"windows": int(len(arr)), "shape": list(arr.shape), "users": int(meta.user_id.nunique()), "features": EVENT_COLUMNS, "transform": "log1p(max(seconds, 0))"}
    (OUT / "sequence_build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
