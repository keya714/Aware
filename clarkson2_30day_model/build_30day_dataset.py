from pathlib import Path
import gzip, json, shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "clarkson2_preprocessing" / "outputs" / "final_clean_dataset" / "clean_events"
OUT = Path(__file__).resolve().parent / "outputs" / "final_clean_dataset"
EVENT_OUT = OUT / "clean_events"
EVENT_OUT.mkdir(parents=True, exist_ok=True)

WINDOW_PAIRS = 50
DAY_100NS = 30 * 24 * 60 * 60 * 10_000_000
FEATURES = ["hold_seconds", "press_to_press_seconds", "release_to_press_seconds", "inter_event_seconds"]

def summary(row):
    def vals(name): return pd.to_numeric(row[name], errors="coerce").dropna().to_numpy(float)
    h, pp, rp, ie = [vals(x) for x in FEATURES]
    def stat(a, q=None): return float(np.quantile(a, q)) if len(a) and q is not None else (float(np.mean(a)) if len(a) else np.nan)
    out = {"pair_count": len(row), "hold_mean": stat(h), "hold_std": float(np.std(h)) if len(h) else np.nan, "hold_median": stat(h,.5), "hold_q25": stat(h,.25), "hold_q75": stat(h,.75),
           "press_to_press_mean": stat(pp), "press_to_press_std": float(np.std(pp)) if len(pp) else np.nan, "release_to_press_mean": stat(rp), "release_to_press_std": float(np.std(rp)) if len(rp) else np.nan,
           "inter_event_mean": stat(ie), "inter_event_std": float(np.std(ie)) if len(ie) else np.nan}
    out["typing_rate"] = 60.0 / out["press_to_press_mean"] if out["press_to_press_mean"] > 0 else np.nan
    return out

def main():
    all_windows, user_report = [], []
    for path in sorted(SOURCE.glob("*.clean_events.tsv.gz")):
        user = path.name.split(".", 1)[0]
        with gzip.open(path, "rt") as fh: d = pd.read_csv(fh, sep="\t")
        d["press_timestamp_100ns"] = pd.to_numeric(d["press_timestamp_100ns"], errors="coerce")
        d = d.dropna(subset=["press_timestamp_100ns"]).sort_values("press_timestamp_100ns").reset_index(drop=True)
        first = int(d.press_timestamp_100ns.min()) if len(d) else None
        original_last = int(d.press_timestamp_100ns.max()) if len(d) else None
        if first is not None: d = d[d.press_timestamp_100ns <= first + DAY_100NS].copy()
        d.to_csv(EVENT_OUT / f"{user}.clean_events.tsv.gz", sep="\t", index=False, compression="gzip")
        window_count = 0
        for (segment, burst), group in d.groupby(["segment_id", "burst_id"], sort=True):
            group = group.sort_values("pair_index").reset_index(drop=True)
            for local_i in range(len(group) // WINDOW_PAIRS):
                row = group.iloc[local_i * WINDOW_PAIRS:(local_i + 1) * WINDOW_PAIRS]
                s = summary(row)
                s.update({"user_id": user, "segment_id": int(segment), "burst_id": int(burst), "window_index": window_count,
                          "start_elapsed_seconds": float((row.press_timestamp_100ns.iloc[0] - first) / 1e7), "end_elapsed_seconds": float((row.press_timestamp_100ns.iloc[-1] - first) / 1e7)})
                all_windows.append(s); window_count += 1
        user_report.append({"user_id": user, "first_timestamp_100ns": first, "available_days": float((original_last-first)/1e7/86400) if first is not None and original_last is not None else 0.0, "pairs_after_30day_cap": int(len(d)), "windows": window_count})
    w = pd.DataFrame(all_windows)
    if len(w):
        w["window_id"] = w.groupby("user_id").cumcount().map(lambda i: f"_w{i:06d}")
        w["window_id"] = w["user_id"].astype(str) + w["window_id"]
        parts=[]
        for user, g in w.groupby("user_id", sort=True):
            idx=g.index.to_numpy(); n=len(idx); ntr=max(1,int(np.floor(.6*n))); nad=int(np.floor(.2*n));
            if n >= 3: nad=max(1,nad); nte=max(1,n-ntr-nad)
            else: nte=0
            split=np.array(["train"]*n, dtype=object); split[ntr:ntr+nad]="adapt"; split[ntr+nad:ntr+nad+nte]="test"; parts.append(pd.Series(split,index=idx))
        w["split"] = pd.concat(parts).sort_index()
    else: w["window_id"] = []; w["split"] = []
    for c in ["hold_mean","hold_std","hold_median","hold_q25","hold_q75","press_to_press_mean","press_to_press_std","release_to_press_mean","release_to_press_std","inter_event_mean","inter_event_std","typing_rate"]:
        tr=pd.to_numeric(w.loc[w.split=="train",c],errors="coerce"); mu=tr.mean(); sd=tr.std(); w["z_"+c]=(pd.to_numeric(w[c],errors="coerce")-mu)/(sd if pd.notna(sd) and sd>1e-9 else 1.0)
    cols=["user_id","window_id","segment_id","burst_id","window_index","start_elapsed_seconds","end_elapsed_seconds","pair_count","hold_mean","hold_std","hold_median","hold_q25","hold_q75","press_to_press_mean","press_to_press_std","release_to_press_mean","release_to_press_std","inter_event_mean","inter_event_std","typing_rate","split"]+[c for c in w.columns if c.startswith("z_")]
    w[cols].to_csv(OUT / "windows_normalized.csv",index=False)
    users=pd.DataFrame(user_report); users.to_csv(OUT / "user_30day_report.csv",index=False); users.user_id.to_csv(OUT/"all_users.txt",index=False,header=False)
    (OUT/"pipeline_config.json").write_text(json.dumps({"cap_days":30,"window_pairs":50,"window_stride":50,"train_fraction":.6,"adapt_fraction":.2,"source":"clean_events after original cleaning"},indent=2),encoding="utf-8")
    report={"input_users":100,"users_with_windows":int((users.windows>0).sum()),"users_with_at_least_3_windows":int((users.windows>=3).sum()),"windows":int(len(w)),"pairs":int(users.pairs_after_30day_cap.sum()),"available_days_ge_30":int((users.available_days>=30).sum()),"available_days_lt_30":int((users.available_days<30).sum()),"note":"All 100 source users are retained; users with less than 30 days keep their available data and are flagged."}
    (OUT/"30day_quality_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2)); print(users.sort_values('available_days').head(10).to_string(index=False))

if __name__=="__main__": main()
