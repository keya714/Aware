from pathlib import Path
import sys
import gzip, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clarkson2_attention_model"))
from train_attention import AttentionEncoder

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "clarkson2_preprocessing" / "outputs" / "final_clean_dataset" / "clean_events"
RESULTS = Path(__file__).resolve().parent / "results"
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)

FEATURES = ["hold_seconds", "press_to_press_seconds", "release_to_press_seconds", "inter_event_seconds"]
DAY = 86400 * 1e7

def sequences_for_remaining():
    sequences, rows = [], []
    for path in sorted(SOURCE.glob("*.clean_events.tsv.gz")):
        user = path.name.split(".", 1)[0]
        with gzip.open(path, "rt") as fh: d = pd.read_csv(fh, sep="\t")
        d["press_timestamp_100ns"] = pd.to_numeric(d.press_timestamp_100ns, errors="coerce")
        d = d.dropna(subset=["press_timestamp_100ns"]).sort_values("press_timestamp_100ns").reset_index(drop=True)
        if len(d) < 2: continue
        first = d.press_timestamp_100ns.min(); remaining = d[d.press_timestamp_100ns > first + 30 * DAY].copy()
        if remaining.empty: continue
        lo, hi = remaining.press_timestamp_100ns.min(), remaining.press_timestamp_100ns.max()
        width = max(1.0, (hi - lo) / 3.0)
        remaining["stage"] = np.minimum(((remaining.press_timestamp_100ns - lo) / width).astype(int) + 1, 3)
        for stage in [1, 2, 3]:
            sd = remaining[remaining.stage == stage]
            for (segment, burst), group in sd.groupby(["segment_id", "burst_id"], sort=True):
                group = group.sort_values("pair_index").reset_index(drop=True)
                for wi in range(len(group) // 50):
                    row = group.iloc[wi*50:(wi+1)*50]
                    x = row[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)
                    sequences.append(np.log1p(np.maximum(x, 0)).astype(np.float32))
                    rows.append({"user_id": user, "stage": f"stage_{stage}", "stage_number": stage, "segment_id": int(segment), "burst_id": int(burst), "window_index": wi, "start_day": float((row.press_timestamp_100ns.iloc[0]-first)/DAY), "end_day": float((row.press_timestamp_100ns.iloc[-1]-first)/DAY)})
    return np.stack(sequences).astype(np.float32), pd.DataFrame(rows)

def embed(model, x, mean, std):
    x = (x - mean) / std
    with torch.no_grad(): return model(torch.from_numpy(x)).numpy()

def stage_metrics(emb, meta, prototypes):
    out=[]
    for stage, g in meta.groupby("stage", sort=True):
        genuine, impostor = [], []
        for i, row in g.reset_index().iterrows():
            u=str(row.user_id)
            if u not in prototypes: continue
            q=emb[row["index"]]
            genuine.append(float(np.linalg.norm(q-prototypes[u])))
            for v, p in prototypes.items():
                if v != u: impostor.append(float(np.linalg.norm(q-p)))
        scores=np.array(genuine+impostor); labels=np.array([1]*len(genuine)+[0]*len(impostor))
        best=None
        for t in np.linspace(scores.min(), scores.max(), 2000):
            pred=scores<=t; far=float(((pred==1)&(labels==0)).sum()/max(1,(labels==0).sum())); frr=float(((pred==0)&(labels==1)).sum()/max(1,(labels==1).sum())); item=(abs(far-frr),t,far,frr)
            if best is None or item[0]<best[0]: best=item
        out.append({"stage":stage,"users_with_queries":int(g.loc[g.user_id.astype(str).isin(prototypes), "user_id"].astype(str).nunique()),"windows":int(len(g)),"genuine_trials":len(genuine),"impostor_trials":len(impostor),"eer":(best[2]+best[3])/2,"threshold":best[1],"far":best[2],"frr":best[3]})
    return pd.DataFrame(out)

def main():
    base=RESULTS
    d=np.load(base/"sequences.npz"); m=pd.read_csv(base/"sequence_metadata.csv",dtype={"user_id":str})
    ck=torch.load(base/"attention_model_best.pt",map_location="cpu",weights_only=False)
    model=AttentionEncoder(); model.classifier=torch.nn.Linear(32,len(ck["users"])); model.load_state_dict(ck["model"]); model.eval()
    mean,std=np.array(ck["mean"],dtype=np.float32),np.array(ck["std"],dtype=np.float32); train_mask=m.split=="train"; z=embed(model,d["X"].astype(np.float32),mean,std)
    prototypes={str(u):z[np.where(train_mask & (m.user_id.astype(str).to_numpy()==str(u)))[0]].mean(0) for u in m.loc[train_mask,"user_id"].astype(str).unique()}
    x, meta=sequences_for_remaining(); meta.to_csv(RESULTS/"remaining_stage_metadata.csv",index=False); np.savez_compressed(RESULTS/"remaining_stage_sequences.npz",X=x)
    ez=embed(model,x,mean,std); metrics=stage_metrics(ez,meta,prototypes); metrics.to_csv(RESULTS/"remaining_stage_metrics.csv",index=False)
    metrics.to_json(RESULTS/"remaining_stage_metrics.json",orient="records",indent=2)
    summary={"model":"attention_model_best.pt","enrollment":"train-split windows from first 30 days","stage_definition":"remaining timestamps divided into three equal chronological intervals per user","remaining_windows":int(len(meta)),"remaining_users":int(meta.user_id.nunique()),"users_with_30day_enrollment":len(prototypes)}
    (RESULTS/"remaining_stage_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(9,5)); xx=np.arange(len(metrics)); w=.25; ax.bar(xx-w,metrics.eer,width=w,label="EER"); ax.bar(xx,metrics.far,width=w,label="FAR"); ax.bar(xx+w,metrics.frr,width=w,label="FRR"); ax.set_xticks(xx,metrics.stage); ax.set_ylim(0,1); ax.set_ylabel("Rate"); ax.set_title("Frozen 30-day attention model on later stages"); ax.legend(); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/"remaining_stage_error_rates.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5)); counts=meta.groupby("stage").size().reindex(["stage_1","stage_2","stage_3"]); ax.bar(counts.index,counts.values,color="#386cb0"); ax.set_ylabel("Windows"); ax.set_title("Later-stage windows after the first 30 days"); fig.tight_layout(); fig.savefig(FIGURES/"remaining_stage_window_counts.png",dpi=180); plt.close(fig)
    print(summary); print(metrics.to_string(index=False))

if __name__=="__main__": main()
