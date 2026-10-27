from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "clarkson2_attention_model"))
from train_attention import AttentionEncoder

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)

def embed(model, x, mean, std):
    with torch.no_grad(): return model(torch.from_numpy(((x - mean) / std).astype(np.float32))).numpy()

def main():
    ck = torch.load(RESULTS / "attention_model_best.pt", map_location="cpu", weights_only=False)
    model = AttentionEncoder(); model.classifier = torch.nn.Linear(32, len(ck["users"])); model.load_state_dict(ck["model"]); model.eval()
    mean, std = np.array(ck["mean"], dtype=np.float32), np.array(ck["std"], dtype=np.float32)
    base = np.load(RESULTS / "sequences.npz"); base_meta = pd.read_csv(RESULTS / "sequence_metadata.csv", dtype={"user_id": str})
    later = np.load(RESULTS / "remaining_stage_sequences.npz"); later_meta = pd.read_csv(RESULTS / "remaining_stage_metadata.csv", dtype={"user_id": str})
    z_base = embed(model, base["X"], mean, std); z_later = embed(model, later["X"], mean, std)
    train_mask = base_meta.split.eq("train").to_numpy()
    prototypes = {str(u): z_base[np.where(train_mask & (base_meta.user_id.astype(str).to_numpy() == str(u)))[0]].mean(0) for u in base_meta.loc[train_mask, "user_id"].astype(str).unique()}
    test_mask = base_meta.split.eq("test").to_numpy()
    threshold = 1.1885696649551392
    rows=[]
    for user, p in prototypes.items():
        test_dist = np.linalg.norm(z_base[test_mask & (base_meta.user_id.astype(str).to_numpy() == user)] - p, axis=1)
        for stage in ["stage_1", "stage_2", "stage_3"]:
            ix = later_meta.stage.eq(stage).to_numpy() & later_meta.user_id.astype(str).eq(user).to_numpy()
            dist = np.linalg.norm(z_later[ix] - p, axis=1)
            if len(dist) == 0: continue
            base_mean = float(np.mean(test_dist)) if len(test_dist) else np.nan
            later_mean = float(np.mean(dist))
            rows.append({"user_id": user, "stage": stage, "baseline_test_windows": len(test_dist), "stage_windows": len(dist), "baseline_mean_distance": base_mean, "stage_mean_distance": later_mean, "absolute_drift": later_mean-base_mean if np.isfinite(base_mean) else np.nan, "relative_drift": later_mean/base_mean-1 if np.isfinite(base_mean) and base_mean>0 else np.nan, "stage_frr_at_30day_test_threshold": float(np.mean(dist > threshold)), "stage_distance_std": float(np.std(dist))})
    out=pd.DataFrame(rows); out.to_csv(RESULTS/"user_drift_metrics.csv",index=False)
    valid=out[out.stage_windows>=5].copy(); valid["drift_rank_score"]=valid.absolute_drift
    overall=valid.groupby("user_id").agg(stages_with_data=("stage","count"),total_later_windows=("stage_windows","sum"),mean_absolute_drift=("absolute_drift","mean"),max_absolute_drift=("absolute_drift","max"),mean_relative_drift=("relative_drift","mean"),mean_later_frr=("stage_frr_at_30day_test_threshold","mean")).reset_index().sort_values("mean_absolute_drift",ascending=False)
    overall.to_csv(RESULTS/"user_drift_overall_ranking.csv",index=False)
    summary={"definition":"Drift is the increase in mean embedding distance from the first-30-day train enrollment prototype to later-stage windows, relative to the same user's first-30-day test distance.","threshold":threshold,"minimum_stage_windows_for_ranking":5,"users_with_any_later_metrics":int(out.user_id.nunique()),"users_ranked":int(overall.user_id.nunique()),"maximum_mean_absolute_drift_user":str(overall.iloc[0].user_id) if len(overall) else None,"maximum_mean_absolute_drift":float(overall.iloc[0].mean_absolute_drift) if len(overall) else None,"maximum_mean_relative_drift":float(overall.iloc[0].mean_relative_drift) if len(overall) else None}
    (RESULTS/"user_drift_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    top=overall.head(20).sort_values("mean_absolute_drift")
    fig,ax=plt.subplots(figsize=(10,7)); ax.barh(top.user_id.astype(str),top.mean_absolute_drift,color="#d95f02"); ax.axvline(0,color="black",linewidth=1); ax.set_xlabel("Mean distance increase from first-30-day test baseline"); ax.set_ylabel("User ID"); ax.set_title("Top user-level drift (minimum 5 windows per stage)"); fig.tight_layout(); fig.savefig(FIGURES/"top_user_drift.png",dpi=200); plt.close(fig)
    piv=valid.pivot_table(index="user_id",columns="stage",values="absolute_drift",aggfunc="mean"); top_ids=overall.head(30).user_id; piv=piv.reindex(top_ids)
    fig,ax=plt.subplots(figsize=(9,10)); sns.heatmap(piv,center=0,cmap="RdBu_r",annot=False,ax=ax); ax.set_title("User-level embedding drift by later stage"); ax.set_xlabel("Later stage"); ax.set_ylabel("User ID"); fig.tight_layout(); fig.savefig(FIGURES/"user_drift_heatmap.png",dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6)); sns.boxplot(data=valid,x="stage",y="absolute_drift",color="#7570b3",ax=ax); ax.axhline(0,color="black",linewidth=1); ax.set_title("Distribution of user-level drift by stage"); ax.set_ylabel("Embedding distance increase"); fig.tight_layout(); fig.savefig(FIGURES/"user_drift_by_stage_boxplot.png",dpi=200); plt.close(fig)
    print(json.dumps(summary,indent=2)); print(overall.head(15).to_string(index=False))

if __name__=="__main__": main()
