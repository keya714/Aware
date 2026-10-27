from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)

def main():
    training = json.loads((RESULTS / "attention_metrics.json").read_text(encoding="utf-8"))
    later = pd.read_csv(RESULTS / "remaining_stage_metrics.csv")
    ref = pd.DataFrame([
        {"period": "First 30 days\nadapt", "eer": training["verification"][0]["eer_approx"], "far": training["verification"][0]["far"], "frr": training["verification"][0]["frr"]},
        {"period": "First 30 days\ntest", "eer": training["verification"][1]["eer_approx"], "far": training["verification"][1]["far"], "frr": training["verification"][1]["frr"]},
    ])
    later_ref = later.rename(columns={"stage":"period", "eer":"eer", "far":"far", "frr":"frr"})[["period","eer","far","frr"]]
    combined = pd.concat([ref, later_ref], ignore_index=True)
    combined["eer_percent"] = combined.eer * 100
    combined["far_percent"] = combined.far * 100
    combined["frr_percent"] = combined.frr * 100
    combined.to_csv(RESULTS / "training_vs_later_stage_comparison.csv", index=False)
    x=np.arange(len(combined)); width=.25
    fig, ax=plt.subplots(figsize=(11,6))
    for i,(col,label,color) in enumerate([("eer_percent","EER","#386cb0"),("far_percent","FAR","#f0027f"),("frr_percent","FRR","#4daf4a")]):
        ax.bar(x+(i-1)*width,combined[col],width,label=label,color=color)
    ax.set_xticks(x,combined.period); ax.set_ylabel("Error rate (%)"); ax.set_ylim(0,max(combined[["eer_percent","far_percent","frr_percent"]].max())*1.25); ax.set_title("Attention-model performance: first 30 days versus later stages"); ax.grid(axis="y",alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES/"training_vs_later_stage_error_rates.png",dpi=200); plt.close(fig)
    base=float(combined.loc[combined.period=="First 30 days\ntest","eer"].iloc[0])
    later["eer_change_vs_30day_test_percentage_points"]=(later.eer-base)*100
    later.to_csv(RESULTS/"later_stage_eer_change_vs_training_test.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,5)); ax.bar(later.stage,later.eer_change_vs_30day_test_percentage_points,color="#984ea3"); ax.axhline(0,color="black",linewidth=1); ax.set_ylabel("EER change (percentage points)"); ax.set_title("Later-stage EER change relative to first-30-day test"); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/"later_stage_eer_change.png",dpi=200); plt.close(fig)
    print(combined[["period","eer_percent","far_percent","frr_percent"]].to_string(index=False))

if __name__=="__main__": main()
