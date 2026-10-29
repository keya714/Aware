"""Create the longitudinal EER/FAR/FRR drift figure from saved metrics."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "clarkson2_30day_model" / "figures" / "longitudinal_drift_metrics.png"
CANDIDATES = [
    ROOT / "clarkson2_30day_model" / "results" / "training_vs_later_stage_comparison.csv",
    ROOT / "clarkson2_30day_model" / "results" / "longitudinal_metrics.csv",
    ROOT / "clarkson2_30day_model" / "results" / "stage_metrics.csv",
    ROOT / "clarkson2_30day_model" / "results" / "longitudinal_metrics.json",
]

source = next((p for p in CANDIDATES if p.exists()), None)
if source is None:
    raise FileNotFoundError("No saved longitudinal metrics file was found.")
if source.suffix == ".json":
    frame = pd.DataFrame(json.loads(source.read_text(encoding="utf-8")))
else:
    frame = pd.read_csv(source)
frame.columns = [c.strip().lower() for c in frame.columns]
frame = frame.rename(columns={"split": "stage", "period": "stage", "fpr": "far", "fnr": "frr"})
required = {"stage", "eer", "far", "frr"}
missing = required - set(frame.columns)
if missing:
    raise ValueError(f"Metrics file is missing: {sorted(missing)}")
frame["stage"] = frame["stage"].replace({
    "test": "30-day test",
    "baseline": "30-day test",
    "First 30 days\\ntest": "30-day test",
    "First 30 days\ntest": "30-day test",
})
order = ["30-day test", "stage_1", "stage_2", "stage_3"]
frame["stage"] = pd.Categorical(frame["stage"], order, ordered=True)
frame = frame.sort_values("stage").dropna(subset=["stage"])
for col in ["eer", "far", "frr"]:
    frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame[col].max() <= 1.0:
        frame[col] *= 100.0

fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=220)
x = range(len(frame))
for col, label, marker, color in [
    ("eer", "EER", "o", "#1f77b4"),
    ("far", "FAR", "s", "#d62728"),
    ("frr", "FRR", "^", "#2ca02c"),
]:
    ax.plot(x, frame[col], marker=marker, linewidth=2, markersize=5, label=label, color=color)
    for i, value in enumerate(frame[col]):
        ax.annotate(f"{value:.1f}", (i, value), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7)
ax.set_xticks(list(x), [str(s).replace("_", " ").title() for s in frame["stage"]])
ax.set_ylabel("Error rate (%)")
ax.set_xlabel("Evaluation period")
ax.set_title("Longitudinal authentication error rates")
ax.grid(axis="y", alpha=.25)
ax.legend(ncol=3, frameon=False, loc="upper left")
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}")
