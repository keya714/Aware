from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "clarkson2_30day_model"
FIGURES = MODEL / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

windows = pd.read_csv(MODEL / "outputs" / "final_clean_dataset" / "windows_normalized.csv")
features = ["z_hold_mean", "z_press_to_press_mean", "z_release_to_press_mean", "z_inter_event_mean"]
labels = ["Hold", "Press-to-press", "Release-to-press", "Inter-event"]

sns.set_theme(style="whitegrid", context="paper")
corr = windows[features].corr()
annot = corr.round(2).astype(object)
annot[corr.isna()] = "N/A"
cmap = sns.color_palette("vlag", as_cmap=True)
cmap.set_bad("#d9d9d9")
ax = sns.heatmap(
    corr,
    annot=annot,
    annot_kws={"fontsize": 8},
    fmt="",
    cmap=cmap,
    vmin=-1,
    vmax=1,
    square=True,
    xticklabels=labels,
    yticklabels=labels,
    cbar_kws={"label": "Pearson correlation"},
)
for row, col in zip(*np.where(corr.isna())):
    ax.text(col + 0.5, row + 0.5, "N/A", ha="center", va="center", color="#222222", fontsize=8)
ax.set_title("Correlation of normalized timing features")
plt.tight_layout()
plt.savefig(FIGURES / "feature_correlation_matrix.png", dpi=220, bbox_inches="tight")
plt.close()

train_counts = (
    pd.read_csv(MODEL / "results" / "sequence_metadata.csv")
    .query("split == 'train'")
    .groupby("user_id")
    .size()
    .reset_index(name="windows")
    .assign(split="train")
)
later = pd.read_csv(MODEL / "results" / "remaining_stage_metadata.csv")
stage_counts = later.groupby(["stage", "user_id"]).size().reset_index(name="windows")
stage_counts = stage_counts.rename(columns={"stage": "split"})
counts = pd.concat([train_counts, stage_counts], ignore_index=True)
order = ["train", "stage_1", "stage_2", "stage_3"]
counts["split"] = pd.Categorical(counts["split"], categories=order, ordered=True)
ax = sns.boxplot(data=counts, x="split", y="windows", color="#9ecae1", showfliers=False)
sns.stripplot(data=counts, x="split", y="windows", color="#08519c", size=2.5, alpha=0.45)
ax.set_title("Per-user usable-window distribution")
ax.set_xlabel("Chronological split")
ax.set_ylabel("Usable windows per user")
plt.tight_layout()
plt.savefig(FIGURES / "user_window_distribution.png", dpi=220, bbox_inches="tight")
plt.close()

hold = windows["hold_mean"].dropna()
fig, (hist_ax, box_ax) = plt.subplots(
    2, 1, figsize=(5.8, 5.2), gridspec_kw={"height_ratios": [4, 1]}
)
hist_ax.hist(hold.clip(upper=hold.quantile(0.99)), bins=40, color="#6baed6", edgecolor="white")
hist_ax.set_title("Univariate hold-time distribution")
hist_ax.set_xlabel("Mean hold time per 50-pair window (s; clipped at 99th percentile)")
hist_ax.set_ylabel("Number of windows")
box_ax.boxplot(hold, vert=False, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#9ecae1"})
box_ax.set_xlabel("Mean hold time per window (s)")
box_ax.set_yticks([])
box_ax.set_title("Boxplot without extreme outliers")
fig.tight_layout()
fig.savefig(FIGURES / "hold_time_distribution.png", dpi=220, bbox_inches="tight")
plt.close(fig)

thresholds = pd.read_csv(
    ROOT / "clarkson2_preprocessing" / "outputs" / "hold_threshold_sensitivity" / "hold_threshold_sensitivity.csv"
)
ax = sns.barplot(data=thresholds, x="threshold_seconds", y="kept_percent", color="#74c476")
ax.set_title("Retention under maximum hold-time thresholds")
ax.set_xlabel("Maximum hold time (s)")
ax.set_ylabel("Retained candidate pairs (%)")
ax.set_ylim(0, 65)
plt.tight_layout()
plt.savefig(FIGURES / "01_threshold_retention.png", dpi=220, bbox_inches="tight")
plt.close()
