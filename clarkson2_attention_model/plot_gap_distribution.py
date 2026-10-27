from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)

def main():
    report = pd.read_csv(RESULTS / "session_boundary_by_user.csv", dtype={"user_id": str})
    eligible = set(pd.read_csv(ROOT / "clarkson2_preprocessing" / "outputs" / "final_clean_dataset" / "eligible_users.txt", header=None, names=["user_id"], dtype=str).user_id)
    report["modeling_subset"] = report.user_id.isin(eligible)
    # 30-day bins make the long-gap distribution readable while retaining the full 0-600 day range.
    upper = max(600, int(np.ceil(report.largest_gap_days.max() / 30.0) * 30))
    bins = np.arange(0, upper + 30, 30)
    labels = [f"{int(bins[i])}-{int(bins[i+1]-1)}" for i in range(len(bins)-1)]
    counts_all, _ = np.histogram(report.largest_gap_days, bins=bins)
    counts_model, _ = np.histogram(report.loc[report.modeling_subset, "largest_gap_days"], bins=bins)
    counts = pd.DataFrame({"gap_bin_days": labels, "all_100_users": counts_all, "modeling_94_users": counts_model})
    counts.to_csv(RESULTS / "maximum_gap_distribution.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for ax, values, title, color in [
        (axes[0], report.largest_gap_days, "All 100 users", "#386cb0"),
        (axes[1], report.loc[report.modeling_subset, "largest_gap_days"], "94-user modeling subset", "#f0027f"),
    ]:
        ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.7)
        ax.axvline(180, color="black", linestyle="--", linewidth=1.2, label="180 days")
        ax.set_ylabel("Number of users")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    axes[1].set_xlabel("Maximum positive timestamp gap per user (days)")
    axes[1].set_xlim(0, upper)
    fig.suptitle("Distribution of users' maximum timestamp gaps", fontsize=14)
    fig.savefig(FIGURES / "maximum_gap_distribution.png", dpi=180)
    plt.close(fig)
    print(counts.to_string(index=False))
    print(f"Saved {FIGURES / 'maximum_gap_distribution.png'}")

if __name__ == "__main__":
    main()
