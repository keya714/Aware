"""ROME-style rank-one model editing for the 30-day Clarkson II model.

The edit is applied to the second Transformer feed-forward projection using
the closed-form ROME update. Stage 1 is the adaptation buffer. Stage 2 is the
primary held-out test and Stage 3 is the secondary held-out test.
"""
from pathlib import Path
import copy
import json
import random
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
OUT = RESULTS / "model_editing"
OUT.mkdir(parents=True, exist_ok=True)
FIGURES = OUT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "clarkson2_attention_model"))
from train_attention import AttentionEncoder

SEED = 228
THRESHOLD = 1.1885696649551392
RETENTION_TOLERANCE_PP = 2.0
FAR_TOLERANCE_PP = 1.0


def load_model():
    ck = torch.load(RESULTS / "attention_model_best.pt", map_location="cpu", weights_only=False)
    model = AttentionEncoder()
    model.classifier = nn.Linear(32, len(ck["users"]))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, np.asarray(ck["mean"], dtype=np.float32), np.asarray(ck["std"], dtype=np.float32)


def build_eligibility(lm):
    counts = lm.groupby(["user_id", "stage"]).size().unstack(fill_value=0)
    out = counts.reindex(columns=["stage_1", "stage_2", "stage_3"], fill_value=0).reset_index()
    out["user_id"] = out.user_id.astype(str)
    out["eligible"] = (out.stage_1 >= 5) & (out.stage_2 >= 5) & (out.stage_3 >= 5)
    return out


def stage1_ranking(base_model, base_arr, base_meta, later_arr, later_meta, mean, std, prototypes):
    """Rank users using only Stage 1 drift; later stages never enter selection."""
    train_x, train_mask = tensors(base_arr, base_meta, mean, std, split="train")
    test_x, test_mask = tensors(base_arr, base_meta, mean, std, split="test")
    stage1_x, stage1_mask = tensors(later_arr, later_meta, mean, std, stage="stage_1")
    train_users = base_meta.loc[train_mask, "user_id"].astype(str).to_numpy()
    test_users = base_meta.loc[test_mask, "user_id"].astype(str).to_numpy()
    stage1_users = later_meta.loc[stage1_mask, "user_id"].astype(str).to_numpy()
    test_z = embeddings(base_model, test_x).numpy()
    stage1_z = embeddings(base_model, stage1_x).numpy()
    rows = []
    for user, prototype in prototypes.items():
        baseline = np.linalg.norm(test_z[test_users == user] - prototype, axis=1)
        later = np.linalg.norm(stage1_z[stage1_users == user] - prototype, axis=1)
        if len(baseline) and len(later):
            rows.append({"user_id": user, "baseline_test_windows": len(baseline), "stage_1_windows": len(later), "baseline_mean_distance": float(baseline.mean()), "stage_1_mean_distance": float(later.mean()), "stage_1_absolute_drift": float(later.mean() - baseline.mean()), "stage_1_relative_drift": float(later.mean() / max(baseline.mean(), 1e-8) - 1)})
    return pd.DataFrame(rows).sort_values("stage_1_absolute_drift", ascending=False).reset_index(drop=True)


def tensors(arr, meta, mean, std, user=None, split=None, stage=None):
    mask = np.ones(len(meta), dtype=bool)
    if user is not None: mask &= meta.user_id.astype(str).to_numpy() == str(user)
    if split is not None: mask &= meta.split.to_numpy() == split
    if stage is not None: mask &= meta.stage.to_numpy() == stage
    return torch.from_numpy(((arr[mask] - mean) / std).astype(np.float32)), mask


def embeddings(model, x):
    with torch.no_grad():
        return model(x)


def prototypes(model, x, meta):
    z = embeddings(model, x).numpy()
    out = {}
    for user in meta.user_id.astype(str).unique():
        out[str(user)] = z[meta.user_id.astype(str).to_numpy() == str(user)].mean(0)
    return out


def distances(model, x, users, proto):
    z = embeddings(model, x).numpy()
    return {u: np.linalg.norm(z - proto[u], axis=1) for u in users}


def capture_keys(model, x):
    """Capture inputs to layer-2 linear2, the ROME key activations."""
    keys = []
    layer = model.encoder.layers[1].linear2
    handle = layer.register_forward_pre_hook(lambda _m, inputs: keys.append(inputs[0].detach()))
    with torch.no_grad():
        model(x)
    handle.remove()
    return torch.cat(keys, dim=0).reshape(-1, layer.in_features)


def covariance(keys, ridge=1e-4):
    c = keys.T @ keys / max(1, keys.shape[0])
    return c + ridge * torch.eye(c.shape[0], dtype=c.dtype)


def optimize_value(model, x, prototype, replay_x, replay_reference, steps):
    """Optimize only ROME's desired value vector; weights stay frozen."""
    layer = model.encoder.layers[1].linear2
    key = capture_keys(model, x).mean(0)
    with torch.no_grad():
        initial = layer.weight @ key + layer.bias
    value = nn.Parameter(initial.detach().clone())
    optimizer = torch.optim.Adam([value], lr=0.02)

    def replace_output(_module, _inputs, output):
        return value.view(1, 1, -1).expand_as(output)

    handle = layer.register_forward_hook(replace_output)
    for _ in range(steps):
        optimizer.zero_grad()
        target_z = model(x)
        replay_z = model(replay_x)
        loss = torch.mean(torch.sum((target_z - prototype) ** 2, dim=1))
        loss = loss + 2.0 * F.mse_loss(replay_z, replay_reference)
        loss = loss + 0.01 * F.mse_loss(value, initial)
        loss.backward()
        optimizer.step()
    handle.remove()
    return key.detach(), value.detach()


def apply_rome(model, key, value, c):
    edited = copy.deepcopy(model)
    layer = edited.encoder.layers[1].linear2
    with torch.no_grad():
        ck = torch.linalg.solve(c, key)
        current = layer.weight @ key + layer.bias
        denominator = torch.dot(ck, key).clamp_min(1e-8)
        delta = torch.outer(value - current, ck) / denominator
        layer.weight.add_(delta)
    return edited, delta


def score_user(model, x, query_users, proto, selected):
    z = embeddings(model, x).numpy()
    rows = []
    for user in selected:
        genuine = np.linalg.norm(z[query_users == user] - proto[user], axis=1)
        impostor = np.linalg.norm(z[query_users != user] - proto[user], axis=1)
        rows.append({
            "user_id": user, "genuine_windows": int(len(genuine)), "impostor_windows": int(len(impostor)),
            "mean_genuine_distance": float(genuine.mean()) if len(genuine) else np.nan,
            "frr": float(np.mean(genuine > THRESHOLD)) if len(genuine) else np.nan,
            "far": float(np.mean(impostor <= THRESHOLD)) if len(impostor) else np.nan,
        })
    return rows


def score_user_embeddings(z, query_users, proto, selected):
    rows = []
    for user in selected:
        genuine = np.linalg.norm(z[query_users == user] - proto[user], axis=1)
        impostor = np.linalg.norm(z[query_users != user] - proto[user], axis=1)
        rows.append({"user_id": user, "genuine_windows": int(len(genuine)), "impostor_windows": int(len(impostor)), "mean_genuine_distance": float(genuine.mean()) if len(genuine) else np.nan, "frr": float(np.mean(genuine > THRESHOLD)) if len(genuine) else np.nan, "far": float(np.mean(impostor <= THRESHOLD)) if len(impostor) else np.nan})
    return rows


def edit_metrics(edited_model, base_arr, base_meta, later_arr, later_meta, selected, proto, base_eval):
    rows = []
    for period, arr, meta, split, stage in [
        ("early_test", base_arr, base_meta, "test", None),
        ("stage_2", later_arr, later_meta, None, "stage_2"),
        ("stage_3", later_arr, later_meta, None, "stage_3"),
    ]:
        if split:
            x, mask = tensors(arr, meta, MEAN, STD, split=split)
        else:
            x, mask = tensors(arr, meta, MEAN, STD, stage=stage)
        q = meta.loc[mask, "user_id"].astype(str).to_numpy()
        before = score_user_embeddings(base_eval[period]["z"], q, proto, selected)
        after = score_user(edited_model, x, q, proto, selected)
        for b, a in zip(before, after):
            user = b["user_id"]
            rows.append({"period": period, "user_id": user, "before_frr": b["frr"], "after_frr": a["frr"], "before_far": b["far"], "after_far": a["far"], "before_distance": b["mean_genuine_distance"], "after_distance": a["mean_genuine_distance"], "genuine_windows": b["genuine_windows"], "impostor_windows": b["impostor_windows"], "frr_reduction_pp": (b["frr"] - a["frr"]) * 100, "far_increase_pp": (a["far"] - b["far"]) * 100, "distance_reduction": b["mean_genuine_distance"] - a["mean_genuine_distance"], "stage_1_adaptation_windows": int(((later_meta.user_id.astype(str) == user) & (later_meta.stage == "stage_1")).sum()), "stage_2_evaluation_windows": int(((later_meta.user_id.astype(str) == user) & (later_meta.stage == "stage_2")).sum()), "stage_3_evaluation_windows": int(((later_meta.user_id.astype(str) == user) & (later_meta.stage == "stage_3")).sum())})
    return pd.DataFrame(rows)


def edit_cost(base_model, edited_model):
    base_state = base_model.state_dict()
    changed = 0
    displacement = 0.0
    total = 0
    for name, value in edited_model.state_dict().items():
        if name not in base_state or value.shape != base_state[name].shape:
            continue
        delta = value.detach().float() - base_state[name].detach().float()
        displacement += float(torch.sum(delta * delta))
        total += value.numel()
        changed += int(torch.count_nonzero(delta).item())
    return changed, total, float(np.sqrt(displacement))


def add_success_flags(result):
    result = result.copy()
    result["delta_rank"] = result["delta_rank"].astype(int)
    result["stage_2_success"] = False
    result["user_success"] = False
    for user in result.user_id.astype(str).unique():
        rows = result[result.user_id.astype(str) == user].set_index("period")
        if not all(p in rows.index for p in ["early_test", "stage_2", "stage_3"]):
            continue
        early = rows.loc["early_test"]
        s2 = rows.loc["stage_2"]
        s3 = rows.loc["stage_3"]
        stage2_ok = (s2.frr_reduction_pp >= 5.0 and s2.far_increase_pp <= 1.0 and early.after_frr - early.before_frr <= 0.02 and s3.frr_reduction_pp >= 0.0 and s3.far_increase_pp <= 1.0 and rows.delta_rank.eq(1).all())
        result.loc[result.user_id.astype(str) == user, "stage_2_success"] = bool(stage2_ok)
        result.loc[result.user_id.astype(str) == user, "user_success"] = bool(stage2_ok)
    return result


def bootstrap_summary(values, seed=228, samples=10000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": np.nan, "lower_95": np.nan, "upper_95": np.nan, "n_users": 0}
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "lower_95": float(np.quantile(draws, .025)), "upper_95": float(np.quantile(draws, .975)), "n_users": int(len(values))}


def population_outputs(result, ranking, eligibility):
    per_user = result[result.period.isin(["early_test", "stage_2", "stage_3"])].copy()
    per_user.to_csv(OUT / "rome_per_user_metrics.csv", index=False)
    summary_rows = []
    ci_rows = []
    tests = {"primary_stage": "stage_2", "secondary_stage": "stage_3", "n_eligible_users": int(eligibility.eligible.sum()), "n_users_with_results": int(per_user.user_id.nunique())}
    for period in ["early_test", "stage_2", "stage_3"]:
        d = per_user[per_user.period == period]
        if d.empty:
            continue
        pooled_before_frr = d.before_frr.mul(d.genuine_windows).sum() / max(1, d.genuine_windows.sum())
        pooled_after_frr = d.after_frr.mul(d.genuine_windows).sum() / max(1, d.genuine_windows.sum())
        pooled_before_far = d.before_far.mul(d.impostor_windows).sum() / max(1, d.impostor_windows.sum())
        pooled_after_far = d.after_far.mul(d.impostor_windows).sum() / max(1, d.impostor_windows.sum())
        summary_rows.append({"period": period, "users": int(d.user_id.nunique()), "macro_before_frr": float(d.before_frr.mean()), "macro_after_frr": float(d.after_frr.mean()), "macro_frr_reduction_pp": float(d.frr_reduction_pp.mean()), "median_frr_reduction_pp": float(d.frr_reduction_pp.median()), "macro_before_far": float(d.before_far.mean()), "macro_after_far": float(d.after_far.mean()), "macro_far_increase_pp": float(d.far_increase_pp.mean()), "pooled_before_frr": float(pooled_before_frr), "pooled_after_frr": float(pooled_after_frr), "pooled_before_far": float(pooled_before_far), "pooled_after_far": float(pooled_after_far), "success_rate": float(d.user_success.mean())})
        for metric in ["frr_reduction_pp", "far_increase_pp", "distance_reduction"]:
            ci = bootstrap_summary(d[metric].to_numpy())
            ci.update({"period": period, "metric": metric})
            ci_rows.append(ci)
        if period in ["stage_2", "stage_3"]:
            values = d.frr_reduction_pp.dropna().to_numpy()
            try:
                stat = wilcoxon(values, alternative="greater", zero_method="wilcox")
                tests[period] = {"wilcoxon_statistic": float(stat.statistic), "p_value": float(stat.pvalue), "n_users": int(len(values))}
            except ValueError:
                tests[period] = {"wilcoxon_statistic": np.nan, "p_value": np.nan, "n_users": int(len(values)), "note": "No nonzero paired differences"}
    pd.DataFrame(summary_rows).to_csv(OUT / "rome_population_summary.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(OUT / "rome_bootstrap_confidence_intervals.csv", index=False)
    (OUT / "rome_statistical_tests.json").write_text(json.dumps(tests, indent=2, allow_nan=True), encoding="utf-8")
    ranking.to_csv(OUT / "stage1_drift_ranking.csv", index=False)
    eligibility.to_csv(OUT / "eligible_users.csv", index=False)
    return pd.DataFrame(summary_rows), pd.DataFrame(ci_rows)


def save_population_plots(result, ranking, summary, ci):
    if result.empty:
        return
    periods = ["stage_2", "stage_3"]
    labels = {"stage_2": "Stage 2", "stage_3": "Stage 3"}
    for metric, ylabel, filename, title in [("frr_reduction_pp", "FRR reduction (percentage points)", "rome_population_frr_distribution.png", "Per-user FRR recovery"), ("far_increase_pp", "FAR increase (percentage points)", "rome_population_far_distribution.png", "Per-user FAR change")]:
        data = [result.loc[result.period == p, metric].dropna().to_numpy() for p in periods]
        fig, ax = plt.subplots(figsize=(8, 5)); ax.boxplot(data, tick_labels=[labels[p] for p in periods], showmeans=True); ax.axhline(0, color="black", linewidth=1); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(FIGURES / filename, dpi=200); plt.close(fig)
    success = result[result.period == "stage_2"].drop_duplicates("user_id").sort_values("user_success")
    fig, ax = plt.subplots(figsize=(12, max(5, .25 * len(success)))); ax.barh(success.user_id.astype(str), success.user_success.astype(int), color=np.where(success.user_success, "#4daf4a", "#e41a1c")); ax.set_xlim(0, 1.1); ax.set_xlabel("Meets all recovery, FAR, retention, and rank constraints (1=yes)"); ax.set_ylabel("User"); ax.set_title("ROME success by user"); ax.grid(axis="x", alpha=.25); fig.tight_layout(); fig.savefig(FIGURES / "rome_success_by_user.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(periods)); s = summary.set_index("period").reindex(periods); ax.bar(x - .18, s.macro_before_frr * 100, .36, label="Frozen FRR", color="#8c8c8c"); ax.bar(x + .18, s.macro_after_frr * 100, .36, label="ROME FRR", color="#386cb0"); ax.set_xticks(x, [labels[p] for p in periods]); ax.set_ylabel("Macro-average FRR (%)"); ax.set_title("Population-average FRR with equal user weighting"); ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(FIGURES / "rome_population_macro_frr.png", dpi=200); plt.close(fig)
    r = ranking.merge(result[result.period == "stage_2"][['user_id', 'frr_reduction_pp']], on="user_id", how="inner")
    fig, ax = plt.subplots(figsize=(8, 6)); ax.scatter(r.stage_1_absolute_drift, r.frr_reduction_pp, c=np.where(r.frr_reduction_pp >= 5, "#4daf4a", "#e41a1c"), s=55); ax.axhline(5, color="#d95f02", linestyle="--", label="5 pp recovery target"); ax.set_xlabel("Stage 1 absolute drift"); ax.set_ylabel("Stage 2 FRR reduction (percentage points)"); ax.set_title("Stage 1 drift versus ROME recovery"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGURES / "rome_stage1_drift_vs_recovery.png", dpi=200); plt.close(fig)


def save_plots(result):
    """Create human-readable figures from the saved ROME metrics table."""
    if result.empty:
        return
    users = list(result.user_id.astype(str).drop_duplicates())
    periods = ["early_test", "stage_2", "stage_3"]
    period_labels = {"early_test": "Early test", "stage_2": "Stage 2", "stage_3": "Stage 3"}

    def grouped_metric(metric, filename, title, ylabel, tolerance=None):
        fig, axes = plt.subplots(max(1, len(users)), 1, figsize=(11, max(4, 3.2 * len(users))), squeeze=False)
        for ax, user in zip(axes[:, 0], users):
            d = result[result.user_id.astype(str) == user].set_index("period").reindex(periods)
            x = np.arange(len(periods)); width = .36
            ax.bar(x - width / 2, d[f"before_{metric}"].to_numpy() * 100, width, label="Frozen model", color="#8c8c8c")
            ax.bar(x + width / 2, d[f"after_{metric}"].to_numpy() * 100, width, label="ROME rank-one edit", color="#386cb0")
            ax.set_xticks(x, [period_labels[p] for p in periods])
            ax.set_ylabel(ylabel)
            ax.set_title(f"User {user}")
            ax.grid(axis="y", alpha=.25)
            if tolerance is not None:
                delta = (d[f"after_{metric}"] - d[f"before_{metric}"]).max() * 100
                ax.axhline(d[f"before_{metric}"].max() * 100 + tolerance, color="#d95f02", linestyle="--", label=f"+{tolerance:.0f} pp tolerance")
            ax.legend(loc="best")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=200)
        plt.close(fig)

    grouped_metric("frr", "rome_frr_before_after.png", "FRR before and after ROME editing", "FRR (%)")
    grouped_metric("far", "rome_far_before_after.png", "FAR before and after ROME editing", "FAR (%)", tolerance=1.0)

    fig, axes = plt.subplots(max(1, len(users)), 1, figsize=(11, max(4, 3.2 * len(users))), squeeze=False)
    for ax, user in zip(axes[:, 0], users):
        d = result[result.user_id.astype(str) == user].set_index("period").reindex(periods)
        x = np.arange(len(periods)); width = .36
        ax.bar(x - width / 2, d.before_distance, width, label="Frozen model", color="#8c8c8c")
        ax.bar(x + width / 2, d.after_distance, width, label="ROME rank-one edit", color="#4daf4a")
        ax.set_xticks(x, [period_labels[p] for p in periods]); ax.set_ylabel("Mean genuine distance")
        ax.set_title(f"User {user}"); ax.grid(axis="y", alpha=.25); ax.legend(loc="best")
    fig.suptitle("Genuine embedding distance before and after ROME")
    fig.tight_layout(); fig.savefig(FIGURES / "rome_distance_recovery.png", dpi=200); plt.close(fig)

    trade = result.copy()
    trade["far_change_pp"] = (trade.after_far - trade.before_far) * 100
    trade["frr_reduction_pp"] = (trade.before_frr - trade.after_frr) * 100
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in trade.iterrows():
        ax.scatter(row.far_change_pp, row.frr_reduction_pp, s=70)
        ax.annotate(f"{row.user_id}\n{period_labels.get(row.period, row.period)}", (row.far_change_pp, row.frr_reduction_pp), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axvline(1.0, color="#d95f02", linestyle="--", label="FAR tolerance: +1 pp")
    ax.axhline(0.0, color="black", linewidth=1, label="No FRR improvement")
    ax.set_xlabel("FAR change (percentage points)"); ax.set_ylabel("FRR reduction (percentage points)")
    ax.set_title("ROME recovery versus security trade-off"); ax.grid(alpha=.25); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(FIGURES / "rome_recovery_security_tradeoff.png", dpi=200); plt.close(fig)

    cost = result.drop_duplicates(["user_id", "method"]).copy()
    fig, ax1 = plt.subplots(figsize=(10, 5)); x = np.arange(len(cost)); width = .36
    ax1.bar(x - width / 2, cost.weight_displacement_l2, width, label="Weight displacement L2", color="#7570b3")
    ax1.set_ylabel("Weight displacement L2"); ax1.set_xticks(x, cost.user_id.astype(str)); ax1.set_xlabel("Edited user")
    ax2 = ax1.twinx(); ax2.bar(x + width / 2, cost.changed_parameters, width, label="Changed parameters", color="#e6ab02", alpha=.8); ax2.set_ylabel("Changed parameters")
    ax1.set_title("ROME rank-one edit cost"); ax1.grid(axis="y", alpha=.25)
    handles1, labels1 = ax1.get_legend_handles_labels(); handles2, labels2 = ax2.get_legend_handles_labels(); ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    fig.tight_layout(); fig.savefig(FIGURES / "rome_edit_cost.png", dpi=200); plt.close(fig)


def main():
    global MEAN, STD, OUT, FIGURES
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=None, help="optional pilot limit; default evaluates every eligible user")
    ap.add_argument("--v-steps", type=int, default=20, help="optimization steps for ROME's desired value vector")
    ap.add_argument("--replay-size", type=int, default=256)
    ap.add_argument("--adaptation-size", type=int, default=256, help="maximum Stage 1 windows used per user's ROME value solve")
    ap.add_argument("--output-dir", type=Path, default=Path("results/model_editing"), help="directory for this run's tables, figures, and checkpoints")
    args = ap.parse_args()
    OUT = (HERE / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES = OUT / "figures"
    FIGURES.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.mha.set_fastpath_enabled(False)
    base_model, MEAN, STD = load_model()
    base = np.load(RESULTS / "sequences.npz")
    bm = pd.read_csv(RESULTS / "sequence_metadata.csv", dtype={"user_id": str})
    later = np.load(RESULTS / "remaining_stage_sequences.npz")
    lm = pd.read_csv(RESULTS / "remaining_stage_metadata.csv", dtype={"user_id": str})
    eligibility = build_eligibility(lm)
    eligible_users = eligibility.loc[eligibility.eligible, "user_id"].astype(str).tolist()
    train_x, train_mask = tensors(base["X"], bm, MEAN, STD, split="train")
    train_meta = bm.loc[train_mask].reset_index(drop=True)
    proto = prototypes(base_model, train_x, train_meta)
    ranking = stage1_ranking(base_model, base["X"], bm, later["X"], lm, MEAN, STD, proto)
    ranking = ranking[ranking.user_id.astype(str).isin(eligible_users)].reset_index(drop=True)
    selected = ranking.user_id.astype(str).tolist()
    if args.users is not None:
        selected = selected[:args.users]
    replay_mask = (bm.split == "train").to_numpy()
    replay_x = torch.from_numpy(((base["X"][replay_mask] - MEAN) / STD).astype(np.float32))
    with torch.no_grad(): replay_reference = base_model(replay_x).detach()
    if len(replay_x) > args.replay_size:
        replay_ix = torch.randperm(len(replay_x))[:args.replay_size]
        replay_x = replay_x[replay_ix]
        replay_reference = replay_reference[replay_ix]
    c = covariance(capture_keys(base_model, train_x))
    base_eval = {}
    for period, split, stage in [("early_test", "test", None), ("stage_2", None, "stage_2"), ("stage_3", None, "stage_3")]:
        arr, meta = (base["X"], bm) if split else (later["X"], lm)
        x, mask = tensors(arr, meta, MEAN, STD, split=split, stage=stage)
        base_eval[period] = {"z": embeddings(base_model, x).numpy(), "users": meta.loc[mask, "user_id"].astype(str).to_numpy()}
    all_rows = []
    edit_summary = []
    for user in selected:
        adapt_x, _ = tensors(later["X"], lm, MEAN, STD, user=user, stage="stage_1")
        if len(adapt_x) > args.adaptation_size:
            adapt_x = adapt_x[torch.randperm(len(adapt_x))[:args.adaptation_size]]
        target_proto = torch.from_numpy(proto[user]).float()
        key, value = optimize_value(base_model, adapt_x, target_proto, replay_x, replay_reference, args.v_steps)
        edited, delta = apply_rome(base_model, key, value, c)
        metrics = edit_metrics(edited, base["X"], bm, later["X"], lm, [user], proto, base_eval)
        method = "ROME_layer2_linear2_rank1"
        metrics["user_id"] = user; metrics["edit_scope"] = "individual"; metrics["method"] = method
        metrics["delta_rank"] = int(torch.linalg.matrix_rank(delta).item())
        metrics["changed_parameters"] = int(torch.count_nonzero(delta).item())
        metrics["weight_displacement_l2"] = float(torch.linalg.vector_norm(delta).item())
        all_rows.append(metrics)
        torch.save(edited.state_dict(), OUT / f"{method}_user_{user}.pt")
        edit_summary.append({"method": method, "scope": "individual", "user_id": user, "rank": 1, "v_optimization_steps": args.v_steps, "key_dimension": int(key.numel()), "value_dimension": int(value.numel()), "weight_displacement_l2": float(torch.linalg.vector_norm(delta).item())})
    result = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not result.empty:
        result = add_success_flags(result)
    result.to_csv(OUT / "model_editing_metrics.csv", index=False)
    save_plots(result)
    summary, ci = population_outputs(result, ranking, eligibility)
    save_population_plots(result, ranking, summary, ci)
    pd.DataFrame(edit_summary).to_csv(OUT / "edit_summary.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({"method": "ROME-style rank-one model editing", "edited_matrix": "Transformer layer 2 linear2 weight", "adaptation_stage": "stage_1", "primary_holdout": "stage_2", "secondary_holdout": "stage_3", "threshold": THRESHOLD, "selected_users": selected, "seed": SEED, "v_optimization_steps": args.v_steps, "replay_size": args.replay_size, "adaptation_size": args.adaptation_size, "rank": 1, "covariance": "training-key uncentered covariance with 1e-4 ridge", "formula": "W_new = W + (v - (W k + b)) outer (C^-1 k) / ((C^-1 k)^T k)"}, indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
