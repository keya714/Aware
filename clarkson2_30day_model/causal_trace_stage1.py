"""Leakage-free causal tracing using Stage 1 as the tracing window.

The original causal-trace script used Stage 2.  This script uses Stage 1 so
Stage 2 and Stage 3 remain held out for model-editing evaluation.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
OUT = RESULTS / "causal_trace_stage1"
OUT.mkdir(parents=True, exist_ok=True)
FIGURES = OUT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
import sys
sys.path.insert(0, str(ROOT / "clarkson2_attention_model"))
from train_attention import AttentionEncoder

THRESHOLD = 1.1885696649551392


def load_model():
    ck = torch.load(RESULTS / "attention_model_best.pt", map_location="cpu", weights_only=False)
    model = AttentionEncoder()
    model.classifier = nn.Linear(32, len(ck["users"]))
    model.load_state_dict(ck["model"])
    model.eval()
    mean = np.asarray(ck["mean"], dtype=np.float32)
    std = np.asarray(ck["std"], dtype=np.float32)
    return model, mean, std


def activation(model, x, layer):
    saved = []
    hook = layer.register_forward_hook(lambda _m, _i, out: saved.append(out.detach()))
    with torch.no_grad():
        model(x)
    hook.remove()
    return torch.cat(saved, dim=0).mean(dim=0)


def patched_embedding(model, x, layer, replacement):
    def hook(_m, _i, out):
        if out.ndim == 3:
            return replacement.unsqueeze(0).expand(out.shape[0], -1, -1).to(out)
        return replacement.unsqueeze(0).expand(out.shape[0], -1).to(out)
    handle = layer.register_forward_hook(hook)
    with torch.no_grad():
        z = model(x).numpy()
    handle.remove()
    return z


def choose_users(base_meta, later_meta, ranking):
    counts = later_meta.groupby(["user_id", "stage"]).size().unstack(fill_value=0)
    eligible = {str(u) for u in counts.index if all(counts.loc[u].get(s, 0) >= 5 for s in ["stage_1", "stage_2", "stage_3"])}
    ordered = [str(u) for u in ranking.user_id if str(u) in eligible]
    return list(dict.fromkeys(ordered[:5] + ordered[-5:]))


def build_stage1_ranking(model, base, base_meta, later, later_meta, mean, std):
    train_mask = (base_meta.split == "train").to_numpy()
    test_mask = (base_meta.split == "test").to_numpy()
    stage_mask = (later_meta.stage == "stage_1").to_numpy()
    with torch.no_grad():
        z_base = model(torch.from_numpy(((base["X"] - mean) / std).astype(np.float32))).numpy()
        z_later = model(torch.from_numpy(((later["X"] - mean) / std).astype(np.float32))).numpy()
    users = base_meta.user_id.astype(str).to_numpy()
    later_users = later_meta.user_id.astype(str).to_numpy()
    rows = []
    for user in sorted(base_meta.loc[train_mask, "user_id"].astype(str).unique()):
        prototype = z_base[train_mask & (users == user)].mean(0)
        baseline = np.linalg.norm(z_base[test_mask & (users == user)] - prototype, axis=1)
        stage1 = np.linalg.norm(z_later[stage_mask & (later_users == user)] - prototype, axis=1)
        if len(baseline) and len(stage1):
            rows.append({"user_id": user, "baseline_mean_distance": float(baseline.mean()), "stage_1_mean_distance": float(stage1.mean()), "stage_1_absolute_drift": float(stage1.mean() - baseline.mean()), "stage_1_windows": int(len(stage1))})
    return pd.DataFrame(rows).sort_values("stage_1_absolute_drift", ascending=False).reset_index(drop=True)


def save_heatmaps(frame):
    """Save layer-by-user heatmaps from the causal-tracing CSV values."""
    if frame.empty:
        return
    order = [
        "input_projection", "transformer_layer_1", "transformer_layer_2",
        "post_encoder_norm", "embedding_projection",
    ]
    users = list(frame.user_id.astype(str).drop_duplicates())
    labels = {
        "input_projection": "Input projection",
        "transformer_layer_1": "Transformer layer 1",
        "transformer_layer_2": "Transformer layer 2",
        "post_encoder_norm": "Post-encoder norm",
        "embedding_projection": "Embedding projection",
    }
    def draw(column, filename, title, fmt):
        table = frame.pivot_table(index="intervention", columns="user_id", values=column, aggfunc="first")
        table = table.reindex(index=[x for x in order if x in table.index], columns=users)
        table.index = [labels.get(x, x) for x in table.index]
        plt.figure(figsize=(max(8, 1.4 * len(users)), 5.8))
        sns.heatmap(table, annot=True, fmt=fmt, cmap="RdYlGn", center=0 if column == "recovery_fraction" else None, linewidths=.5, cbar_kws={"label": column.replace("_", " ").title()})
        plt.title(title)
        plt.xlabel("User")
        plt.ylabel("Intervention layer/component")
        plt.tight_layout()
        plt.savefig(FIGURES / filename, dpi=200)
        plt.close()
    draw("recovery_fraction", "causal_trace_recovery_heatmap.png", "Causal-tracing recovery by layer and user", ".1%")
    draw("patched_mean_distance", "causal_trace_distance_heatmap.png", "Patched embedding distance by layer and user", ".3f")
    draw("wrong_user_control_distance", "causal_trace_control_heatmap.png", "Wrong-user control distance by layer and user", ".3f")


def main():
    model, mean, std = load_model()
    base = np.load(RESULTS / "sequences.npz")
    bm = pd.read_csv(RESULTS / "sequence_metadata.csv", dtype={"user_id": str})
    later = np.load(RESULTS / "remaining_stage_sequences.npz")
    lm = pd.read_csv(RESULTS / "remaining_stage_metadata.csv", dtype={"user_id": str})
    ranking = build_stage1_ranking(model, base, bm, later, lm, mean, std)
    ranking.to_csv(OUT / "stage1_drift_ranking.csv", index=False)
    users = choose_users(bm, lm, ranking)
    layers = {
        "input_projection": model.proj,
        "transformer_layer_1": model.encoder.layers[0],
        "transformer_layer_2": model.encoder.layers[1],
        "post_encoder_norm": model.norm,
        "embedding_projection": model.embedding,
    }
    rows = []
    for user in users:
        tr = np.where((bm.split == "train").to_numpy() & (bm.user_id.astype(str) == user).to_numpy())[0]
        st1 = np.where((lm.stage == "stage_1").to_numpy() & (lm.user_id.astype(str) == user).to_numpy())[0]
        if not len(tr) or not len(st1):
            continue
        xtr = torch.from_numpy(((base["X"][tr] - mean) / std).astype(np.float32))
        xst1 = torch.from_numpy(((later["X"][st1] - mean) / std).astype(np.float32))
        with torch.no_grad():
            prototype = model(xtr).mean(0).numpy()
            unpatched = model(xst1).numpy()
        baseline = np.linalg.norm(unpatched - prototype, axis=1)
        control_user = next((u for u in bm.user_id.astype(str).unique() if u != user), None)
        control = np.where((bm.split == "train").to_numpy() & (bm.user_id.astype(str) == control_user).to_numpy())[0][:len(tr)]
        xcontrol = torch.from_numpy(((base["X"][control] - mean) / std).astype(np.float32))
        for name, layer in layers.items():
            reference = activation(model, xtr, layer)
            patched = patched_embedding(model, xst1, layer, reference)
            wrong_ref = activation(model, xcontrol, layer)
            wrong = patched_embedding(model, xst1, layer, wrong_ref)
            d = np.linalg.norm(patched - prototype, axis=1)
            wd = np.linalg.norm(wrong - prototype, axis=1)
            rows.append({
                "user_id": user, "stage": "stage_1", "intervention": name,
                "n_windows": len(d), "baseline_mean_distance": float(baseline.mean()),
                "patched_mean_distance": float(d.mean()),
                "distance_recovery": float(baseline.mean() - d.mean()),
                "recovery_fraction": float((baseline.mean() - d.mean()) / max(baseline.mean(), 1e-8)),
                "patched_frr": float(np.mean(d > THRESHOLD)),
                "wrong_user_control_distance": float(wd.mean()),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "causal_trace_stage1.csv", index=False)
    save_heatmaps(out)
    summary = {
        "users": sorted(out.user_id.unique().tolist()) if len(out) else [],
        "cohort_definition": "top five and bottom five eligible users ranked by Stage 1 drift only",
        "stage_used_for_tracing": "stage_1",
        "held_out_stages": ["stage_2", "stage_3"],
        "threshold": THRESHOLD,
        "best_by_user": out.loc[out.groupby("user_id").recovery_fraction.idxmax()].to_dict("records") if len(out) else [],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
