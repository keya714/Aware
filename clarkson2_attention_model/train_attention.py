from pathlib import Path
import argparse
import json
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

class AttentionEncoder(nn.Module):
    def __init__(self, features=4, d_model=64, embedding_dim=32, heads=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 50, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=128, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.embedding = nn.Sequential(nn.Linear(d_model, embedding_dim), nn.LayerNorm(embedding_dim))
        self.classifier = None

    def forward(self, x):
        h = self.encoder(self.proj(x) + self.pos)
        h = self.norm(h).mean(dim=1)
        z = nn.functional.normalize(self.embedding(h), dim=1)
        return z

def verification_metrics(emb, meta, split):
    rows = meta[meta.split == split].reset_index()
    e = emb[rows["index"].to_numpy()]
    by_user = {}
    for i, u in enumerate(rows.user_id.astype(str)):
        by_user.setdefault(u, []).append(i)
    genuine, impostor = [], []
    # Enrollment is the first training window for each user; test/adapt windows are queries.
    for u, idx in by_user.items():
        enroll = e[idx[0]]
        for j in idx[1:]:
            genuine.append(float(np.linalg.norm(enroll - e[j])))
    users = list(by_user)
    for u in users:
        enroll = e[by_user[u][0]]
        for v in users:
            if u != v:
                for j in by_user[v][:max(1, min(5, len(by_user[v])) )]:
                    impostor.append(float(np.linalg.norm(enroll - e[j])))
    scores = np.array(genuine + impostor)
    labels = np.array([1] * len(genuine) + [0] * len(impostor))
    thresholds = np.unique(scores)
    best = None
    for t in thresholds[::max(1, len(thresholds)//2000)]:
        pred = scores <= t
        far = float(((pred == 1) & (labels == 0)).sum() / max(1, (labels == 0).sum()))
        frr = float(((pred == 0) & (labels == 1)).sum() / max(1, (labels == 1).sum()))
        item = (abs(far - frr), float(t), far, frr)
        if best is None or item[0] < best[0]: best = item
    return {"split": split, "genuine_trials": len(genuine), "impostor_trials": len(impostor), "eer_approx": (best[2] + best[3]) / 2, "threshold": best[1], "far": best[2], "frr": best[3]}

def main():
    global RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10, help="final epoch number")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=228)
    ap.add_argument("--resume", type=Path, default=RESULTS / "attention_model.pt")
    ap.add_argument("--from-scratch", action="store_true", help="ignore any existing checkpoint")
    ap.add_argument("--resume-epoch", type=int, default=3, help="epoch represented by legacy checkpoints without epoch metadata")
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    ap.add_argument("--patience", type=int, default=15, help="epochs without validation EER improvement before stopping")
    ap.add_argument("--min-delta", type=float, default=1e-4, help="minimum validation EER improvement")
    args = ap.parse_args()
    RESULTS = args.results_dir.resolve()
    RESULTS.mkdir(exist_ok=True, parents=True)
    if args.from_scratch:
        args.resume = None
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    data = np.load(RESULTS / "sequences.npz")
    meta = pd.read_csv(RESULTS / "sequence_metadata.csv", dtype={"user_id": str})
    X = data["X"].astype(np.float32)
    train = meta.split.to_numpy() == "train"
    mu, sd = X[train].reshape(-1, 4).mean(0), X[train].reshape(-1, 4).std(0)
    sd[sd < 1e-6] = 1.0
    X = (X - mu) / sd
    users = sorted(meta.loc[train, "user_id"].astype(str).unique())
    user_to_id = {u:i for i,u in enumerate(users)}
    y = np.array([user_to_id.get(str(u), -1) for u in meta.user_id])
    train_idx = np.where(train & (y >= 0))[0]
    model = AttentionEncoder()
    model.classifier = nn.Linear(32, len(users))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    ds = TensorDataset(torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx]).long())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    history = []
    start_epoch = 1
    if args.resume and args.resume.exists():
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            opt.load_state_dict(checkpoint["optimizer"])
        history = checkpoint.get("history", [])
        if "epoch" in checkpoint:
            start_epoch = int(checkpoint["epoch"]) + 1
        else:
            # The original 3-epoch checkpoint predates per-epoch metadata.
            start_epoch = args.resume_epoch + 1
            metrics_path = RESULTS / "attention_metrics.json"
            if metrics_path.exists():
                history = json.loads(metrics_path.read_text(encoding="utf-8")).get("history", [])
        print(f"Resuming from epoch {start_epoch - 1}; training through epoch {args.epochs}")
    best_eer = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = RESULTS / "attention_model_best.pt"
    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        best_validation = best_checkpoint.get("validation", {})
        if best_validation.get("eer_approx") is not None:
            best_eer = float(best_validation["eer_approx"])
            best_epoch = int(best_checkpoint.get("epoch", 0))
            epochs_without_improvement = max(0, start_epoch - best_epoch - 1)
    else:
        validation_history = [h for h in history if h.get("validation_eer") is not None]
        if validation_history:
            best_item = min(validation_history, key=lambda h: h["validation_eer"])
            best_eer = float(best_item["validation_eer"])
            best_epoch = int(best_item["epoch"])
            epochs_without_improvement = max(0, start_epoch - best_epoch - 1)
    if best_epoch:
        print(f"Current best validation EER={best_eer:.6f} at epoch {best_epoch}; patience used={epochs_without_improvement}/{args.patience}")
    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        total = correct = n = 0
        for xb, yb in loader:
            opt.zero_grad(); z = model(xb); logits = model.classifier(z); loss = loss_fn(logits, yb); loss.backward(); opt.step()
            total += loss.item() * len(yb); correct += (logits.argmax(1) == yb).sum().item(); n += len(yb)
        model.eval()
        with torch.no_grad():
            validation_embeddings = model(torch.from_numpy(X)).numpy()
        validation = verification_metrics(validation_embeddings, meta, "adapt")
        val_eer = validation["eer_approx"]
        item = {"epoch": epoch, "loss": total/n, "train_accuracy": correct/n, "validation_eer": val_eer}
        history.append(item)
        print(item)
        checkpoint = {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "users": users, "mean": mu, "std": sd, "history": history, "validation": validation}
        torch.save(checkpoint, RESULTS / f"attention_model_epoch_{epoch:02d}.pt")
        if val_eer < best_eer - args.min_delta:
            best_eer = val_eer
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, RESULTS / "attention_model_best.pt")
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best validation EER={best_eer:.6f} at epoch {best_epoch}")
            break
        model.train()
    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best_checkpoint["model"])
    model.eval()
    with torch.no_grad(): emb = model(torch.from_numpy(X)).numpy()
    metrics = [verification_metrics(emb, meta, s) for s in ["adapt", "test"]]
    out = {"users": len(users), "windows": len(meta), "feature_names": ["hold", "press_to_press", "release_to_press", "inter_event"], "normalization": {"mean": mu.tolist(), "std": sd.tolist()}, "max_epochs": args.epochs, "patience": args.patience, "best_epoch": best_epoch, "best_validation_eer": best_eer, "history": history, "verification": metrics}
    RESULTS.mkdir(exist_ok=True)
    torch.save({"epoch": best_epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "users": users, "mean": mu, "std": sd, "history": history, "best_validation_eer": best_eer}, RESULTS / "attention_model.pt")
    (RESULTS / "attention_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

if __name__ == "__main__": main()
