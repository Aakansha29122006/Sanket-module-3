"""
Compute classification and regression metrics for the saved road blockage checkpoint.

Loads saved_models/best_road_blockage_model.pt, runs it on the held-out test split
(same synthetic-data generator and 75/25 split as run_benchmark.py), and reports:
  - Accuracy (Hamming), exact match ratio
  - Precision, recall, F1 (macro and micro), per-class precision/recall/F1
  - ROC-AUC (per class, macro, micro) and average precision (PR-AUC, macro)
  - Blockage MAE, RMSE, R², tolerance accuracy (+/-10 points)
  - Severity accuracy and severity macro F1

Usage:
  python compute_metrics.py --project <path to cvip_anti> --weights <path to .pt> --out metrics.json
"""

import argparse
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    r2_score,
)


def build_model(ckpt, state_dict):
    """Pick the architecture from checkpoint metadata, falling back to trying each class."""
    from road_blockage.models.unet_cnn import UNetRoadBlockageCNN
    from road_blockage.models.efficientnet_cnn import EfficientNetRoadBlockageCNN

    name = str(ckpt.get("model_class", "")) + " " + str(ckpt.get("model_name", ""))
    candidates = []
    if "UNet" in name or "unet" in name.lower():
        candidates.append(("UNet", lambda: UNetRoadBlockageCNN(base_features=16)))
    if "Efficient" in name or "efficient" in name.lower():
        candidates.append(("EfficientNet", lambda: EfficientNetRoadBlockageCNN()))
    candidates += [
        ("UNet", lambda: UNetRoadBlockageCNN(base_features=16)),
        ("EfficientNet", lambda: EfficientNetRoadBlockageCNN()),
    ]
    for label, make in candidates:
        try:
            model = make()
            model.load_state_dict(state_dict, strict=True)
            return label, model
        except Exception:
            continue
    raise RuntimeError("Checkpoint does not match UNet or EfficientNet architectures")


def pct_to_sev_index(pct):
    # Same bins as road_blockage/evaluation/metrics.py
    return np.digitize(pct, [0.5, 25.0, 60.0, 85.0])


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Folder containing the road_blockage package")
    parser.add_argument("--weights", required=True, help="Path to best_road_blockage_model.pt")
    parser.add_argument("--samples", type=int, default=100, help="Dataset size used by run_benchmark.py")
    parser.add_argument("--seed", type=int, default=42, help="Dataset seed used by run_benchmark.py")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.40, help="Sigmoid threshold for obstacle labels")
    parser.add_argument("--out", default="metrics.json")
    args = parser.parse_args()

    sys.path.insert(0, args.project)
    from road_blockage.data.dataset import RoadBlockageDataset, OBSTACLE_CLASSES

    # 1. Load checkpoint and identify the architecture
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    arch_label, model = build_model(ckpt, state_dict)
    model.eval()
    print(f"[*] Checkpoint architecture: {arch_label} "
          f"(metadata model_name={ckpt.get('model_name')}, model_class={ckpt.get('model_class')})")

    # 2. Rebuild the same held-out test split used by the benchmark
    dataset = RoadBlockageDataset(num_samples=args.samples, img_size=args.img_size, seed=args.seed)
    train_size = int(0.75 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_set = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(test_set, batch_size=16, shuffle=False)
    print(f"[*] Test split: {test_size} images at {args.img_size}x{args.img_size}")

    # 3. Run inference
    true_pct, pred_pct = [], []
    true_obs, prob_obs = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"])
            true_pct.extend(batch["blockage_pct"].numpy().reshape(-1).tolist())
            pred_pct.extend(out["blockage_pct"].numpy().reshape(-1).tolist())
            true_obs.extend(batch["obstacle_labels"].numpy().tolist())
            prob_obs.extend(torch.sigmoid(out["obstacle_logits"]).numpy().tolist())

    true_pct = np.array(true_pct)
    pred_pct = np.clip(np.array(pred_pct), 0.0, 100.0)
    y_true = np.array(true_obs).astype(int)
    y_prob = np.array(prob_obs)
    y_pred = (y_prob > args.threshold).astype(int)

    # 4. Multi-label classification metrics
    m = {}
    m["accuracy_hamming_pct"] = float(np.mean(y_true == y_pred) * 100)
    m["exact_match_pct"] = float(np.mean(np.all(y_true == y_pred, axis=1)) * 100)

    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_mic, r_mic, f_mic, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    m["precision_macro_pct"] = float(p_mac * 100)
    m["recall_macro_pct"] = float(r_mac * 100)
    m["f1_macro_pct"] = float(f_mac * 100)
    m["precision_micro_pct"] = float(p_mic * 100)
    m["recall_micro_pct"] = float(r_mic * 100)
    m["f1_micro_pct"] = float(f_mic * 100)

    per_p, per_r, per_f, per_s = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    per_class = {}
    aucs = []
    for i, cls in enumerate(OBSTACLE_CLASSES):
        auc = safe_auc(y_true[:, i], y_prob[:, i])
        aucs.append(auc)
        per_class[cls] = {
            "precision_pct": round(float(per_p[i] * 100), 2),
            "recall_pct": round(float(per_r[i] * 100), 2),
            "f1_pct": round(float(per_f[i] * 100), 2),
            "support_positive": int(per_s[i]),
            "roc_auc": None if np.isnan(auc) else round(auc, 4),
        }
    m["roc_auc_macro"] = float(np.nanmean(aucs)) if not np.all(np.isnan(aucs)) else float("nan")
    m["roc_auc_micro"] = safe_auc(y_true.ravel(), y_prob.ravel())
    ap = [average_precision_score(y_true[:, i], y_prob[:, i]) for i in range(y_true.shape[1])
          if y_true[:, i].sum() > 0]
    m["average_precision_macro"] = float(np.mean(ap)) if ap else float("nan")

    # 5. Blockage regression metrics
    err = np.abs(true_pct - pred_pct)
    m["blockage_mae"] = float(np.mean(err))
    m["blockage_rmse"] = float(np.sqrt(np.mean((true_pct - pred_pct) ** 2)))
    m["blockage_r2"] = float(r2_score(true_pct, pred_pct))
    m["tolerance_accuracy_10pts_pct"] = float(np.mean(err <= 10.0) * 100)

    # 6. Severity metrics (5 buckets)
    t_sev = pct_to_sev_index(true_pct)
    p_sev = pct_to_sev_index(pred_pct)
    m["severity_accuracy_pct"] = float(np.mean(t_sev == p_sev) * 100)
    _, _, f_sev, _ = precision_recall_fscore_support(t_sev, p_sev, average="macro", zero_division=0)
    m["severity_f1_macro_pct"] = float(f_sev * 100)

    # 7. Report
    result = {
        "architecture": arch_label,
        "test_images": int(test_size),
        "obstacle_threshold": args.threshold,
        "metrics": {k: (None if np.isnan(v) else round(v, 4)) for k, v in m.items()},
        "per_class_obstacles": per_class,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== Metrics ===")
    for k, v in result["metrics"].items():
        print(f"{k:32s} {v}")
    print("\n=== Per-class obstacles ===")
    for cls, d in per_class.items():
        print(f"{cls:14s} P={d['precision_pct']:6.2f}  R={d['recall_pct']:6.2f}  "
              f"F1={d['f1_pct']:6.2f}  AUC={d['roc_auc']}  positives={d['support_positive']}")
    print(f"\n[✓] Saved to {args.out}")


if __name__ == "__main__":
    main()
