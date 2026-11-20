"""
Evaluation Metrics for Road Blockage and Obstacle Identification.
Computes comprehensive accuracy, precision, recall, F1, MAE, R², tolerance accuracy, and latency.
"""

import time
import math
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, r2_score


def evaluate_predictions(
    true_blockage: np.ndarray,
    pred_blockage: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    tolerance: float = 10.0
) -> Dict[str, float]:
    """
    Comprehensive evaluation matrix for multi-task road blockage models.
    
    Args:
        true_blockage: Ground truth blockage percentages (N,)
        pred_blockage: Predicted blockage percentages (N,)
        true_labels: Ground truth binary obstacle matrix (N, num_classes)
        pred_labels: Predicted binary obstacle matrix (N, num_classes)
        tolerance: Acceptable error threshold in percentage points for tolerance accuracy
    """
    # 1. Regression Metrics
    mae = float(np.mean(np.abs(true_blockage - pred_blockage)))
    mse = float(np.mean((true_blockage - pred_blockage) ** 2))
    rmse = float(math.sqrt(mse))
    
    # R² Score (Coefficient of determination)
    r2 = float(r2_score(true_blockage, pred_blockage))
    
    # Tolerance Accuracy: Percentage of predictions within ±tolerance% of ground truth
    abs_err = np.abs(true_blockage - pred_blockage)
    tolerance_acc = float(np.mean(abs_err <= tolerance) * 100.0)

    # 2. Multi-Label Classification Metrics
    # Hamming Accuracy: fraction of correct label predictions across all instances and classes
    hamming_acc = float(np.mean(true_labels == pred_labels) * 100.0)
    
    # Exact Match Ratio (Subset Accuracy): all classes must match exactly
    exact_match = float(np.mean(np.all(true_labels == pred_labels, axis=1)) * 100.0)

    # Precision, Recall, F1 (macro averaged across classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0
    )

    # Severity accuracy
    def to_sev(p):
        return np.digitize(p, [0.5, 25.0, 60.0, 85.0])
    
    true_sev = to_sev(true_blockage)
    pred_sev = to_sev(pred_blockage)
    severity_acc = float(np.mean(true_sev == pred_sev) * 100.0)

    return {
        "obstacle_accuracy": round(hamming_acc, 2),
        "exact_match_ratio": round(exact_match, 2),
        "precision": round(float(precision) * 100.0, 2),
        "recall": round(float(recall) * 100.0, 2),
        "f1_score": round(float(f1) * 100.0, 2),
        "blockage_mae": round(mae, 2),
        "blockage_rmse": round(rmse, 2),
        "blockage_r2": round(max(0.0, r2), 4),
        "tolerance_accuracy_10pct": round(tolerance_acc, 2),
        "severity_accuracy": round(severity_acc, 2)
    }


def compute_miou(true_masks: np.ndarray, pred_masks: np.ndarray, num_classes: int = 3) -> float:
    """Compute Mean Intersection over Union for semantic segmentation masks."""
    ious = []
    for cls_id in range(num_classes):
        intersection = np.sum((true_masks == cls_id) & (pred_masks == cls_id))
        union = np.sum((true_masks == cls_id) | (pred_masks == cls_id))
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(float(intersection / union))
    return round(float(np.mean(ious)) * 100.0, 2)


def measure_inference_speed(
    model: torch.nn.Module,
    input_size: Tuple[int, int, int] = (3, 224, 224),
    num_runs: int = 50,
    device: str = "cpu"
) -> Dict[str, float]:
    """Measure inference latency (ms) and throughput (FPS)."""
    model.eval()
    dummy = torch.randn(1, *input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy)
    total_time = time.perf_counter() - start_time

    avg_ms = (total_time / num_runs) * 1000.0
    fps = num_runs / total_time

    # Parameter count
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    return {
        "latency_ms": round(avg_ms, 2),
        "fps": round(fps, 1),
        "parameters_m": round(params_m, 2)
    }
