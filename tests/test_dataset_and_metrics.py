"""
Dataset Synthesis and Evaluation Metrics Tests
Validates scene generator, area calculations, metrics evaluation, and severity categorization.
"""

import unittest
import numpy as np

from road_blockage.data.dataset import (
    generate_synthetic_road_scene,
    RoadBlockageDataset,
    pct_to_severity,
    OBSTACLE_CLASSES
)
from road_blockage.evaluation.metrics import evaluate_predictions, compute_miou


class TestDatasetAndMetrics(unittest.TestCase):
    """Test data synthesis, geometric blockage calculation, and evaluation metrics."""

    def test_synthetic_scene_zero_blockage(self):
        scene = generate_synthetic_road_scene(width=128, height=128, target_blockage=0.0, seed=123)
        self.assertEqual(scene["blockage_pct"], 0.0)
        self.assertEqual(len(scene["obstacles"]), 0)
        self.assertEqual(scene["severity"], 0)
        self.assertEqual(scene["blocked_pixels"], 0)

    def test_synthetic_scene_high_blockage(self):
        scene = generate_synthetic_road_scene(
            width=128, height=128, target_blockage=95.0,
            obstacles=["fallen_tree", "fallen_pole"], seed=456
        )
        self.assertGreater(scene["blockage_pct"], 70.0)
        self.assertIn("fallen_tree", scene["obstacles"])
        self.assertIn("fallen_pole", scene["obstacles"])

    def test_severity_mapping(self):
        self.assertEqual(pct_to_severity(0.0), 0)    # Clear
        self.assertEqual(pct_to_severity(15.0), 1)   # Minor
        self.assertEqual(pct_to_severity(45.0), 2)   # Moderate
        self.assertEqual(pct_to_severity(75.0), 3)   # Severe
        self.assertEqual(pct_to_severity(95.0), 4)   # Critical

    def test_metrics_evaluation(self):
        y_true = np.array([0.0, 10.0, 34.0, 70.0, 95.0, 100.0])
        y_pred = np.array([0.0, 12.0, 32.0, 68.0, 93.0, 98.0])

        # Ensure all 6 classes are represented
        obs_true = np.array([
            [1, 0, 0, 0, 0, 0],  # class 0: fallen_tree
            [0, 1, 0, 0, 0, 0],  # class 1: fallen_pole
            [0, 0, 1, 0, 0, 0],  # class 2: loose_soil
            [0, 0, 0, 1, 0, 0],  # class 3: fallen_rocks
            [0, 0, 0, 0, 1, 0],  # class 4: accident
            [0, 0, 0, 0, 0, 1],  # class 5: water_flood
        ])
        obs_pred = obs_true.copy()  # perfect predictions for baseline validation

        metrics = evaluate_predictions(y_true, y_pred, obs_true, obs_pred, tolerance=10.0)

        self.assertEqual(metrics["obstacle_accuracy"], 100.0)
        self.assertEqual(metrics["f1_score"], 100.0)
        self.assertEqual(metrics["tolerance_accuracy_10pct"], 100.0)
        self.assertLess(metrics["blockage_mae"], 3.0)
        self.assertGreater(metrics["blockage_r2"], 0.95)

    def test_miou_computation(self):
        m1 = np.zeros((64, 64), dtype=int)
        m2 = np.zeros((64, 64), dtype=int)
        m1[10:30, 10:30] = 1
        m2[10:30, 10:30] = 1
        miou = compute_miou(m1, m2, num_classes=2)
        self.assertEqual(miou, 100.0)


if __name__ == "__main__":
    unittest.main()
