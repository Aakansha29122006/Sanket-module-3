"""
High-Level RoadBlockageDetector API Tests
Validates inference, image type flexibility, batch prediction, and output serialization.
"""

import os
import unittest
import numpy as np
import torch
from PIL import Image

from road_blockage.detector import RoadBlockageDetector, RoadBlockageResult
from road_blockage.models.efficientnet_cnn import EfficientNetRoadBlockageCNN


class TestDetectorAPI(unittest.TestCase):
    """Test RoadBlockageDetector class methods."""

    def setUp(self):
        self.model = EfficientNetRoadBlockageCNN()
        self.detector = RoadBlockageDetector(
            model=self.model,
            model_name="EfficientNet_B0_Test",
            device="cpu"
        )
        self.test_img_path = "tests/temp_test_image.jpg"
        img = Image.new("RGB", (300, 300), color=(120, 140, 100))
        img.save(self.test_img_path)

    def tearDown(self):
        if os.path.exists(self.test_img_path):
            os.remove(self.test_img_path)

    def test_predict_from_filepath(self):
        res = self.detector.predict(self.test_img_path)
        self.assertIsInstance(res, RoadBlockageResult)
        self.assertIsInstance(res.blockage_percentage, float)
        self.assertIn(res.severity, ["Clear", "Minor", "Moderate", "Severe", "Critical"])
        self.assertIsInstance(res.summary, str)
        self.assertGreater(res.inference_time_ms, 0.0)

    def test_predict_from_pil(self):
        pil_img = Image.new("RGB", (128, 128), (80, 80, 80))
        res = self.detector.predict(pil_img)
        self.assertIsInstance(res, RoadBlockageResult)

    def test_predict_from_numpy_rgb(self):
        arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        res = self.detector.predict(arr)
        self.assertIsInstance(res, RoadBlockageResult)

    def test_predict_from_numpy_grayscale(self):
        arr = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        res = self.detector.predict(arr)
        self.assertIsInstance(res, RoadBlockageResult)

    def test_predict_from_numpy_rgba(self):
        arr = np.random.randint(0, 255, (150, 150, 4), dtype=np.uint8)
        res = self.detector.predict(arr)
        self.assertIsInstance(res, RoadBlockageResult)

    def test_nonexistent_file_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            self.detector.predict("nonexistent_path_to_image_xyz.jpg")

    def test_batch_prediction(self):
        img1 = Image.new("RGB", (100, 100), (50, 50, 50))
        img2 = Image.new("RGB", (100, 100), (150, 150, 150))
        results = self.detector.predict_batch([img1, img2])
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, RoadBlockageResult)

    def test_serialization(self):
        res = self.detector.predict(self.test_img_path)
        d = res.to_dict()
        self.assertIn("blockage_percentage", d)
        self.assertIn("obstacles", d)
        self.assertIn("summary", d)

        json_str = res.to_json()
        self.assertIsInstance(json_str, str)
        self.assertIn('"blockage_percentage":', json_str)


if __name__ == "__main__":
    unittest.main()
