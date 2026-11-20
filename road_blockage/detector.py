"""
High-Level Production Inference Engine for Road Blockage and Obstacle Identification.
Designed for seamless integration into larger autonomous systems, dispatch engines, and disaster response pipelines.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Union, Optional, Any
import numpy as np
import torch
from PIL import Image

from .data.dataset import OBSTACLE_CLASSES, SEVERITY_LEVELS, pct_to_severity
from .models.efficientnet_cnn import EfficientNetRoadBlockageCNN
from .models.unet_cnn import UNetRoadBlockageCNN
from .models.yolo_cnn import YOLORoadBlockageCNN, build_yolov5_cnn, build_yolov8_cnn
from .models.resnet_cnn import ResNetRoadBlockageCNN


# Human-friendly natural language obstacle labels
OBSTACLE_DISPLAY_NAMES = {
    "fallen_tree": "fallen trees",
    "fallen_pole": "poles",
    "loose_soil": "loose soil",
    "fallen_rocks": "fallen rocks",
    "accident": "accident",
    "water_flood": "flood water"
}


@dataclass
class RoadBlockageResult:
    """Standardized output structure for downstream integration."""
    blockage_percentage: float
    is_blocked: bool
    obstacles: List[str]
    obstacle_display_names: List[str]
    severity: str
    confidence_scores: Dict[str, float]
    summary: str
    model_name: str
    inference_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert result to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def __str__(self) -> str:
        return self.summary


def generate_natural_language_summary(
    blockage_pct: float,
    obstacles: List[str]
) -> str:
    """
    Generate exact natural language phrase matching user requirements:
    - '0% road is clear'
    - '95% road is blocked due to fallen trees and poles'
    - '10% road is blocked due to loose soil'
    - '34% road is blocked due to fallen rocks'
    - '100% road is blocked due to accident'
    """
    pct_rounded = int(round(blockage_pct))

    if pct_rounded <= 0 or len(obstacles) == 0:
        return "0% road is clear"

    # Map to display names
    display_obs = [OBSTACLE_DISPLAY_NAMES.get(o, o) for o in obstacles]
    
    if len(display_obs) == 1:
        cause = display_obs[0]
    elif len(display_obs) == 2:
        cause = f"{display_obs[0]} and {display_obs[1]}"
    else:
        cause = ", ".join(display_obs[:-1]) + f", and {display_obs[-1]}"

    return f"{pct_rounded}% road is blocked due to {cause}"


class RoadBlockageDetector:
    """
    Production-grade Road Blockage Detector (Pure CNN).
    Strictly Transformer-Free.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str = "EfficientNet_B0",
        classes: Optional[List[str]] = None,
        confidence_threshold: float = 0.40,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.model_name = model_name
        self.classes = classes or OBSTACLE_CLASSES
        self.threshold = confidence_threshold

        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @classmethod
    def load(
        cls,
        weights_path: str,
        device: Optional[str] = None,
        confidence_threshold: float = 0.40
    ) -> "RoadBlockageDetector":
        """Load trained model weights from disk."""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model checkpoint not found: {weights_path}")

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(weights_path, map_location=dev)

        model_name = checkpoint.get("model_name", "EfficientNet_B0")
        classes = checkpoint.get("classes", OBSTACLE_CLASSES)
        state_dict = checkpoint["state_dict"]

        # Instantiate appropriate pure CNN model
        if "EfficientNet" in model_name:
            model = EfficientNetRoadBlockageCNN(num_obstacles=len(classes))
        elif "UNet" in model_name:
            bf = 16
            if "inc.conv.0.weight" in state_dict:
                bf = state_dict["inc.conv.0.weight"].shape[0]
            model = UNetRoadBlockageCNN(num_obstacles=len(classes), base_features=bf)
        elif "YOLOv8" in model_name:
            model = build_yolov8_cnn(num_obstacles=len(classes))
        elif "YOLOv5" in model_name:
            model = build_yolov5_cnn(num_obstacles=len(classes))
        elif "ResNet" in model_name:
            model = ResNetRoadBlockageCNN(num_obstacles=len(classes))
        else:
            model = EfficientNetRoadBlockageCNN(num_obstacles=len(classes))

        model.load_state_dict(state_dict)
        input_res = checkpoint.get("input_resolution", [128, 128])
        detector = cls(
            model=model,
            model_name=model_name,
            classes=classes,
            confidence_threshold=confidence_threshold,
            device=dev
        )
        detector.input_resolution = (int(input_res[0]), int(input_res[1]))
        return detector

    def preprocess_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """Standardize input image into preprocessed normalized float tensor."""
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file does not exist: {image_input}")
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:  # Grayscale
                pil_img = Image.fromarray(image_input).convert("RGB")
            elif image_input.ndim == 3 and image_input.shape[2] == 4:  # RGBA
                pil_img = Image.fromarray(image_input).convert("RGB")
            else:
                pil_img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        res = getattr(self, "input_resolution", (128, 128))
        pil_resized = pil_img.resize(res, Image.Resampling.BILINEAR)
        arr = np.array(pil_resized).astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    def predict(
        self,
        image_input: Union[str, Image.Image, np.ndarray]
    ) -> RoadBlockageResult:
        """
        Predict road blockage percentage and detected obstacles for a single image.
        """
        import time
        t0 = time.perf_counter()

        tensor = self.preprocess_image(image_input)

        with torch.no_grad():
            output = self.model(tensor)

        inference_time = (time.perf_counter() - t0) * 1000.0

        # Extract continuous percentage
        raw_pct = float(output["blockage_pct"].cpu().item())
        blockage_pct = round(float(np.clip(raw_pct, 0.0, 100.0)), 1)

        # Extract obstacle probabilities
        obs_logits = output["obstacle_logits"].cpu().squeeze(0)
        obs_probs = torch.sigmoid(obs_logits).numpy()

        detected_obstacles: List[str] = []
        conf_dict: Dict[str, float] = {}

        # Treat negligible blockage (< 9%) as clear
        if blockage_pct < 9.0:
            blockage_pct = 0.0

        for idx, cls_name in enumerate(self.classes):
            conf = float(obs_probs[idx])
            conf_dict[cls_name] = round(conf, 4)

        if blockage_pct > 0.0:
            # Sort obstacles by descending confidence
            sorted_indices = np.argsort(obs_probs)[::-1]
            top_prob = float(obs_probs[sorted_indices[0]])

            for idx in sorted_indices:
                prob = float(obs_probs[idx])
                # Include if probability is high and close to top probability (max 2 obstacles)
                if (prob >= self.threshold or prob >= top_prob - 0.10) and prob > 0.40:
                    detected_obstacles.append(self.classes[idx])
                if len(detected_obstacles) >= 2:
                    break

            if len(detected_obstacles) == 0:
                top_idx = int(sorted_indices[0])
                detected_obstacles.append(self.classes[top_idx])
        else:
            detected_obstacles = []

        is_blocked = blockage_pct > 0.5
        severity_idx = pct_to_severity(blockage_pct)
        severity = SEVERITY_LEVELS[severity_idx]

        display_names = [OBSTACLE_DISPLAY_NAMES.get(o, o) for o in detected_obstacles]
        summary = generate_natural_language_summary(blockage_pct, detected_obstacles)

        return RoadBlockageResult(
            blockage_percentage=blockage_pct,
            is_blocked=is_blocked,
            obstacles=detected_obstacles,
            obstacle_display_names=display_names,
            severity=severity,
            confidence_scores=conf_dict,
            summary=summary,
            model_name=self.model_name,
            inference_time_ms=round(inference_time, 2)
        )

    def predict_batch(
        self,
        image_inputs: List[Union[str, Image.Image, np.ndarray]]
    ) -> List[RoadBlockageResult]:
        """Run batch inference over a list of images."""
        return [self.predict(img) for img in image_inputs]
