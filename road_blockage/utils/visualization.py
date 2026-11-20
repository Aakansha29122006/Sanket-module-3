"""
Visualization and Sample Data Generation Utilities
Creates annotated output overlays and test images matching the user scenarios.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..data.dataset import generate_synthetic_road_scene


def generate_scenario_test_images(output_dir: str = "sample_data") -> Dict[str, str]:
    """
    Generate test scenarios specified in the user prompt:
    1. 0% clear road
    2. 95% blocked due to fallen trees and poles
    3. 10% blocked due to loose soil
    4. 34% blocked due to fallen rocks
    5. 100% blocked due to accident
    """
    os.makedirs(output_dir, exist_ok=True)
    scenarios = [
        {"name": "scenario_0pct_clear.jpg", "pct": 0.0, "obstacles": []},
        {"name": "scenario_95pct_tree_pole.jpg", "pct": 95.0, "obstacles": ["fallen_tree", "fallen_pole"]},
        {"name": "scenario_10pct_soil.jpg", "pct": 10.0, "obstacles": ["loose_soil"]},
        {"name": "scenario_34pct_rocks.jpg", "pct": 34.0, "obstacles": ["fallen_rocks"]},
        {"name": "scenario_100pct_accident.jpg", "pct": 100.0, "obstacles": ["accident"]},
    ]

    saved_paths: Dict[str, str] = {}
    for sc in scenarios:
        data = generate_synthetic_road_scene(
            width=256,
            height=256,
            target_blockage=sc["pct"],
            obstacles=sc["obstacles"],
            seed=100 + int(sc["pct"])
        )
        filepath = os.path.join(output_dir, sc["name"])
        data["image"].save(filepath, quality=95)
        saved_paths[sc["name"]] = filepath

    return saved_paths


def annotate_prediction_overlay(
    image: Image.Image,
    summary_text: str,
    blockage_pct: float,
    severity: str
) -> Image.Image:
    """Draw a diagnostic overlay badge onto the image for logging/visualization."""
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    # Severity color
    sev_colors = {
        "Clear": (46, 125, 50),
        "Minor": (104, 159, 56),
        "Moderate": (245, 124, 0),
        "Severe": (211, 47, 47),
        "Critical": (183, 28, 28)
    }
    badge_color = sev_colors.get(severity, (30, 30, 30))

    # Top banner background
    banner_h = int(h * 0.16)
    draw.rectangle([0, 0, w, banner_h], fill=(20, 20, 20))

    # Text overlay
    draw.text((10, 8), summary_text, fill=(255, 255, 255))
    draw.text((10, 24), f"Severity: {severity} | Model: Pure CNN", fill=badge_color)

    # Progress bar on the bottom
    bar_h = 6
    draw.rectangle([0, h - bar_h, w, h], fill=(60, 60, 60))
    blocked_w = int((blockage_pct / 100.0) * w)
    draw.rectangle([0, h - bar_h, blocked_w, h], fill=badge_color)

    return annotated
