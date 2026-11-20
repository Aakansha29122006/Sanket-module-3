"""
Road Blockage Dataset Module
Implements multi-task dataset formatting, data synthesis for road obstruction scenarios,
and evaluation batch generators.
"""

import math
import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFilter

OBSTACLE_CLASSES: List[str] = [
    "fallen_tree",
    "fallen_pole",
    "loose_soil",
    "fallen_rocks",
    "accident",
    "water_flood"
]

SEVERITY_LEVELS: List[str] = [
    "Clear",         # 0%
    "Minor",         # 1 - 25%
    "Moderate",      # 26 - 60%
    "Severe",        # 61 - 85%
    "Critical"       # 86 - 100%
]

def pct_to_severity(pct: float) -> int:
    """Map continuous blockage percentage (0-100) to severity class index."""
    if pct <= 0.5:
        return 0
    elif pct <= 25.0:
        return 1
    elif pct <= 60.0:
        return 2
    elif pct <= 85.0:
        return 3
    else:
        return 4

def generate_synthetic_road_scene(
    width: int = 256,
    height: int = 256,
    target_blockage: Optional[float] = None,
    obstacles: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate a photometrically consistent synthetic road scene with road geometry,
    obstacle occlusions, exact pixel ground truth masks, and blockage percentage.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 1. Base background (terrain/grass/urban)
    bg_color = (
        random.randint(40, 70),
        random.randint(60, 100),
        random.randint(30, 60)
    )
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # 2. Road geometry (trapezoidal perspective road corridor)
    horizon_y = int(height * random.uniform(0.35, 0.45))
    top_w = int(width * random.uniform(0.15, 0.25))
    top_cx = int(width * random.uniform(0.45, 0.55))
    bot_w = int(width * random.uniform(0.70, 0.85))
    bot_cx = int(width * random.uniform(0.45, 0.55))

    road_poly = [
        (top_cx - top_w // 2, horizon_y),
        (top_cx + top_w // 2, horizon_y),
        (bot_cx + bot_w // 2, height - 1),
        (bot_cx - bot_w // 2, height - 1)
    ]

    road_color = (random.randint(70, 90), random.randint(70, 90), random.randint(75, 95))
    draw.polygon(road_poly, fill=road_color)

    # Center dash line
    dash_color = (220, 210, 80)
    draw.line([((top_cx), horizon_y), ((bot_cx), height - 1)], fill=dash_color, width=max(1, width // 100))

    # Road mask for area calculation (0=bg, 1=road)
    road_mask_img = Image.new("L", (width, height), 0)
    draw_rm = ImageDraw.Draw(road_mask_img)
    draw_rm.polygon(road_poly, fill=1)
    road_area_total = np.sum(np.array(road_mask_img) == 1)

    # Obstacle selection
    if target_blockage is None:
        target_blockage = random.choice([0.0, 10.0, 25.0, 34.0, 50.0, 75.0, 95.0, 100.0])

    if target_blockage <= 0.5:
        chosen_obstacles = []
    elif obstacles is not None:
        chosen_obstacles = obstacles
    else:
        num_obs = random.choices([1, 2, 3], weights=[0.7, 0.25, 0.05])[0]
        chosen_obstacles = random.sample(OBSTACLE_CLASSES, min(num_obs, len(OBSTACLE_CLASSES)))

    # 3. Render obstacles and track obstacle mask on the road
    obs_mask_img = Image.new("L", (width, height), 0)
    draw_obs = ImageDraw.Draw(obs_mask_img)

    if chosen_obstacles and target_blockage > 0.5:
        # Scale obstacle coverage to match target blockage
        desired_ratio = min(1.0, max(0.05, target_blockage / 100.0))
        road_h = height - horizon_y
        obs_band_h = max(8, int(road_h * desired_ratio))

        # Position obstacle band along the road
        if desired_ratio > 0.85:
            band_y0 = horizon_y + int(road_h * 0.05)
            band_y1 = min(height - 1, band_y0 + obs_band_h)
        else:
            max_start = max(horizon_y, height - obs_band_h - 5)
            band_y0 = random.randint(horizon_y, max_start)
            band_y1 = min(height - 1, band_y0 + obs_band_h)

        # Get road bounds at this band
        band_mask = np.zeros((height, width), dtype=bool)
        band_mask[band_y0:band_y1, :] = True
        road_arr_full = np.array(road_mask_img) == 1
        target_road_zone = np.logical_and(road_arr_full, band_mask)

        # Obstacle bounding box
        cols_in_zone = np.where(target_road_zone.any(axis=0))[0]
        if len(cols_in_zone) > 0:
            x0 = max(0, cols_in_zone[0] - 5)
            x1 = min(width - 1, cols_in_zone[-1] + 5)
        else:
            x0 = max(0, bot_cx - bot_w // 2)
            x1 = min(width - 1, bot_cx + bot_w // 2)
        y0 = band_y0
        y1 = max(band_y0 + 2, band_y1)
        x0, x1 = min(x0, x1), max(x0, x1)
        if x1 - x0 < 4:
            x1 = min(width - 1, x0 + 4)

        for obs in chosen_obstacles:
            if obs == "fallen_tree":
                trunk_color = (95, 55, 30)
                foliage_color = (40, 90, 35)
                # Primary trunk
                draw.line([(x0, y0 + (y1 - y0) // 2), (x1, y1)], fill=trunk_color, width=max(6, width // 25))
                draw_obs.line([(x0, y0 + (y1 - y0) // 2), (x1, y1)], fill=1, width=max(6, width // 25))
                # Dense foliage covering target road band
                num_foliage = max(10, int(30 * desired_ratio))
                for _ in range(num_foliage):
                    fx = random.randint(x0, x1)
                    fy = random.randint(y0, y1)
                    fr = random.randint(max(4, width // 25), max(8, width // 10))
                    draw.ellipse([fx - fr, fy - fr, fx + fr, fy + fr], fill=foliage_color)
                    draw_obs.ellipse([fx - fr, fy - fr, fx + fr, fy + fr], fill=1)
            elif obs == "fallen_pole":
                pole_color = (160, 165, 175)
                wire_color = (30, 30, 30)
                # Diagonal poles
                draw.line([(x0, y1), (x1, y0)], fill=pole_color, width=max(4, width // 35))
                draw_obs.line([(x0, y1), (x1, y0)], fill=1, width=max(6, width // 25))
                draw.line([(x0 + 10, y0), (x1 - 10, y1)], fill=pole_color, width=max(3, width // 45))
                draw_obs.line([(x0 + 10, y0), (x1 - 10, y1)], fill=1, width=max(5, width // 30))
                # Tangled wires & transformer debris
                for _ in range(8):
                    wx0 = random.randint(x0, x1)
                    wy0 = random.randint(y0, y1)
                    wx1 = random.randint(x0, x1)
                    wy1 = random.randint(y0, y1)
                    draw.line([(wx0, wy0), (wx1, wy1)], fill=wire_color, width=2)
                    draw_obs.line([(wx0, wy0), (wx1, wy1)], fill=1, width=max(3, width // 40))
            elif obs == "loose_soil":
                soil_color = (130, 85, 45)
                poly = [
                    (x0, y1),
                    (x0 + (x1 - x0) // 4, y0),
                    (x0 + 3 * (x1 - x0) // 4, y0 + max(2, (y1 - y0) // 6)),
                    (x1, y1)
                ]
                draw.polygon(poly, fill=soil_color)
                draw_obs.polygon(poly, fill=1)
            elif obs == "fallen_rocks":
                rock_color = (120, 120, 120)
                num_rocks = max(6, int(20 * desired_ratio))
                for _ in range(num_rocks):
                    rx = random.randint(x0, x1)
                    ry = random.randint(y0, y1)
                    rr = random.randint(max(4, width // 22), max(8, width // 10))
                    draw.ellipse([rx - rr, ry - rr, rx + rr, ry + rr], fill=rock_color, outline=(80, 80, 80))
                    draw_obs.ellipse([rx - rr, ry - rr, rx + rr, ry + rr], fill=1)
            elif obs == "accident":
                car_color = (random.randint(180, 240), 40, 40)
                draw.rectangle([x0, y0, x1, y1], fill=car_color)
                draw_obs.rectangle([x0, y0, x1, y1], fill=1)
                # Crushed vehicle components (safeguarded inner bounds)
                if (x1 - x0) > 12 and (y1 - y0) > 12:
                    draw.rectangle([x0 + 4, y0 + 4, x1 - 4, y1 - 4], fill=(30, 30, 30))
            elif obs == "water_flood":
                water_color = (50, 90, 130)
                draw.rectangle([x0, y0, x1, y1], fill=water_color)
                draw_obs.rectangle([x0, y0, x1, y1], fill=1)

    # Compute exact ground truth geometric blockage
    road_arr = np.array(road_mask_img) == 1
    obs_arr = np.array(obs_mask_img) == 1
    blocked_on_road = np.logical_and(road_arr, obs_arr)

    actual_road_pixels = np.sum(road_arr)
    blocked_pixels = np.sum(blocked_on_road)

    if actual_road_pixels > 0:
        calc_pct = (blocked_pixels / actual_road_pixels) * 100.0
    else:
        calc_pct = 0.0

    # Clean exact percentage
    calc_pct = float(np.clip(calc_pct, 0.0, 100.0))
    if len(chosen_obstacles) == 0:
        calc_pct = 0.0

    # Semantic segmentation ground truth: 0=background, 1=road_clear, 2=road_blocked
    seg_mask = np.zeros((height, width), dtype=np.int64)
    seg_mask[road_arr] = 1
    seg_mask[blocked_on_road] = 2

    # Multi-label binary vector
    label_vec = np.zeros(len(OBSTACLE_CLASSES), dtype=np.float32)
    for obs in chosen_obstacles:
        if obs in OBSTACLE_CLASSES:
            label_vec[OBSTACLE_CLASSES.index(obs)] = 1.0

    severity = pct_to_severity(calc_pct)

    return {
        "image": img,
        "blockage_pct": calc_pct,
        "obstacles": chosen_obstacles,
        "obstacle_labels": label_vec,
        "severity": severity,
        "road_mask": seg_mask,
        "road_pixels": int(actual_road_pixels),
        "blocked_pixels": int(blocked_pixels)
    }


class RoadBlockageDataset(Dataset):
    """
    Multi-Task PyTorch Dataset for Road Blockage and Obstacle Identification.
    Produces image tensors, blockage percentage regression targets,
    multi-label obstacle vectors, and segmentation masks.
    """
    def __init__(
        self,
        num_samples: int = 500,
        img_size: int = 224,
        seed: int = 42,
        augment: bool = False
    ):
        self.num_samples = num_samples
        self.img_size = img_size
        self.augment = augment
        self.samples: List[Dict[str, Any]] = []

        # Pre-generate reproducible diverse samples
        rng = random.Random(seed)
        blockage_distribution = [
            0.0, 0.0, 5.0, 10.0, 15.0, 25.0, 34.0, 45.0,
            60.0, 75.0, 85.0, 95.0, 100.0
        ]
        
        for i in range(num_samples):
            s_seed = seed + i
            desired_pct = rng.choice(blockage_distribution)
            sample = generate_synthetic_road_scene(
                width=img_size,
                height=img_size,
                target_blockage=desired_pct,
                seed=s_seed
            )
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        img = item["image"]

        # PIL to normalized FloatTensor (3, H, W)
        img_np = np.array(img).astype(np.float32) / 255.0
        # ImageNet standardization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        pct_tensor = torch.tensor([item["blockage_pct"]], dtype=torch.float32)
        label_tensor = torch.from_numpy(item["obstacle_labels"]).float()
        severity_tensor = torch.tensor(item["severity"], dtype=torch.long)
        mask_tensor = torch.from_numpy(item["road_mask"]).long()

        return {
            "image": img_tensor,
            "blockage_pct": pct_tensor,
            "obstacle_labels": label_tensor,
            "severity": severity_tensor,
            "mask": mask_tensor
        }
