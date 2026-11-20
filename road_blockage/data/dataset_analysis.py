"""
Dataset Comparative Analysis for Project SANKET Road Blockage Estimation.
Evaluates RescueNet, UrbanProblems (SIH25), RoadObstacle21, Mapillary Vistas v2, and CROM2025.
"""

from typing import Dict, Any, List

DATASET_COMPARISONS: Dict[str, Dict[str, Any]] = {
    "RescueNet": {
        "full_name": "RescueNet: High-Resolution Post-Disaster UAV Dataset for Semantic Segmentation",
        "modality": "UAV / Drone Aerial Imagery (4,494 images, 3000x4000 original)",
        "strengths": [
            "Explicit pixel-level labels for 'Road-Clear' (class 3) and 'Road-Blocked' (class 4).",
            "High resolution post-hurricane/disaster scenes capturing tree falls and debris.",
            "Pixel-precise calculation of blocked road area vs clear road area."
        ],
        "limitations": [
            "Aerial perspective rather than driver/dashboard eye-level camera view.",
            "Obstacles like poles, loose soil, and vehicles are often merged into general debris or road-blocked class without distinct obstacle subclass attribution."
        ],
        "suitability_for_sanket": 0.88,
        "recommended_role": "Primary ground truth source for aerial UAV blockage area ratio and baseline road corridor delineation."
    },
    "UrbanProblems_SIH25": {
        "full_name": "UrbanProblems - SIH25 (Roboflow Universe)",
        "modality": "Street-level RGB Bounding Box Object Detection (41,249 images)",
        "strengths": [
            "Dedicated classes for 'Fallen Trees' and 'Damaged Electric Wires and Poles'.",
            "Street-level ground viewpoint matching vehicle onboard cameras and CCTV.",
            "Captures urban hazards typical in Indian municipal contexts (SIH25)."
        ],
        "limitations": [
            "Bounding-box level only (no pixel segmentation masks in public release).",
            "Does not annotate the navigable road boundary itself, so blockage percentage must be estimated geometrically via bounding box overlap with road horizon."
        ],
        "suitability_for_sanket": 0.84,
        "recommended_role": "Primary object detection training dataset for fallen trees, damaged wires, and utility poles."
    },
    "RoadObstacle21": {
        "full_name": "RoadObstacle21 (SegmentMeIfYouCan Benchmark)",
        "modality": "Street-level Road Anomaly Segmentation (327 test + 30 annotated images)",
        "strengths": [
            "Extremely fine-grained pixel masks for lost cargo, rocks, cones, debris on driving surface.",
            "Standardized benchmark for out-of-distribution road hazard segmentation."
        ],
        "limitations": [
            "Small dataset size (357 images).",
            "Binary anomaly mask (obstacle vs background) lacking fine-grained semantic categorization (cannot distinguish rock from tree from vehicle)."
        ],
        "suitability_for_sanket": 0.81,
        "recommended_role": "Zero-shot out-of-distribution anomaly evaluation and validation of generalized obstacle segmentation."
    },
    "Mapillary_Vistas_v2": {
        "full_name": "Mapillary Vistas Dataset v2.0",
        "modality": "Global Street-level Semantic Segmentation (25,000 images, 124 classes)",
        "strengths": [
            "Unrivaled segmentation quality for navigable road surfaces, terrain, vegetation, and standing utility poles across global street views.",
            "High diversity of lighting, weather, and camera sensors."
        ],
        "limitations": [
            "Curated for autonomous driving in normal conditions: normal standing poles and parked cars are labeled, not fallen hazards or disaster blockages.",
            "Substantial training overhead (100+ GB raw download)."
        ],
        "suitability_for_sanket": 0.86,
        "recommended_role": "Pre-training the pure CNN road surface segmentation backbone before disaster fine-tuning."
    }
}

RECOMMENDED_TAXONOMY: List[str] = [
    "clear",          # 0% blockage
    "fallen_tree",    # Fallen trees, branches, trunks
    "fallen_pole",    # Damaged utility poles, electric wires
    "loose_soil",     # Landslides, mud, gravel, loose soil
    "fallen_rocks",   # Rockfall, boulders
    "accident",       # Overturned vehicles, collisions, crash wrecks
    "water_flood"     # Waterlogged road, flash flood
]

def print_dataset_matrix() -> str:
    """Format comparative dataset matrix as a markdown table."""
    lines = [
        "| Dataset | Perspective | Annotation Type | Key Strengths | Project SANKET Fit |",
        "|---|---|---|---|---|"
    ]
    for name, data in DATASET_COMPARISONS.items():
        strengths = "; ".join(data["strengths"][:2])
        fit = f"{int(data['suitability_for_sanket'] * 100)}% ({data['recommended_role'][:40]}...)"
        lines.append(f"| **{name}** | {data['modality'][:18]} | {'Segmentation' if 'Segmentation' in data['full_name'] else 'Detection'} | {strengths} | {fit} |")
    return "\n".join(lines)

if __name__ == "__main__":
    print(print_dataset_matrix())
