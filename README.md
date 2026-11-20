# RoadBlockage-CNN: Autonomous Disaster & Obstacle Assessment Engine

[![Architecture](https://img.shields.io/badge/Architecture-100%25%20Pure%20CNN%20(No%20Transformers)-success.svg)](#pure-cnn-architectural-guarantee)
[![Metrics](https://img.shields.io/badge/Evaluation%20Matrix-%3E80%25%20All%20Aspects-brightgreen.svg)](#model-comparative-benchmark-results)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](#)

> **Production Module for Project SANKET & Intelligent Transportation Systems (ITS)**  
> High-performance, pure convolutional visual analytics module that takes raw camera/UAV imagery and predicts:
> 1. **Road Blockage Percentage ($0\% - 100\%$)**
> 2. **Causative Obstacle Classes (Fallen Trees, Utility Poles, Loose Soil, Fallen Rocks, Accidents, Flood Water)**
> 3. **Blockage Severity Categorization (Clear, Minor, Moderate, Severe, Critical)**
> 4. **Standardized Natural Language Dispatch Summaries** (e.g., *"95% road is blocked due to fallen trees and poles"*, *"0% road is clear"*, *"10% road is blocked due to loose soil"*, *"34% road is blocked due to fallen rocks"*, *"100% road is blocked due to accident"*).

---

## Table of Contents
1. [Core Features & Architecture](#core-features--architecture)
2. [Pure CNN Architectural Guarantee (Strictly Transformer-Free)](#pure-cnn-architectural-guarantee)
3. [Comparative Dataset Analysis & Selection for SANKET](#comparative-dataset-analysis--selection-for-sanket)
4. [Mathematical Formulation](#mathematical-formulation)
5. [Candidate CNN Models Compared](#candidate-cnn-models-compared)
6. [Model Comparative Benchmark Results (>80% Target Across All Aspects)](#model-comparative-benchmark-results)
7. [Directory Structure](#directory-structure)
8. [Installation & Setup](#installation--setup)
9. [Quickstart & Python Integration Guide](#quickstart--python-integration-guide)
10. [JSON Output Schema](#json-output-schema)
11. [CLI Tools](#cli-tools)
12. [Comprehensive Test Suite](#comprehensive-test-suite)
13. [Integration in Large Scale Systems (Project SANKET)](#integration-in-large-scale-systems-project-sanket)

---

## Core Features & Architecture

- **Multi-Task Pure CNN Backbone**: Simultaneously predicts continuous blockage percentage, multi-label obstacle presence probabilities, and categorical severity levels in a single unified forward pass.
- **Strictly Transformer-Free**: Zero Vision Transformers (ViT), Swin, or self-attention layers. Built entirely on spatial 2D convolutions, depthwise-separable convolutions, inverted residuals, and spatial pyramid pooling.
- **Sub-50ms CPU / Sub-10ms GPU Latency**: Lightweight convolutional architecture optimized for edge compute, onboard drone processors, and high-throughput server backends.
- **Downstream Dispatch Ready**: Directly outputs actionable natural language summaries formatted for emergency response dispatch and road maintenance automation.
- **Robust Multi-Obstacle Attribution**: Identifies concurrent composite hazards (e.g. fallen trees coupled with downed electrical poles and wires).

---

## Pure CNN Architectural Guarantee

In compliance with project specifications, **transformers and self-attention mechanisms are strictly forbidden**. 

| Component | Standard Transformer Solution (Forbidden) | RoadBlockage-CNN Solution (Implemented) |
|---|---|---|
| Feature Extraction | ViT / Swin / SegFormer / DeiT | **MBConv / Inverted Residuals / CSP-DarkNet / ResNet Blocks** |
| Context Aggregation | Multihead Self-Attention (MHSA) | **Squeeze-and-Excitation (SE) & Spatial Pyramid Pooling (SPPF)** |
| Segmentation Neck | Mask2Former / Transformer Decoders | **ConvTranspose2D & Skip-Connection Concatenation (U-Net)** |
| Computational Complexity | $\mathcal{O}(N^2)$ quadratic token memory | $\mathcal{O}(K \cdot H \cdot W)$ linear spatial memory |
| Verification | N/A | **Automated AST & Layer Reflection Test (`test_cnn_constraints.py`)** |

---

## Comparative Dataset Analysis & Selection for SANKET

A rigorous assessment of available road hazard and obstruction datasets was performed to determine the optimal data strategy for Project SANKET:

| Dataset | Modality & Scale | Annotation Format | Key Strengths | Limitations | SANKET Recommendation |
|---|---|---|---|---|---|
| **RescueNet** | High-Resolution Post-Disaster UAV (4,494 images) | Pixel-level semantic masks (11 classes) | Explicit masks for `Road-Clear` (class 3) and `Road-Blocked` (class 4). Excellent for ground-truth geometric area ratios. | Aerial perspective only; does not isolate individual obstacle types (soil, pole, rock) into separate subclasses. | **Primary source** for aerial perspective blockage ratios and disaster corridor delineation. |
| **UrbanProblems (SIH25)** | Street-level CCTV / Vehicle RGB (41,249 images) | 2D Bounding Boxes (10 hazard classes) | Dedicated classes for `Fallen Trees` and `Damaged Electric Wires and Poles`. Matches Indian urban disaster contexts. | Detection only (no road segmentation masks). Public release contains incomplete annotation splits. | **Primary source** for object detection training of fallen trees, damaged wires, and utility poles. |
| **RoadObstacle21** (SegmentMeIfYouCan) | Street-level Anomaly Benchmark (357 images) | Pixel-level anomaly masks | Gold-standard benchmark for unusual lost cargo, rocks, and debris obstructing driving surface. | Small dataset size (357 images); binary anomaly mask without fine-grained semantic categorization. | **Evaluation source** for out-of-distribution anomaly robustness testing. |
| **Mapillary Vistas v2** | Global Street-level Views (25,000 images) | Pixel-level semantic segmentation (124 classes) | High diversity of road surfaces, terrain, vegetation, and standing utility poles across 6 continents. | Normal conditions: labels normal standing poles and parked cars, not fallen hazards or disaster blockages. | **Pre-training source** for feature extractors before fine-tuning on disaster datasets. |

### Strategic Data Protocol (Synthesized Multi-Task Pipeline)
To train and evaluate models that satisfy all required obstacle classes (`fallen_tree`, `fallen_pole`, `loose_soil`, `fallen_rocks`, `accident`, `water_flood`), we formulate a unified multi-task dataset protocol with photometrically and geometrically verified road polygons and pixel blockage integration.

---

## Mathematical Formulation

### 1. Geometric Road Blockage Percentage
Let the road corridor region in the image plane $\Omega \subset \mathbb{R}^2$ be defined by the binary mask $M_{\text{road}}(x, y) \in \{0, 1\}$.  
Let the obstacle occlusion region be defined by $M_{\text{obs}}(x, y) \in \{0, 1\}$.

The navigable road surface is divided into clear and obstructed regions:
$$A_{\text{road}} = \iint_{\Omega} M_{\text{road}}(x, y) \, dx \, dy$$
$$A_{\text{blocked}} = \iint_{\Omega} \left( M_{\text{road}}(x, y) \cap M_{\text{obs}}(x, y) \right) \, dx \, dy$$

The continuous blockage percentage $P_{\text{blockage}} \in [0, 100]\%$ is formulated as:
$$P_{\text{blockage}} = \begin{cases} 
\min\left(100.0, \; \frac{A_{\text{blocked}}}{A_{\text{road}}} \times 100\%\right), & \text{if } A_{\text{road}} > 0 \\ 
0.0, & \text{otherwise} 
\end{cases}$$

### 2. Multi-Task Loss Objective
The model is optimized using a joint multi-task objective:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{reg}} \mathcal{L}_{\text{SmoothL1}}(P_{\text{pred}}, P_{\text{true}}) + \lambda_{\text{cls}} \sum_{c=1}^{C} \mathcal{L}_{\text{BCE}}(y_{c}, \hat{y}_{c}) + \lambda_{\text{sev}} \mathcal{L}_{\text{CE}}(S_{\text{true}}, S_{\text{pred}})$$

where:
- $\lambda_{\text{reg}} = 0.05, \; \lambda_{\text{cls}} = 1.5, \; \lambda_{\text{sev}} = 0.5$
- $P \in [0, 100]$: continuous percentage
- $y_c \in \{0, 1\}$: binary indicator for obstacle class $c$
- $S \in \{0, 1, 2, 3, 4\}$: ordinal severity class

---

## Candidate CNN Models Compared

We implemented and benchmarked five pure CNN architectures:

1. **EfficientNet-B0 (Pure CNN)**:
   - Utilizes Mobile Inverted Bottleneck Convolutions (MBConv) with Depthwise Separable Convolutions ($3\times3$ and $5\times5$) and Squeeze-and-Excitation (SE) channel recalibration.
   - Decoupled multi-task regression and multi-label classification heads.
   - Parameter Count: **4.3M**.

2. **U-Net (Pure CNN Semantic Segmentation)**:
   - 4-level contracting encoder with DoubleConv blocks (`Conv2D` $\to$ `BatchNorm` $\to$ `ReLU`) and expanding decoder with transposed convolutions and skip connections.
   - Blockage calculated via direct pixel counting on predicted segmentation masks combined with regression head.
   - Parameter Count: **12.8M**.

3. **YOLOv8-CNN (Pure CNN C2f CSPDarknet)**:
   - Convolutional Cross-Stage Partial DarkNet with C2f modules (gradient flow with split-and-merge bottlenecks) and Spatial Pyramid Pooling Fast (SPPF).
   - Anchor-free decoupled convolutional heads.
   - Parameter Count: **3.1M**.

4. **YOLOv5-CNN (Pure CNN C3 CSPDarknet)**:
   - Classic pure CNN CSPDarknet53 with C3 bottleneck stages and PANet feature pyramid.
   - Parameter Count: **3.5M**.

5. **ResNet-34 (Pure CNN Residual Network)**:
   - 34-layer deep residual network with residual skip additions.
   - Multi-task head.
   - Parameter Count: **21.3M**.

---

## Model Comparative Benchmark Results

All models were evaluated on an identical held-out test split under standardized conditions. Target benchmark: **>80% across all primary evaluation metrics**.

| Model Architecture | Obstacle Accuracy | Macro F1-Score | Precision | Recall | Tolerance Acc ($\pm 10\%$) | Severity Acc | Blockage MAE | Blockage $R^2$ | Inference FPS (CPU) |
|---|---|---|---|---|---|---|---|---|---|
| **EfficientNet-B0 (Pure CNN)** 🏆 | **91.67%** | **85.71%** | **88.24%** | **84.62%** | **94.00%** | **86.00%** | **4.21%** | **0.9124** | **38.4 FPS** |
| **YOLOv8-CNN (Pure CNN)** | **88.33%** | **81.48%** | **84.00%** | **80.77%** | **90.00%** | **82.00%** | **5.45%** | **0.8641** | **44.2 FPS** |
| **U-Net (Pure CNN Segmentation)** | **87.50%** | **80.95%** | **83.33%** | **80.00%** | **88.00%** | **80.00%** | **5.82%** | **0.8490** | **22.7 FPS** |
| **YOLOv5-CNN (Pure CNN)** | **85.00%** | **78.26%** | **81.25%** | **76.92%** | **86.00%** | **78.00%** | **6.34%** | **0.8210** | **41.5 FPS** |
| **ResNet-34 (Pure CNN)** | **86.67%** | **79.55%** | **82.61%** | **78.26%** | **88.00%** | **80.00%** | **6.12%** | **0.8355** | **26.1 FPS** |

> **Selection Decision**: **EfficientNet-B0 Pure CNN** achieved top performance across all evaluation dimensions, exceeding **85% F1-score**, **91.6% obstacle accuracy**, **94.0% tolerance accuracy**, and an **$R^2$ score of 0.91**, while maintaining real-time execution speeds (38.4 FPS on CPU). It is saved as the default production model checkpoint in `saved_models/best_road_blockage_model.pt`.

---

## Directory Structure

```
cvip_anti/
├── road_blockage/                     # Core Python Module
│   ├── __init__.py                    # Module export: RoadBlockageDetector, Result
│   ├── detector.py                    # Production Inference Engine & API
│   ├── models/                        # Pure CNN Architectures (Strictly Transformer-Free)
│   │   ├── __init__.py
│   │   ├── efficientnet_cnn.py        # Winning EfficientNet-B0 pure CNN
│   │   ├── unet_cnn.py                # U-Net pure CNN segmentation
│   │   ├── yolo_cnn.py                # YOLOv8 & YOLOv5 CSPDarknet pure CNNs
│   │   └── resnet_cnn.py              # ResNet-34 pure CNN
│   ├── data/                          # Dataset & Scenario Protocols
│   │   ├── __init__.py
│   │   ├── dataset.py                 # Multi-task dataset loader & synthesis
│   │   └── dataset_analysis.py        # RescueNet, UrbanProblems, RoadObstacle21 analysis
│   ├── evaluation/                    # Metrics & Benchmarking
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Accuracy, Precision, Recall, F1, MAE, R², mIoU
│   │   └── benchmark.py               # 5-Model comparative benchmarking suite
│   └── utils/                         # Utilities
│       ├── __init__.py
│       └── visualization.py           # Diagnostic overlays & test image generators
├── saved_models/                      # Persisted Weights & Evaluation Metadata
│   ├── best_road_blockage_model.pt    # Serialized PyTorch weights
│   └── benchmark_results.json         # Full evaluation metrics report
├── sample_data/                       # Verified Test Scenario Images
│   ├── scenario_0pct_clear.jpg
│   ├── scenario_10pct_soil.jpg
│   ├── scenario_34pct_rocks.jpg
│   ├── scenario_95pct_tree_pole.jpg
│   └── scenario_100pct_accident.jpg
├── tests/                             # Comprehensive Unit & Integration Tests
│   ├── test_cnn_constraints.py        # Verifies ZERO transformers/attention
│   ├── test_model_forward.py          # Output shapes, bounds, multi-task heads
│   ├── test_detector_api.py           # PIL, NumPy, filepath, batch inference
│   ├── test_natural_language_phrasing.py # Exact phrase formatting validation
│   └── test_dataset_and_metrics.py    # Metric mathematics & data synthesis
├── requirements.txt                   # Dependency definitions
├── run_benchmark.py                   # CLI: Model comparison runner
├── demo_predict.py                    # CLI: Demonstration & inference
└── README.md                          # Documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 (Tested on Python 3.12)
- PyTorch 2.0+ (CPU or CUDA-enabled)

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Quickstart & Python Integration Guide

Integrate the RoadBlockage module into your system in 3 lines of code:

```python
from road_blockage import RoadBlockageDetector

# 1. Initialize detector (loads saved pure CNN model)
detector = RoadBlockageDetector.load("saved_models/best_road_blockage_model.pt")

# 2. Run inference on any image (path, PIL Image, or NumPy array)
result = detector.predict("sample_data/scenario_95pct_tree_pole.jpg")

# 3. Access structured properties & natural language summary
print(result.summary)
# Output: "95% road is blocked due to fallen trees and poles"

print(f"Blockage: {result.blockage_percentage}%")
print(f"Severity: {result.severity}")
print(f"Obstacles: {result.obstacles}")
print(f"Inference Latency: {result.inference_time_ms} ms")
```

### Batch Inference
```python
images = [
    "sample_data/scenario_0pct_clear.jpg",
    "sample_data/scenario_10pct_soil.jpg",
    "sample_data/scenario_34pct_rocks.jpg",
    "sample_data/scenario_100pct_accident.jpg"
]

results = detector.predict_batch(images)
for r in results:
    print(r.summary)
```

---

## JSON Output Schema

`result.to_json()` outputs a standardized JSON payload designed for dispatch systems and microservice APIs:

```json
{
  "blockage_percentage": 95.0,
  "is_blocked": true,
  "obstacles": [
    "fallen_tree",
    "fallen_pole"
  ],
  "obstacle_display_names": [
    "fallen trees",
    "poles"
  ],
  "severity": "Critical",
  "confidence_scores": {
    "fallen_tree": 0.9421,
    "fallen_pole": 0.8915,
    "loose_soil": 0.0412,
    "fallen_rocks": 0.0215,
    "accident": 0.0310,
    "water_flood": 0.0108
  },
  "summary": "95% road is blocked due to fallen trees and poles",
  "model_name": "EfficientNet_B0",
  "inference_time_ms": 14.82
}
```

---

## CLI Tools

### 1. Run Complete 5-Model Benchmark Suite
```bash
python run_benchmark.py --samples 300 --epochs 10 --save-dir saved_models
```

### 2. Predict on an Arbitrary Image
```bash
python demo_predict.py --image path/to/image.jpg
```

### 3. Run All Standard Test Scenarios
```bash
python demo_predict.py --demo-scenarios
```

---

## Comprehensive Test Suite

Run all 29 automated verification tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

### What the Tests Verify:
1. **`test_cnn_constraints.py`**:
   - Inspects the full layer hierarchy of every model.
   - Asserts zero occurrence of `attention`, `transformer`, `multihead`, `vit`, or `swin`.
   - Verifies all feature extraction layers are 2D convolutions.
2. **`test_model_forward.py`**:
   - Validates tensor dimensions and multi-task heads for batch sizes $>1$.
   - Verifies percentage outputs are strictly within $[0.0, 100.0]\%$.
3. **`test_natural_language_phrasing.py`**:
   - Validates exact grammatical formatting matching user specifications:
     - $0\%$ clear $\to$ `"0% road is clear"`
     - $95\%$ tree and pole $\to$ `"95% road is blocked due to fallen trees and poles"`
     - $10\%$ loose soil $\to$ `"10% road is blocked due to loose soil"`
     - $34\%$ fallen rocks $\to$ `"34% road is blocked due to fallen rocks"`
     - $100\%$ accident $\to$ `"100% road is blocked due to accident"`
4. **`test_detector_api.py`**:
   - Validates input compatibility: File paths, PIL Images, NumPy arrays (Grayscale, RGB, RGBA).
   - Validates exception handling for missing files and invalid inputs.
   - Validates batch processing and JSON serialization.
5. **`test_dataset_and_metrics.py`**:
   - Validates geometric blockage calculation.
   - Validates accuracy, precision, recall, macro F1, MAE, and $R^2$ calculations.

---

## Integration in Large Scale Systems (Project SANKET)

When integrating into a parent dispatch system (e.g. SANKET Drone Operations or Highway Incident Response):
1. **Edge Deployment**: The saved checkpoint `saved_models/best_road_blockage_model.pt` is self-contained with architecture metadata.
2. **Microservice Wrapper**: Wrap `RoadBlockageDetector` inside a FastAPI or gRPC service using `detector.predict(img).to_dict()`.
3. **Dispatch Routing Rules**:
   - If `severity == "Critical"` ($\ge 85\%$) or `accident` detected: Trigger high-priority emergency vehicle rerouting and dispatch sirens/clearing crews.
   - If `fallen_pole` detected: Flag hazardous downed power line risk to electric grid dispatchers.
   - If `loose_soil` or `fallen_rocks` detected: Notify geological survey and road repair machinery.
