# SANKET — AI-Powered Landslide Early Warning and Road Obstruction Analysis

> **Module 3: Road Obstruction Detection, Quantification and Cause Analysis**  
> Part of the SANKET multi-module landslide monitoring and early-warning platform

**Project:** SANKET  
**Focus:** Landslide risk intelligence, damage assessment, road accessibility, and emergency response support  
**Primary geography:** Eight northeastern states of India  
**Module 3 stack:** Python, PyTorch, FastAPI, a lightweight web interface, and geospatial data  
**Deployment objective:** Locally runnable Windows demonstration, without Docker

## 1. Project overview

SANKET is a geospatial AI/ML platform designed to support landslide preparedness and situational awareness in landslide-prone regions. Its goal is to connect **prediction of where landslides may occur** with **observation of what has happened on the ground** and the potential implications for roads and communities.

The system is organized into four complementary modules:

| Module | Responsibility | Primary output |
| --- | --- | --- |
| **1 — Landslide Risk Prediction** | Assess landslide susceptibility or risk using relevant environmental and terrain factors | Area-based risk level and supporting risk indicators |
| **2 — Landslide Scar Segmentation** | Identify landslide-affected terrain using satellite imagery | Landslide extent and spatial overlays |
| **3 — Road Obstruction Analysis** | Examine uploaded road photographs for obstruction materials and the extent of visible road coverage | Obstruction masks, material labels, coverage estimate, location-linked report |
| **4 — Crack Progression Monitoring** | Track observable crack development over time | Change indicators and supporting monitoring evidence |

These modules answer different questions. **Module 1 is predictive and area-oriented; Module 3 is observational and image-oriented.** Together, they can connect a warning about potential landslide risk with photographic evidence of a possible impact on road access.

SANKET is a **decision-support and research system**, not an authoritative road-closure service or a substitute for field inspection, geological assessment, or emergency-agency decisions.

## 2. Why Module 3 matters

A landslide risk score alone cannot tell a user whether a particular road is blocked. A road may be affected by rocks, soil, mud, fallen trees, floodwater, or damage even when the wider region's prediction is uncertain. Conversely, high landslide risk does not mean a road is already obstructed.

Module 3 closes part of this information gap by analyzing photographic evidence of a road. Its objectives are to:

- Recognize **observable obstruction materials**, such as rocks/boulders, loose soil, mud/sediment, mixed debris, fallen trees/branches, water, and damaged roadway, when supported by trained classes.
- Locate the **road footprint** and **obstruction regions** within an image.
- Estimate **the proportion of the projected road footprint covered by visible obstructions**; separately evaluate remaining unobstructed width when possible.
- Distinguish the **observed material** from a **suspected cause** (for example, rocks do not by themselves prove a rockfall or landslide).
- Associate results with a location from photo EXIF GPS, permission-based device geolocation, or manually confirmed coordinates.
- Produce an annotated image and a concise machine-readable and human-readable report.
- Compare candidate CNN-based architectures under reproducible evaluation conditions.

The desired user journey is simple: **upload a road photograph → select Analyse → review the annotated result and report**. Missing GPS permission or low-confidence vision predictions should not prevent a clear explanation of what is and is not known.

## 3. Main functions and outputs

| Capability | Mechanism | Expected output |
| --- | --- | --- |
| Image input | Validate type, decode image, correct orientation, normalize for inference | Validated image and metadata |
| Road segmentation | Predict the approximate full road footprint, including occluded portions where feasible | Road mask |
| Obstruction segmentation | Detect/segment supported objects or materials on the road | Obstacle masks and labels |
| Quantification | Calculate overlap of road and obstruction masks | Estimated image-plane coverage percentage |
| Material classification | Recognize visible class(es) from region masks/crops or a classifier | Per-material predictions and confidence |
| Cause assessment | Infer a *possible* event using appropriate labelled evidence or contextual information | Suspected cause or `unknown` |
| Location capture | Read EXIF; optionally request browser position; accept manual location | Coordinates, source, accuracy if available |
| Severity summary | Apply documented descriptive rules and model uncertainty | Clear/partial/severe/potentially complete/unknown |
| Reporting | Combine predictions, evidence, model provenance, and limitations | On-screen summary, annotated image, JSON/PDF exports |
| Model comparison | Run trained, compatible models on labelled evaluation sets | Metrics, failure examples, and model selection rationale |

**Not every output is always available.** If road boundaries, obstacle masks, or usable labels are missing, the system must return an explicit `unable_to_determine` or `unknown` result rather than inventing a number or cause.

## 4. How Module 3 works

```text
Road photograph
      |
      v
Upload validation + EXIF parsing + preprocessing
      |
      +---------------------+------------------------+
      |                     |                        |
      v                     v                        v
Road-footprint model   Obstacle mask model    Material/cause classifier
      |                     |                        |
      +----------+----------+------------------------+
                 |
                 v
       Mask alignment and quality checks
                 |
                 v
      Road/obstacle overlap calculation
                 |
                 v
     Location and Module 1 context linkage
                 |
                 v
      Annotated image + structured report
```

### 4.1 Road identification

The road segmentation model predicts where the road lies within the photograph. A key challenge is identifying the **full road footprint**, including parts partially hidden by debris or vehicles. If the model only segments visible uncovered asphalt, the denominator in the coverage calculation will be too small. Perspective, shadows, unusual road materials, and narrow Himalayan roads must be considered in evaluation.

### 4.2 Obstruction detection and material recognition

A detection/segmentation model locates individual obstructions or semantic material regions. A classification model can assign supported labels, potentially using obstacle crops. The taxonomy should distinguish **rock/boulder**, **loose soil**, **mud/sediment**, **mixed debris**, **fallen tree/branches**, **water**, **damaged road**, **other**, and **unknown**. Class names must be backed by actual annotations: a dataset's generic `obstacle` label must not be silently converted to a specific material.

### 4.3 Estimating blockage percentage

Given aligned binary masks for the road footprint \(R\) and obstruction \(O\):

\[
\text{Projected road coverage (\%)} = 100 \times \frac{|R \cap O|}{|R|}
\]

- Use the **union** of overlapping obstruction masks to avoid double-counting pixels.
- Ignore obstacle pixels outside the predicted road region.
- Preserve a quality flag if occlusion or road-footprint uncertainty is substantial.
- If \(|R|=0\), return `unable_to_determine`, not `0%`.
- Report the statistic as **projected image-plane coverage**, not measured ground area or proof that vehicles can pass.

Where feasible, a separate analysis can estimate the narrowest continuous unobstructed road-width fraction. Neither coverage nor width fraction alone provides a certified passability decision.

### 4.4 Cause and severity

The model should report **what is visible** before **what may have caused it**. For instance:

- Observed: rocks and loose soil; suspected cause: landslide or rockfall (**unconfirmed**).
- Observed: fallen branches; suspected cause: tree fall (**unconfirmed**).
- Observed: standing water; suspected cause: flooding (**unconfirmed**).

When contextual evidence is insufficient, return `unknown`. Severity labels describe the **visible obstruction estimate**, not a legally or operationally verified road closure.

### 4.5 Geolocation provenance

Three independent location sources are supported in the intended workflow:

1. **Image EXIF GPS** — coordinates recorded by the camera at capture time, if present.
2. **Browser/device location** — the uploader's current location, obtained only with permission; this may be far from the photographed road.
3. **Manual location** — coordinates selected or confirmed by the user.

The interface should display source and uncertainty, keep conflicting values separate, and never silently claim the uploader's location is the road's location. Any location match to OpenStreetMap roads is a contextual approximation requiring its own quality checks.

## 5. How Module 3 connects to Module 1

### 5.1 Distinct roles

**Module 1 — Landslide Risk Prediction** analyses geographic/environmental inputs such as rainfall (e.g. IMERG), soil moisture (e.g. SMAP), elevation/slope (DEM), and other available factors. It assigns area-based categories such as **Normal, Watch, Warning, and Danger**, subject to its actual model definitions and validation.

**Module 3 — Road Obstruction Analysis** examines a specific photograph and outputs visual evidence of a possible road impact. It does **not** use a risk score as proof that an obstruction exists.

### 5.2 Proposed integration sequence

1. **Risk context:** Module 1 generates a location- and time-referenced risk prediction for a geographical area.
2. **Incident evidence:** A user uploads a road photograph to Module 3, with EXIF, device, or manually confirmed geographic location.
3. **Geospatial matching:** The backend checks whether the reported photo location falls within, or near, an area covered by a Module 1 risk prediction, considering timestamp, spatial resolution, and GPS accuracy.
4. **Independent analysis:** Module 3 predicts road and obstruction masks, obstruction materials, projected coverage, and a qualified suspected cause, **without forcing its output to agree with Module 1**.
5. **Joint incident view:** The UI/report presents **predicted regional risk** alongside **photographic road condition evidence**, preserving their distinct sources, confidence, time windows, and limitations.
6. **Optional escalation:** A policy layer may flag high-priority records for human review if high predicted risk overlaps a credible reported obstruction; this must be labelled as a review priority, not an automatic road closure.

```text
MODULE 1: ENVIRONMENTAL / GEOSPATIAL RISK
Rainfall + soil moisture + DEM + other indicators
                      |
                      v
             Risk level + area + time
                      |
                      v
                [Spatial/time match]
                      ^
                      |
MODULE 3: PHOTOGRAPHIC ROAD EVIDENCE
Image + location -> road/obstruction masks
                      |
                      v
      Material + coverage + uncertainty
                      |
                      +--------------------+
                                           v
                        Combined location-aware view
                       Risk context + observed road status
                                           |
                                           v
                              Human review / response support
```

**Important integration boundary:** The proposed integration is an interface contract and workflow. Actual inter-module API wiring, database tables, and operational alerts must be verified in the repository; do not describe them as deployed unless tested.

### 5.3 Suggested API contract (proposed; adapt to existing routes)

**Module 1 provides:** `risk_id`, `area_geometry` or coordinates and spatial resolution, `risk_level`, `risk_score` if meaningful, `valid_at`, `generated_at`, `model_version`, and quality metadata.

**Module 3 provides:** `analysis_id`, `image_capture_time` if known, `location` with `source`/`accuracy_m`, `obstruction_detected`, `materials`, `road_coverage_percent`, `road_coverage_status`, `suspected_cause`, `road_condition`, `model_versions`, and `report_uri`.

The integration layer performs the spatial/time match and produces a **linked incident record**, not a single blended score unless such a score has been separately designed and validated.

### 5.4 Example joint interpretation

**Illustrative only:** Module 1 reports **Warning** for a nearby slope area; Module 3 receives a geolocated road photo and estimates **42% projected road coverage** by rock and soil debris. The joint view can show an elevated risk context plus photographic evidence of an affected road and recommend verification. Neither module alone proves when the obstruction happened, its geological cause, or whether all vehicle types can pass.

The modules may also disagree: **high risk + no visible obstruction** or **low risk + observed obstruction** are both possible and must remain visible to the user.

## 6. Datasets and data governance

Potential public sources include the following. **Availability, licence terms, image counts, masks, and actual classes must be rechecked before use**; these are not automatically approved or included in this repository.

| Candidate source | Potential contribution | Limitation to verify |
| --- | --- | --- |
| [LandslidePTIT](https://github.com/buiquocbao98/LandslidePTIT-Dataset) | Road/landslide imagery and annotations | Synthetic characteristics, licence, class coverage |
| [EPFL Road Anomaly](https://www.epfl.ch/labs/cvlab/data/road-anomaly/) | Unexpected road-obstacle masks | Material-specific labels may be absent |
| [IIIT-H Small Obstacle](https://small-obstacle-dataset.github.io/) | Road, obstacle and off-road pixels | Not a complete natural-disaster taxonomy |
| [RoadObstacle21 / RoadAnomaly21](https://segmentmeifyoucan.com/datasets) | Obstacle generalization benchmarks | Follow benchmark-use restrictions |
| [MobimaSkidRoads](https://radar.kit.edu/radar/en/dataset/glefEuwTUORWTWLs) | Forest-road obstacles / tree-related examples | Scene domain and annotations differ from mountain roads |
| [TerrainSense](https://data.mendeley.com/datasets/r6cmjrr6kv/3) | Outdoor obstacle recognition | Bounding boxes may not support pixel area |
| [India Driving Dataset](https://idd.insaan.iiit.ac.in/) | Indian road-scene generalization | Obstruction mask coverage may be limited |
| [Geofabrik Northeast India](https://download.geofabrik.de/asia/india/north-eastern-zone.html) | Geospatial road-network context | Road map, **not** labelled obstruction photographs |

Each dataset needs a provenance entry (original publisher, URL, verified licence, version, SHA-256, annotation format/classes, real/synthetic status, permitted use, and train/validation/test grouping). Do not convert bounding boxes into ground-truth masks or silently invent missing class annotations. Protect withheld test sets from leakage and report synthetic-to-real transfer separately.

## 7. Model families and research comparison

The project is intended to compare several **CNN-based** model families, subject to verified datasets and compute resources. The architecture list is an experimental plan, **not a claim that every model has been trained successfully**.

| Task | Candidate models | Appropriate measurements |
| --- | --- | --- |
| Road / semantic obstruction segmentation | U-Net, U-Net++, DeepLabV3+, FPN, PSPNet (CNN backbones) | Per-class IoU, mIoU, Dice, coverage error |
| Instance-level obstruction segmentation | YOLO-seg, Mask R-CNN | Mask AP50/AP50:95, recall, latency |
| Material / suspected-cause classification | EfficientNet-B0, ResNet-50, ConvNeXt-Tiny | Macro F1, per-class precision/recall, calibration |
| End-to-end report quality | Best validated combination | Coverage MAE in percentage points, material-label performance, reliability, latency |

**Architecture constraint:** No transformer-based vision models for this module. Do not replace the CNN experiments with SegFormer or ViT.

### Evaluation standards

- Use scene-/source-aware, leakage-controlled train/validation/test splits.
- Tune thresholds using **validation only**; preserve locked test sets for final evaluation.
- Record dataset versions, seeds, hyperparameters, checkpoints, hardware, and code revisions.
- Evaluate per class and on real-world natural-event images, not only on common road/pothole benchmarks.
- Report obstacle confusion, false positives, missing classes, and uncertainty.
- Distinguish **IoU**, **AP50**, **macro F1**, and **coverage MAE**; they are not interchangeable “accuracy” scores.
- Treat an 80% performance threshold as a **research objective**, not as a guaranteed result or a reason to alter test data.

A testing dashboard should support single-image inference, compatible-model side-by-side comparisons, labelled-dataset benchmarking, manual reference-mask annotation, a task-specific leaderboard, and exportable metrics. Annotation mode is a validation tool, not a dependency of normal one-click inference.

## 8. Example report

The following is **illustrative, not a real inference result**:

```json
{
  "analysis_id": "SANKET-DEMO-001",
  "obstruction_detected": true,
  "materials": ["rock", "loose_soil"],
  "projected_road_coverage_percent": 42.0,
  "road_coverage_status": "estimated",
  "suspected_cause": "landslide_or_rockfall_unconfirmed",
  "road_condition": "partially_obstructed",
  "passability": "requires_field_verification",
  "location": {
    "latitude": null,
    "longitude": null,
    "source": "unavailable",
    "accuracy_m": null
  },
  "module_1_context": {
    "matched_risk_level": "Warning",
    "match_status": "illustrative_only"
  },
  "limitations": ["image_plane_measurement", "cause_not_verified"]
}
```

The report should always include the model/checkpoint versions, analysis time, mask-availability flags, and links or file paths to its annotated output where available.

## 9. Implementation and local setup

A recommended organization is:

```text
SANKET/
├── module1-risk/                  # Existing Module 1 implementation (if separate)
├── module3-road-obstruction/
│   ├── backend/                   # FastAPI routes and analysis services
│   ├── frontend/                  # Upload, results, model comparison
│   ├── ml/                        # Model training, inference, evaluation
│   ├── data/                      # Raw/processed manifests (large files ignored)
│   ├── models/                    # Model registry / checkpoints
│   ├── experiments/               # Metrics, configs and plots
│   ├── reports/                   # Generated analyses
│   ├── tests/                     # Software and ML checks
│   ├── requirements.txt
│   ├── requirements-ml.txt
│   ├── .env.example
│   └── README.md
└── integration/                   # Optional risk-to-incident join and contracts
```

**This is a recommended layout, not a verified directory listing.** Map it to the actual repository before adopting it.

### Windows quick start (reference commands)

Run from the **actual Module 3 project root** where `requirements.txt` exists:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

These commands assume an `app.main:app` entry point. If the project uses `backend.app.main:app` or another path, substitute the correct module; use the existing README/scripts as the source of truth. Model experiments may require `requirements-ml.txt` and compatible checkpoint paths. Installing packages does **not** automatically download training datasets or model weights.

**Configuration:** Keep paths and runtime settings in `.env`/`.env.example`; do not commit secrets, private coordinates, uploaded images, or licence-restricted data. No Docker is required for the intended local development environment.

### Minimal configuration fields

```dotenv
APP_ENV=development
DATA_ROOT=./data
MODEL_ROOT=./models
REPORT_ROOT=./reports
INFERENCE_DEVICE=auto
DEFAULT_PIPELINE=auto
MAX_UPLOAD_MB=15
ENABLE_EXIF_GPS=true
ENABLE_BROWSER_GEOLOCATION=true
```

Actual variable names must match the implementation. A missing model checkpoint should return a clear `model_not_ready` error; it must not silently fabricate a successful prediction.

## 10. Testing, status, and honest limitations

Testing should cover image validation, no-road/empty-mask inputs, multi-obstacle masks, pixel-overlap calculations, GPS permission denial, EXIF/device coordinate conflicts, JSON/PDF exports, and model checkpoint loading. Model acceptance additionally requires **independent real-photo annotations**, per-class segmentation/classification metrics, and coverage percentage-point error against reference masks.

### Status based on the latest reported project audit

The latest agent reports shared during development indicate:

- Upload, location capture, PDF/JSON export, and parts of the model-comparison workflow passed software tests.
- Road segmentation showed strong IoU on some datasets (reported 89.9% RTK and 89.1% KITTI), but weaker generalization on DNIT (reported 65.2% IoU for a newer checkpoint).
- Generic obstacle predictions and automatic blockage percentage remained unreliable on real flood photographs; all 12 images in one audit returned `Unable to determine` for automatic coverage.
- Pothole and road-damage experiments did not establish reliable recognition of the required landslide/tree/rock/flood materials.
- A separate saved U-Net checkpoint was reported to estimate blockage with 4.65 percentage-point MAE on a small 38-image test; its dataset provenance and real-photo generalization were **not established** and must not be equated with validated real-world coverage estimates.

These are **reported experimental findings**, not results independently reproduced by this README. Check the repository's `MODEL_COMPARISON.md`, experiment files, and `TESTING.md` for the definitive checkpoint-specific evidence. Do not combine metrics from overwritten checkpoints or different datasets.

### Known challenges

- Missing high-quality masks for natural-event obstructions on real, diverse roads.
- Generalization from synthetic/city-road datasets to narrow Himalayan mountain roads.
- Correct road-footprint estimation when obstacles cover the roadway.
- Distinguishing observed debris from its underlying geological or weather cause.
- Estimating physical passability from a single uncalibrated photograph.
- Ensuring uploaded device GPS is not confused with photograph capture location.

## 11. Future work and contributions

Short-term priorities are (1) verifying usable licences/annotations, (2) building a balanced, independently labelled natural-obstruction test set, (3) improving road-footprint and obstacle masks together, (4) validating automatic coverage error on real photos, and (5) connecting Module 1 risk records to Module 3 reports through a tested spatial/temporal interface.

In the wider SANKET platform, Module 2 may provide satellite-confirmed landslide extent and Module 4 may provide crack-progression evidence. These additional signals can enrich an incident view **without conflating prediction, observation, and confirmation**.

**Overall contribution:** Module 1 helps answer **“Where is landslide risk elevated?”** Module 3 helps answer **“What does an available road photograph show, and how much of the visible roadway appears affected?”** Their integration can help prioritize field verification and emergency response assessment while retaining transparent uncertainty and source provenance.

---

**Project note:** This README describes the intended architecture, interfaces, research workflow, and reported development status. Check live source code and test artifacts before treating any proposed endpoint, model, dataset integration, or performance claim as verified functionality.
