"""
CLI Entry Point: Run Comprehensive Model Benchmark & Model Selection
Compares EfficientNet, U-Net, YOLOv8, YOLOv5, ResNet (Pure CNNs strictly).
Saves best model and evaluation report.
"""

import os
import sys
import argparse
from road_blockage.evaluation.benchmark import run_benchmark_suite
from road_blockage.utils.visualization import generate_scenario_test_images


def main():
    parser = argparse.ArgumentParser(description="Road Blockage CNN Benchmark and Training")
    parser.add_argument("--samples", type=int, default=100, help="Number of dataset samples")
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs per model")
    parser.add_argument("--img-size", type=int, default=128, help="Resolution (H=W) for benchmarking")
    parser.add_argument("--save-dir", type=str, default="saved_models", help="Directory to save model weights")
    args = parser.parse_args()

    # Generate sample scenario images for testing
    print("[*] Generating scenario test images in sample_data/...", flush=True)
    generate_scenario_test_images("sample_data")

    # Run benchmark suite
    results = run_benchmark_suite(
        save_dir=args.save_dir,
        num_samples=args.samples,
        epochs=args.epochs,
        img_size=args.img_size,
        device="cpu"
    )

    print("\n" + "=" * 70, flush=True)
    print("BENCHMARK SUMMARY & MODEL SELECTION", flush=True)
    print("=" * 70, flush=True)
    for model_name, m in results["results"].items():
        print(f"Model: {model_name:<18} | Acc: {m['obstacle_accuracy']:>5}% | F1: {m['f1_score']:>5}% | "
              f"TolAcc: {m['tolerance_accuracy_10pct']:>5}% | MAE: {m['blockage_mae']:>5}% | "
              f"R²: {m['blockage_r2']:>5} | FPS: {m['fps']:>5}", flush=True)

    print(f"\n[>>>] Selected Best Model: {results['best_model']} (Score: {results['best_score']})", flush=True)
    print(f"[>>>] Checkpoint saved at: {results['weights_path']}", flush=True)


if __name__ == "__main__":
    main()
