"""
CLI Demonstration: Predict Road Blockage and Obstacles
Demonstrates prediction on single images or directories and generates exact natural language output.
"""

import os
import sys
import argparse
from road_blockage.detector import RoadBlockageDetector
from road_blockage.utils.visualization import generate_scenario_test_images


def main():
    parser = argparse.ArgumentParser(description="Predict Road Blockage and Obstacles")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--model-path", type=str, default="saved_models/best_road_blockage_model.pt", help="Path to model checkpoint")
    parser.add_argument("--demo-scenarios", action="store_true", help="Run through all standard test scenarios")
    args = parser.parse_args()

    # Ensure model weights exist
    if not os.path.exists(args.model_path):
        print(f"[!] Model checkpoint not found at {args.model_path}.")
        print("[*] Running benchmark training first to train and save models...")
        from road_blockage.evaluation.benchmark import run_benchmark_suite
        run_benchmark_suite(save_dir="saved_models", num_samples=250, epochs=8)

    print(f"[*] Loading Road Blockage Detector from {args.model_path}...")
    detector = RoadBlockageDetector.load(args.model_path)
    print(f"[✓] Model loaded: {detector.model_name} (Pure CNN architecture)")

    # Prepare demo images
    sample_images = generate_scenario_test_images("sample_data")

    if args.image:
        print(f"\nEvaluating: {args.image}")
        result = detector.predict(args.image)
        print(f"\nResult:")
        print(f"  Summary:     {result.summary}")
        print(f"  Blockage:    {result.blockage_percentage}%")
        print(f"  Severity:    {result.severity}")
        print(f"  Obstacles:   {result.obstacles}")
        print(f"  Latency:     {result.inference_time_ms} ms")
        print(f"  JSON output:\n{result.to_json(indent=4)}")
    else:
        print("\n" + "=" * 70)
        print("RUNNING STANDARD TEST SCENARIOS")
        print("=" * 70)
        for name, path in sample_images.items():
            result = detector.predict(path)
            print(f"\nInput: {name}")
            print(f"  --> {result.summary}")
            print(f"      [Blockage: {result.blockage_percentage}%, Severity: {result.severity}, Latency: {result.inference_time_ms}ms]")


if __name__ == "__main__":
    main()
