"""
🚀 Digital Twin Spacecraft Navigation Simulator
================================================

Entry point — Initialises all three nodes and launches
the Mission Control dashboard.

Usage:
    python main.py              Launch the full simulator
    python main.py --generate   Generate training dataset (10,000 images)
    python main.py --train      Train the Star CNN
    python main.py --quick      Generate 500 images + train 10 epochs (fast test)
"""

import os
import sys
import argparse

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def check_model_exists():
    """Check if a trained model exists."""
    return os.path.exists(config.MODEL_SAVE_PATH)


def check_dataset_exists():
    """Check if training images have been generated."""
    labels_path = os.path.join(config.TRAINING_IMAGES_DIR, "labels.npy")
    return os.path.exists(labels_path)


def run_generate(num_images=None):
    """Generate synthetic training dataset."""
    from node_b_perception.dataset_generator import generate_dataset

    if num_images is None:
        num_images = config.NUM_TRAINING_IMAGES
    print(f"\n{'='*60}")
    print(f"  STEP 1: Generating {num_images} synthetic star images")
    print(f"{'='*60}\n")
    generate_dataset(num_images=num_images)


def run_train(epochs=None, batch_size=None):
    """Train the Star CNN."""
    from node_b_perception.train import train

    if epochs is None:
        epochs = config.NUM_EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    print(f"\n{'='*60}")
    print(f"  STEP 2: Training ResNet-18 Star CNN ({epochs} epochs)")
    print(f"{'='*60}\n")
    train(epochs=epochs, batch_size=batch_size)


def run_dashboard():
    """Launch the Mission Control dashboard."""
    print(f"\n{'='*60}")
    print(f"  🚀 LAUNCHING MISSION CONTROL DASHBOARD")
    print(f"{'='*60}\n")

    from ui.dashboard import launch_dashboard
    app, window = launch_dashboard()
    sys.exit(app.exec())


def main():
    parser = argparse.ArgumentParser(
        description="Digital Twin Spacecraft Navigation Simulator"
    )
    parser.add_argument("--generate", action="store_true",
                        help="Generate synthetic training dataset")
    parser.add_argument("--train", action="store_true",
                        help="Train the Star CNN")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 500 images + 10 epochs")
    parser.add_argument("--launch", action="store_true",
                        help="Launch dashboard (default if model exists)")
    parser.add_argument("-n", "--num-images", type=int, default=None,
                        help="Number of training images to generate")
    parser.add_argument("-e", "--epochs", type=int, default=None,
                        help="Number of training epochs")

    args = parser.parse_args()

    print(r"""
    ╔══════════════════════════════════════════════════════╗
    ║   🛰️  DIGITAL TWIN — Spacecraft Navigation Sim     ║
    ║   ─────────────────────────────────────────────     ║
    ║   Physical Engine  │  AI Star Sensor  │  Nav Ctrl   ║
    ╚══════════════════════════════════════════════════════╝
    """)
    print(f"  Device : {config.DEVICE}")
    print(f"  Model  : {'✓ Found' if check_model_exists() else '✗ Not trained'}")
    print(f"  Dataset: {'✓ Found' if check_dataset_exists() else '✗ Not generated'}")
    print()

    # Handle --quick mode
    if args.quick:
        run_generate(num_images=500)
        run_train(epochs=10, batch_size=32)
        run_dashboard()
        return

    # Handle individual steps
    if args.generate:
        run_generate(num_images=args.num_images)
        if not args.train and not args.launch:
            return

    if args.train:
        if not check_dataset_exists():
            print("[!] No dataset found. Generating first …")
            run_generate(num_images=args.num_images)
        run_train(epochs=args.epochs)
        if not args.launch:
            return

    # Default: launch dashboard
    if args.launch or (not args.generate and not args.train):
        if not check_model_exists():
            print("="*60)
            print("  ⚠  No trained model found!")
            print("  The AI Star Sensor will use random weights.")
            print()
            print("  To train the full model, run:")
            print("    python main.py --generate --train")
            print()
            print("  For a quick test (5 minutes), run:")
            print("    python main.py --quick")
            print("="*60)
            print()
            resp = input("  Launch anyway with untrained model? [y/N]: ")
            if resp.lower() != 'y':
                return

        run_dashboard()


if __name__ == "__main__":
    main()
