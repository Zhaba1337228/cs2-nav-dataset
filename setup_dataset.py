"""
Setup script for CS2 navigation training on server.
Downloads dataset, extracts archives, and prepares environment.
Python version for cross-platform compatibility.
"""

import os
import sys
import zipfile
import urllib.request
import shutil
from pathlib import Path
from typing import Optional


# Configuration
SERVER_URL = "https://23rb2p37-8000.euw.devtunnels.ms"
DATASET_DIR = Path("dataset")
RAW_SESSIONS_DIR = DATASET_DIR / "raw_sessions"


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def print_info(msg: str) -> None:
    """Print info message."""
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {msg}")


def print_warn(msg: str) -> None:
    """Print warning message."""
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}")


def print_error(msg: str) -> None:
    """Print error message."""
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")


def print_header(msg: str) -> None:
    """Print header message."""
    print(f"\n{Colors.BLUE}{'='*50}{Colors.NC}")
    print(f"{Colors.BLUE}{msg}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*50}{Colors.NC}\n")


def download_file(url: str, output_path: Path, desc: str) -> bool:
    """Download file with progress bar."""
    try:
        print_info(f"Downloading {desc}...")

        # Create a simple progress callback
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                bar_length = 40
                filled = int(bar_length * percent / 100)
                bar = '█' * filled + '-' * (bar_length - filled)
                size_mb = total_size / (1024 * 1024)
                downloaded_mb = downloaded / (1024 * 1024)
                print(f"\r  [{bar}] {percent:.1f}% ({downloaded_mb:.1f}/{size_mb:.1f} MB)", end='')

        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        print()  # New line after progress bar
        print_info(f"{desc} downloaded successfully")
        return True
    except Exception as e:
        print_error(f"Failed to download {desc}: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path, desc: str) -> bool:
    """Extract zip file."""
    try:
        print_info(f"Extracting {desc}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total size for progress
            total_files = len(zip_ref.namelist())
            for i, member in enumerate(zip_ref.namelist()):
                zip_ref.extract(member, extract_to)
                if i % 100 == 0:  # Update every 100 files
                    percent = (i + 1) * 100 / total_files
                    print(f"\r  Extracting... {percent:.1f}% ({i+1}/{total_files} files)", end='')
            print()  # New line
        print_info(f"{desc} extracted successfully")
        return True
    except Exception as e:
        print_error(f"Failed to extract {desc}: {e}")
        return False


def count_files(directory: Path, pattern: str = "*") -> int:
    """Count files matching pattern in directory."""
    try:
        return len(list(directory.glob(pattern)))
    except:
        return 0


def verify_dataset() -> bool:
    """Verify dataset structure."""
    print_info("Verifying dataset structure...")

    # Check manifests
    train_manifest = DATASET_DIR / "manifests" / "train_manifest.jsonl"
    val_manifest = DATASET_DIR / "manifests" / "val_manifest.jsonl"

    if not train_manifest.exists():
        print_error("train_manifest.jsonl not found!")
        return False

    if not val_manifest.exists():
        print_error("val_manifest.jsonl not found!")
        return False

    # Count manifest entries
    with open(train_manifest, 'r') as f:
        train_count = sum(1 for line in f if line.strip())
    with open(val_manifest, 'r') as f:
        val_count = sum(1 for line in f if line.strip())

    print_info(f"Train samples: {train_count}")
    print_info(f"Val samples: {val_count}")

    # Count sessions
    if RAW_SESSIONS_DIR.exists():
        sessions = list(RAW_SESSIONS_DIR.glob("session_*"))
        print_info(f"Found {len(sessions)} sessions in raw_sessions/")

        # Count frames in first session
        first_session = RAW_SESSIONS_DIR / "session_0001"
        if first_session.exists():
            frames_dir = first_session / "frames"
            if frames_dir.exists():
                frame_count = count_files(frames_dir, "*.jpg")
                print_info(f"Session 0001 has {frame_count} frames")

    return True


def check_python_env() -> None:
    """Check Python environment and dependencies."""
    print_info("Checking Python environment...")
    print_info(f"Python version: {sys.version}")

    # Check PyTorch
    try:
        import torch
        print_info(f"PyTorch version: {torch.__version__}")
        print_info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print_info(f"CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print_info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print_warn("PyTorch not installed. Install with:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

    # Check other dependencies
    missing = []
    for package in ["PIL", "numpy", "pandas"]:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print_warn(f"Missing packages: {', '.join(missing)}")
        print("  Install with: pip install pillow numpy pandas")


def main():
    """Main setup function."""
    print_header("CS2 Navigation Dataset Setup")

    # Create directories
    print_info("Creating directories...")
    DATASET_DIR.mkdir(exist_ok=True)
    RAW_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Download dataset.zip
    dataset_zip = Path("dataset.zip")
    if dataset_zip.exists():
        print_warn("dataset.zip already exists, skipping download")
    else:
        if not download_file(f"{SERVER_URL}/dataset.zip", dataset_zip, "dataset.zip"):
            sys.exit(1)

    # Download raw_sessions.zip
    raw_sessions_zip = Path("raw_sessions.zip")
    if raw_sessions_zip.exists():
        print_warn("raw_sessions.zip already exists, skipping download")
    else:
        if not download_file(f"{SERVER_URL}/raw_sessions.zip", raw_sessions_zip, "raw_sessions.zip"):
            sys.exit(1)

    # Extract dataset.zip
    if (DATASET_DIR / "manifests").exists():
        print_warn("Dataset already extracted, skipping")
    else:
        if not extract_zip(dataset_zip, Path("."), "dataset.zip"):
            sys.exit(1)

    # Extract raw_sessions.zip
    if (RAW_SESSIONS_DIR / "session_0001").exists():
        print_warn("Raw sessions already extracted, skipping")
    else:
        if not extract_zip(raw_sessions_zip, DATASET_DIR, "raw_sessions.zip"):
            sys.exit(1)

    # Verify dataset
    if not verify_dataset():
        sys.exit(1)

    # Check Python environment
    check_python_env()

    # Ask to clean up
    print()
    response = input("Delete downloaded zip files to save space? (y/n): ").strip().lower()
    if response == 'y':
        print_info("Removing zip files...")
        if dataset_zip.exists():
            dataset_zip.unlink()
        if raw_sessions_zip.exists():
            raw_sessions_zip.unlink()
        print_info("Zip files removed")

    # Print summary
    print_header("Setup Complete!")

    print("\nDataset structure:")
    print("  dataset/")
    print("    ├── manifests/")
    print("    │   ├── train_manifest.jsonl")
    print("    │   └── val_manifest.jsonl")
    print("    └── raw_sessions/")
    print("        ├── session_0001/")
    print("        ├── session_0002/")
    print("        └── ...")

    print("\nNext steps:")
    print("  1. Install dependencies:")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("     pip install pillow numpy pandas")
    print()
    print("  2. Start training (2 GPUs):")
    print("     python -m training.train \\")
    print("       --train-manifest dataset/manifests/train_manifest.jsonl \\")
    print("       --val-manifest dataset/manifests/val_manifest.jsonl \\")
    print("       --dataset-root dataset \\")
    print("       --backbone resnet18 \\")
    print("       --batch-size 64 \\")
    print("       --epochs 100 \\")
    print("       --world-size 2")
    print()
    print("  3. Monitor training:")
    print("     Check checkpoints/ directory for saved models")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_error("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
