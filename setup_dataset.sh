#!/bin/bash
# Setup script for CS2 navigation training on server
# Downloads dataset, extracts archives, and prepares environment

set -e  # Exit on error

echo "=========================================="
echo "CS2 Navigation Dataset Setup"
echo "=========================================="

# Configuration
SERVER_URL="https://23rb2p37-8000.euw.devtunnels.ms"
DATASET_DIR="dataset"
RAW_SESSIONS_DIR="dataset/raw_sessions"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
print_info "Checking required tools..."
command -v wget >/dev/null 2>&1 || { print_error "wget is required but not installed. Aborting."; exit 1; }
command -v unzip >/dev/null 2>&1 || { print_error "unzip is required but not installed. Aborting."; exit 1; }

# Create directories
print_info "Creating directories..."
mkdir -p "$DATASET_DIR"
mkdir -p "$RAW_SESSIONS_DIR"
mkdir -p "checkpoints"
mkdir -p "logs"

# Download dataset.zip
print_info "Downloading dataset.zip..."
if [ -f "dataset.zip" ]; then
    print_warn "dataset.zip already exists, skipping download"
else
    wget --no-check-certificate -O dataset.zip "${SERVER_URL}/dataset.zip" || {
        print_error "Failed to download dataset.zip"
        exit 1
    }
    print_info "dataset.zip downloaded successfully"
fi

# Download raw_sessions.zip
print_info "Downloading raw_sessions.zip..."
if [ -f "raw_sessions.zip" ]; then
    print_warn "raw_sessions.zip already exists, skipping download"
else
    wget --no-check-certificate -O raw_sessions.zip "${SERVER_URL}/raw_sessions.zip" || {
        print_error "Failed to download raw_sessions.zip"
        exit 1
    }
    print_info "raw_sessions.zip downloaded successfully"
fi

# Extract dataset.zip
print_info "Extracting dataset.zip..."
if [ -d "$DATASET_DIR/manifests" ]; then
    print_warn "Dataset already extracted, skipping"
else
    unzip -q dataset.zip -d . || {
        print_error "Failed to extract dataset.zip"
        exit 1
    }
    print_info "dataset.zip extracted successfully"
fi

# Extract raw_sessions.zip
print_info "Extracting raw_sessions.zip..."
if [ -d "$RAW_SESSIONS_DIR/session_0001" ]; then
    print_warn "Raw sessions already extracted, skipping"
else
    unzip -q raw_sessions.zip -d "$DATASET_DIR" || {
        print_error "Failed to extract raw_sessions.zip"
        exit 1
    }
    print_info "raw_sessions.zip extracted successfully"
fi

# Verify structure
print_info "Verifying dataset structure..."

# Check for manifests
if [ ! -f "$DATASET_DIR/manifests/train_manifest.jsonl" ]; then
    print_error "train_manifest.jsonl not found!"
    exit 1
fi

if [ ! -f "$DATASET_DIR/manifests/val_manifest.jsonl" ]; then
    print_error "val_manifest.jsonl not found!"
    exit 1
fi

# Count sessions
SESSION_COUNT=$(find "$RAW_SESSIONS_DIR" -maxdepth 1 -type d -name "session_*" | wc -l)
print_info "Found $SESSION_COUNT sessions in raw_sessions/"

# Count frames in first session (if exists)
if [ -d "$RAW_SESSIONS_DIR/session_0001/frames" ]; then
    FRAME_COUNT=$(find "$RAW_SESSIONS_DIR/session_0001/frames" -name "*.jpg" | wc -l)
    print_info "Session 0001 has $FRAME_COUNT frames"
fi

# Count manifest entries
TRAIN_COUNT=$(wc -l < "$DATASET_DIR/manifests/train_manifest.jsonl")
VAL_COUNT=$(wc -l < "$DATASET_DIR/manifests/val_manifest.jsonl")
print_info "Train samples: $TRAIN_COUNT"
print_info "Val samples: $VAL_COUNT"

# Check Python environment
print_info "Checking Python environment..."
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    print_error "Python not found!"
    exit 1
fi

print_info "Python command: $PYTHON_CMD"
$PYTHON_CMD --version

# Check if PyTorch is installed
print_info "Checking PyTorch installation..."
$PYTHON_CMD -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null || {
    print_warn "PyTorch not installed. Install with:"
    echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
}

# Check CUDA
print_info "Checking CUDA availability..."
$PYTHON_CMD -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null || {
    print_warn "Could not check CUDA (PyTorch not installed)"
}

# Optional: Clean up zip files
read -p "Delete downloaded zip files to save space? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Removing zip files..."
    rm -f dataset.zip raw_sessions.zip
    print_info "Zip files removed"
fi

echo ""
echo "=========================================="
print_info "Setup complete!"
echo "=========================================="
echo ""
echo "Dataset structure:"
echo "  dataset/"
echo "    ├── manifests/"
echo "    │   ├── train_manifest.jsonl ($TRAIN_COUNT samples)"
echo "    │   └── val_manifest.jsonl ($VAL_COUNT samples)"
echo "    └── raw_sessions/"
echo "        ├── session_0001/"
echo "        ├── session_0002/"
echo "        └── ..."
echo ""
echo "Next steps:"
echo "  1. Install dependencies:"
echo "     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
echo "     pip install pillow numpy pandas"
echo ""
echo "  2. Start training (2 GPUs):"
echo "     python -m training.train \\"
echo "       --train-manifest dataset/manifests/train_manifest.jsonl \\"
echo "       --val-manifest dataset/manifests/val_manifest.jsonl \\"
echo "       --dataset-root dataset \\"
echo "       --backbone resnet18 \\"
echo "       --batch-size 64 \\"
echo "       --epochs 100 \\"
echo "       --world-size 2"
echo ""
echo "  3. Monitor training:"
echo "     tail -f logs/training.log"
echo ""
