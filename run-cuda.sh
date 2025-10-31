#!/bin/bash
# Convenience script to run the CNN project with CUDA support
#
# Usage:
#   ./run-cuda.sh

# Change to project directory
cd "$(dirname "$0")"

# Setup CUDA environment
source setup-cuda.sh

# Copy DLLs to target directory (Windows only, safe to run on any platform)
if [ -d "libtorch-cuda/libtorch/lib" ] && [ -d "target/release" ]; then
    echo "Copying LibTorch DLLs to executable directory..."
    cp libtorch-cuda/libtorch/lib/*.dll target/release/ 2>/dev/null || true
fi

# Build and run
echo ""
echo "Building and running with GPU acceleration..."
cargo run --release
