#!/bin/bash
# CUDA Environment Setup for CNN-Pooling
# Source this file before building or running the project with GPU support
#
# Usage:
#   source setup-cuda.sh
#   cargo run --release

# Set the LibTorch path (adjust if your project is in a different location)
export LIBTORCH=G:/Code/CNN-Pooling/libtorch-cuda/libtorch

# Bypass version check (needed if LibTorch version doesn't exactly match tch crate)
export LIBTORCH_BYPASS_VERSION_CHECK=1

echo "CUDA environment configured:"
echo "  LIBTORCH: $LIBTORCH"
echo "  LIBTORCH_BYPASS_VERSION_CHECK: $LIBTORCH_BYPASS_VERSION_CHECK"
echo ""
echo "You can now run: cargo build --release"
