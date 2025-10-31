# CNN-Pooling Setup Guide

This guide will walk you through setting up the CNN-Pooling project after cloning the repository.

## Prerequisites

- **Rust**: Install from [rustup.rs](https://rustup.rs/)
- **Git**: For cloning the repository
- **CUDA Toolkit** (Optional): For GPU acceleration
  - Supported CUDA versions: 12.1+
  - NVIDIA GPU with compute capability 3.5 or higher

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd CNN-Pooling
```

### 2. Choose Your Setup

You have two options:

#### Option A: CPU-Only Setup (Easiest)

The project will automatically download a CPU-only version of LibTorch.

```bash
cargo run --release
```

That's it! The first build will take some time as it downloads dependencies.

#### Option B: GPU/CUDA Setup (Recommended for better performance)

Follow the detailed instructions below.

---

## Detailed GPU/CUDA Setup

### Step 1: Check CUDA Availability

Verify your GPU and CUDA version:

```bash
nvidia-smi
```

You should see your GPU information and CUDA version. Note the CUDA version (e.g., 12.9, 12.4, etc.).

### Step 2: Download CUDA-Enabled LibTorch

#### Windows

1. Visit the [PyTorch Download Page](https://pytorch.org/get-started/locally/)

2. Or download directly for your CUDA version:
   - **CUDA 12.4**: https://download.pytorch.org/libtorch/cu124/libtorch-win-shared-with-deps-2.9.0%2Bcu124.zip
   - **CUDA 13.0**: https://download.pytorch.org/libtorch/cu130/libtorch-win-shared-with-deps-2.9.0%2Bcu130.zip

3. Create a directory for LibTorch:
   ```bash
   mkdir -p libtorch-cuda
   ```

4. Download the zip file and place it in the `libtorch-cuda/` directory

5. Extract the archive:
   ```bash
   cd libtorch-cuda
   unzip libtorch-win-shared-with-deps-*.zip
   cd ..
   ```

#### Linux

```bash
# Create directory
mkdir -p libtorch-cuda
cd libtorch-cuda

# Download for CUDA 12.4
wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.9.0%2Bcu124.zip

# Extract
unzip libtorch-cxx11-abi-shared-with-deps-*.zip
cd ..
```

#### macOS

CUDA is not available on macOS. Use the CPU-only setup or consider using MPS (Metal Performance Shaders) if available.

### Step 3: Configure Environment Variables

#### Windows (Git Bash/MINGW64)

Create a file `setup-cuda.sh`:

```bash
#!/bin/bash
export LIBTORCH=G:/Code/CNN-Pooling/libtorch-cuda/libtorch
export LIBTORCH_BYPASS_VERSION_CHECK=1
```

**Note**: Replace `G:/Code/CNN-Pooling` with your actual project path.

Load it before building:
```bash
source setup-cuda.sh
```

#### Linux/macOS

Create a file `setup-cuda.sh`:

```bash
#!/bin/bash
export LIBTORCH=/path/to/CNN-Pooling/libtorch-cuda/libtorch
export LIBTORCH_BYPASS_VERSION_CHECK=1
```

Make it executable and source it:
```bash
chmod +x setup-cuda.sh
source setup-cuda.sh
```

### Step 4: Build the Project

```bash
cargo build --release
```

The first build will take several minutes as it compiles all dependencies.

### Step 5: Copy DLLs (Windows Only)

Windows needs the LibTorch DLLs to be accessible at runtime:

```bash
cp libtorch-cuda/libtorch/lib/*.dll target/release/
```

### Step 6: Run the Project

#### Windows (Git Bash)
```bash
source setup-cuda.sh
./target/release/GitTrial.exe
```

#### Linux
```bash
source setup-cuda.sh
cargo run --release
```

---

## Expected Output

When running successfully, you should see:

```
File data/train-images-idx3-ubyte already exists, skipping download.
File data/train-labels-idx1-ubyte already exists, skipping download.
File data/t10k-images-idx3-ubyte already exists, skipping download.
File data/t10k-labels-idx1-ubyte already exists, skipping download.
Epoch:    1, Test Accuracy: 87.72%
Epoch:    2, Test Accuracy: 90.76%
Epoch:    3, Test Accuracy: 92.47%
Epoch:    4, Test Accuracy: 93.55%
Epoch:    5, Test Accuracy: 94.49%
Epoch:    6, Test Accuracy: 95.12%
Epoch:    7, Test Accuracy: 95.66%
Epoch:    8, Test Accuracy: 96.22%
Epoch:    9, Test Accuracy: 96.47%
Epoch:   10, Test Accuracy: 96.62%
```

The model should achieve approximately **96-99% accuracy** on the MNIST test set.

---

## Verify GPU Usage

### Check GPU Memory Usage

While the program is running (or immediately after), check GPU utilization:

```bash
nvidia-smi
```

You should see increased GPU memory usage (~200-400 MB) and the `GitTrial.exe` process listed.

### Performance Comparison

- **CPU-only**: ~2-5 minutes for 10 epochs
- **GPU (CUDA)**: ~30-60 seconds for 10 epochs

GPU acceleration provides significant speedup, especially with larger models or datasets.

---

## Troubleshooting

### Issue: "PyTorch version mismatch"

**Error**: `this tch version expects PyTorch X.X.X, got Y.Y.Y`

**Solution**: Set the bypass environment variable:
```bash
export LIBTORCH_BYPASS_VERSION_CHECK=1
```

### Issue: "STATUS_ENTRYPOINT_NOT_FOUND" (Windows)

**Error**: `exit code: 0xc0000139`

**Solution**: Copy LibTorch DLLs to the executable directory:
```bash
cp libtorch-cuda/libtorch/lib/*.dll target/release/
```

### Issue: "Input type (CPUFloatType) and weight type (CUDAFloatType) mismatch"

**Error**: Tensor device mismatch

**Solution**: This is already fixed in the current code. Ensure you're using the latest version from the repository. The data tensors are moved to GPU with `.to(vs.device())`.

### Issue: Build fails with CUDA linking errors

**Error**: `unresolved external symbol warp_size`

**Solution**: The build script (`build.rs`) handles this automatically. If issues persist:
1. Ensure `LIBTORCH` environment variable is set correctly
2. Clean and rebuild: `cargo clean && cargo build --release`

### Issue: Download fails with 403 Forbidden

**Solution**: Download LibTorch manually from the PyTorch website:
1. Go to https://pytorch.org/get-started/locally/
2. Select your platform and CUDA version
3. Download the C++/LibTorch distribution
4. Extract to `libtorch-cuda/libtorch/`

### Issue: Out of GPU memory

**Error**: `CUDA out of memory`

**Solution**:
1. Reduce batch size in `src/main.rs` (line 107, change `256` to `128` or `64`)
2. Close other GPU-intensive applications
3. Use CPU-only mode if GPU has insufficient memory

---

## Project Structure

```
CNN-Pooling/
├── src/
│   └── main.rs              # Main CNN implementation
├── build.rs                 # Build configuration for CUDA
├── Cargo.toml               # Rust dependencies
├── data/                    # MNIST dataset (auto-downloaded)
├── libtorch-cuda/           # CUDA-enabled LibTorch (manual setup)
│   └── libtorch/
│       ├── lib/
│       ├── include/
│       └── share/
├── target/                  # Build artifacts
└── SETUP.md                 # This file
```

---

## Configuration Options

### Changing Training Parameters

Edit `src/main.rs`:

- **Epochs**: Line 105 - `for epoch in 1..=10` (change `10` to desired number)
- **Batch Size**: Line 107 - `train_images.split(256, 0)` (change `256`)
- **Learning Rate**: Line 96 - `build(&vs, 1e-4)` (change `1e-4`)

### Using Different Pooling Methods

The project currently uses **average pooling** (line 77, 79). To use **max pooling**:

```rust
// Change from:
.avg_pool2d_default(2)

// To:
.max_pool2d_default(2, 2, 0, 1, false)
```

---

## Clean Build

If you encounter persistent issues, perform a clean build:

```bash
# Remove all build artifacts
cargo clean

# Remove downloaded data (will be re-downloaded)
rm -rf data/

# Rebuild
source setup-cuda.sh  # If using CUDA
cargo build --release
```

---

## Additional Resources

- **tch-rs Documentation**: https://github.com/LaurentMazare/tch-rs
- **PyTorch LibTorch**: https://pytorch.org/cppdocs/
- **MNIST Dataset**: http://yann.lecun.com/exdb/mnist/
- **Rust Book**: https://doc.rust-lang.org/book/

---

## Support

For issues specific to this project, please check:
1. This setup guide
2. The repository's issue tracker
3. The troubleshooting section above

For general PyTorch or CUDA issues, consult the official PyTorch documentation.
