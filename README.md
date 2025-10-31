# CNN-Pooling

A Convolutional Neural Network (CNN) implementation in Rust for MNIST digit classification, with support for GPU acceleration via CUDA.

## Overview

This project trains a CNN to classify handwritten digits from the MNIST dataset, achieving approximately **96-99% accuracy**. The implementation uses PyTorch's Rust bindings (`tch-rs`) and supports both CPU and GPU execution.

### Features

- **GPU Acceleration**: Full CUDA support for training on NVIDIA GPUs
- **Automatic Data Download**: MNIST dataset is downloaded automatically on first run
- **Configurable Architecture**: Easy to modify pooling strategies and network parameters
- **Cross-Platform**: Works on Windows, Linux, and macOS (CPU mode)

### CNN Architecture

```
Input (28x28 grayscale images)
    ↓
Conv2D (1→32 channels, 5x5 kernel)
    ↓
Average Pooling (2x2)
    ↓
Conv2D (32→64 channels, 5x5 kernel)
    ↓
Average Pooling (2x2)
    ↓
Fully Connected (1024→1024) + ReLU + Dropout(50%)
    ↓
Fully Connected (1024→10)
    ↓
Output (10 classes)
```

## Quick Start

### Prerequisites

- **Rust** (1.70+): Install from [rustup.rs](https://rustup.rs/)
- **CUDA Toolkit** (Optional, for GPU support): CUDA 12.1+

### CPU-Only Setup

```bash
git clone <repository-url>
cd CNN-Pooling
cargo run --release
```

That's it! The program will automatically download MNIST data and start training.

### GPU/CUDA Setup

For detailed GPU setup instructions, see **[SETUP.md](SETUP.md)**.

Quick steps:
1. Download CUDA-enabled LibTorch
2. Extract to `libtorch-cuda/libtorch/`
3. Run the convenience script:
   ```bash
   ./run-cuda.sh
   ```

## Performance

| Setup | Training Time (10 epochs) | Accuracy |
|-------|--------------------------|----------|
| CPU (Ryzen/Intel i7) | ~2-5 minutes | 96-99% |
| GPU (RTX 4060 Ti) | ~30-60 seconds | 96-99% |

GPU acceleration provides **3-10x speedup** depending on your hardware.

## Usage

### Basic Training

```bash
cargo run --release
```

### With GPU Acceleration

```bash
source setup-cuda.sh
cargo run --release
```

Or use the convenience script:
```bash
./run-cuda.sh
```

### Expected Output

```
File data/train-images-idx3-ubyte already exists, skipping download.
File data/train-labels-idx1-ubyte already exists, skipping download.
File data/t10k-images-idx3-ubyte already exists, skipping download.
File data/t10k-labels-idx1-ubyte already exists, skipping download.
Epoch:    1, Test Accuracy: 87.72%
Epoch:    2, Test Accuracy: 90.76%
Epoch:    3, Test Accuracy: 92.47%
...
Epoch:   10, Test Accuracy: 96.62%
```

## Configuration

### Training Parameters

Edit `src/main.rs` to customize:

- **Number of epochs**: Line 105 - `for epoch in 1..=10`
- **Batch size**: Line 107 - `.split(256, 0)`
- **Learning rate**: Line 96 - `build(&vs, 1e-4)`
- **Pooling method**: Lines 77, 79 - `.avg_pool2d_default(2)`

### Switching Pooling Strategies

Change from average pooling to max pooling:

```rust
// From:
.avg_pool2d_default(2)

// To:
.max_pool2d_default(2, 2, 0, 1, false)
```

## Project Structure

```
CNN-Pooling/
├── src/
│   └── main.rs          # Main CNN implementation
├── build.rs             # Build configuration for CUDA
├── Cargo.toml           # Rust dependencies
├── data/                # MNIST dataset (auto-downloaded)
├── libtorch-cuda/       # CUDA-enabled LibTorch (manual setup)
├── target/              # Build artifacts
├── setup-cuda.sh        # CUDA environment setup script
├── run-cuda.sh          # Convenience script for GPU execution
├── SETUP.md             # Detailed setup guide
└── README.md            # This file
```

## Branches

- **`master`**: Main development branch
- **`build-fix`**: CUDA build fixes and current stable implementation
- **`brokek`**: Experimental branch exploring flexible pooling strategies

## Dependencies

Main dependencies (see `Cargo.toml` for full list):

- **tch** (0.22.0): PyTorch Rust bindings
- **tokio**: Async runtime for dataset downloads
- **reqwest**: HTTP client
- **anyhow**: Error handling

## Troubleshooting

### Common Issues

1. **"PyTorch version mismatch"**
   - Solution: Set `export LIBTORCH_BYPASS_VERSION_CHECK=1`

2. **"STATUS_ENTRYPOINT_NOT_FOUND" (Windows)**
   - Solution: Copy DLLs with `cp libtorch-cuda/libtorch/lib/*.dll target/release/`

3. **"CUDA out of memory"**
   - Solution: Reduce batch size from 256 to 128 or 64

For more troubleshooting tips, see **[SETUP.md](SETUP.md#troubleshooting)**.

## Development

### Building

```bash
cargo build --release
```

### Running Tests

```bash
cargo test
```

### Clean Build

```bash
cargo clean
cargo build --release
```

## Technical Details

### Device Selection

The program automatically detects and uses CUDA if available:

```rust
let vs = nn::VarStore::new(Device::cuda_if_available());
```

### Data Pipeline

1. Download MNIST dataset (if not present)
2. Load and normalize images (divide by 255.0)
3. Move tensors to GPU (if available)
4. Train with batch size 256
5. Evaluate on test set after each epoch

### Memory Management

- Training images: ~60,000 × 784 bytes ≈ 47 MB
- Test images: ~10,000 × 784 bytes ≈ 8 MB
- Model parameters: ~2.5 million weights ≈ 10 MB
- Total GPU memory usage: ~200-400 MB

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Data augmentation
- [ ] Learning rate scheduling
- [ ] More pooling strategies
- [ ] Different architectures (ResNet, etc.)
- [ ] Multi-GPU support
- [ ] TensorBoard integration

## License

[Add your license here]

## Acknowledgments

- MNIST dataset: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- tch-rs: Laurent Mazare
- PyTorch: Facebook AI Research

## Resources

- **Setup Guide**: [SETUP.md](SETUP.md)
- **tch-rs**: https://github.com/LaurentMazare/tch-rs
- **PyTorch**: https://pytorch.org/
- **MNIST**: http://yann.lecun.com/exdb/mnist/
