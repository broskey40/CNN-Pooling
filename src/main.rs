use anyhow::{Result, Context};
use flate2::read::GzDecoder;
use std::{fs, fs::File, io::copy, path::Path};
use std::fmt::{Debug, Formatter};
use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Device, Tensor, vision};
use reqwest;

// URLs for the MNIST dataset files
const MNIST_URLS: &[(&str, &str)] = &[
    ("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz", "data/train-images-idx3-ubyte"),
    ("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz", "data/train-labels-idx1-ubyte"),
    ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz", "data/t10k-images-idx3-ubyte"),
    ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz", "data/t10k-labels-idx1-ubyte"),
];

// Function to download and extract MNIST dataset files if they don't exist.
async fn download_mnist() -> Result<()> {
    fs::create_dir_all("data").context("Failed to create data directory")?;

    for &(url, file_path) in MNIST_URLS {
        if !Path::new(file_path).exists() {
            println!("Downloading {}...", url);
            let response = reqwest::get(url).await?;

            // Check if the response is a valid GZIP file
            if response.headers().get("content-type").map(|v| v != "application/x-gzip").unwrap_or(true) {
                return Err(anyhow::anyhow!("Invalid content type for {}: {:?}", url, response.headers().get("content-type")));
            }

            // Attempt to extract the GZIP file
            let bytes = response.bytes().await?;
            let mut gz = GzDecoder::new(bytes.as_ref());
            let mut out_file = File::create(file_path).context("Failed to create MNIST file")?;
            copy(&mut gz, &mut out_file).context("Failed to extract MNIST file")?;
            println!("Downloaded and extracted to {}", file_path);
        } else {
            println!("File {} already exists, skipping download.", file_path);
        }
    }
    Ok(())
}

// Main function to download MNIST data and run the CNN model.
#[tokio::main]
async fn main() -> Result<()> {
    // Ensure the MNIST dataset is downloaded and extracted.
    download_mnist().await?;

    // Run the CNN model training
    run_conv()
}

// CNN Model - Should reach around 99% accuracy.
#[derive(Debug, Clone, Copy)]
enum PoolingMethod {
    Max,
    Avg,
}

struct Net {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
    pooling: PoolingMethod,
}

impl Net {
    // Initializes a new CNN model with layers defined in the `Net` structure.
    fn new(vs: &nn::Path, pooling: PoolingMethod) -> Net {
        let conv1 = nn::conv2d(vs, 1, 32, 5, Default::default());
        let conv2 = nn::conv2d(vs, 32, 64, 5, Default::default());
        let fc1 = nn::linear(vs, 1024, 1024, Default::default());
        let fc2 = nn::linear(vs, 1024, 10, Default::default());
        Net {
            conv1,
            conv2,
            fc1,
            fc2,
            pooling,
        }
    }
}

trait TensorExt {
    fn apply_pooling(&self, pooling: PoolingMethod, ksize: i64) -> Tensor;
}

impl TensorExt for Tensor {
    fn apply_pooling(&self, pooling: PoolingMethod, ksize: i64) -> Tensor {
        match pooling {
            PoolingMethod::Max => self.max_pool2d_default(ksize),
            PoolingMethod::Avg => self.avg_pool2d_default(ksize),
        }
    }
}

impl Debug for Net {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

// Implementing the forward pass of the CNN model with ReLU and Dropout.
impl ModuleT for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view([-1, 1, 28, 28]) // Reshape input to 1x28x28 images.
            .apply(&self.conv1) // Apply first convolutional layer.
            .apply_pooling(self.pooling, 2)
            .apply(&self.conv2) // Apply second convolutional layer.
            .apply_pooling(self.pooling, 2)
            .view([-1, 1024]) // Flatten.
            .apply(&self.fc1) // Apply first linear layer.
            .relu() // ReLU activation.
            .dropout(0.5, train) // Dropout layer for regularization.
            .apply(&self.fc2) // Final linear layer for classification.
    }
}

// Function to train and test the CNN model on the MNIST dataset.
fn run_conv() -> Result<()> {
    // Load the MNIST dataset; this will download if the files are missing.
    let m = vision::mnist::load_dir("data")?;

    // Use GPU if available, otherwise use CPU
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let net = Net::new(&vs.root(), PoolingMethod::Max); // Initialize the CNN model.
    let mut opt = nn::Adam::default().build(&vs, 1e-4)?; // Set up the optimizer.

    // Reshape and normalize the training and test images
    let train_images = m.train_images.view([-1, 1, 28, 28]) / 255.0;
    let train_labels = m.train_labels;
    let test_images = m.test_images.view([-1, 1, 28, 28]) / 255.0;
    let test_labels = m.test_labels;

    // Training loop for the CNN model.
    for epoch in 1..=10 {
        // Shuffle and split the training data into batches
        for (bimages, blabels) in train_images.split(256, 0).into_iter().zip(train_labels.split(256, 0).into_iter()) {
            let loss = net.forward_t(&bimages, true).cross_entropy_for_logits(&blabels);
            opt.backward_step(&loss); // Backpropagation step.
        }

        // Calculate and print test accuracy at the end of each epoch
        let test_accuracy = net.batch_accuracy_for_logits(&test_images, &test_labels, vs.device(), 1024);
        println!("Epoch: {:4}, Test Accuracy: {:5.2}%", epoch, 100. * test_accuracy);
    }
    Ok(())
}