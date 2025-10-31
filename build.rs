fn main() {
    // Instruct the linker to retain the CUDA warp_size symbol
    // Only needed when using CUDA-enabled LibTorch
    if std::env::var("LIBTORCH").is_ok() {
        println!("cargo:rustc-link-arg=/INCLUDE:?warp_size@cuda@at@@YAHXZ");
    }
}
