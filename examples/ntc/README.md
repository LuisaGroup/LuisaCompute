# LuisaCompute NTC

An implementation of NVIDIA RTX Neural Texture Compression (NTC) written in LuisaCompute.

---

## Features

- **NTEX v3 compatibility**: Read and write official `.ntc` packages with bi-directional compatibility verified against NVIDIA's `ntc-cli`.
- **Numerical parity**: Decoded outputs match reference INT8 decoders at >54 dB PSNR across all material channels.
- **GPU BCn encoders**: Built-in compute shaders for BC1, BC3, BC4, BC5, and BC7 block compression directly in VRAM without CPU readback.
- **Native C++ training**: GPU trainer with analytical gradients, fused Adam optimizer, and QAT exporter in C++ (no Python or PyTorch required).
- **Real-time viewer**: GLFW-based interactive viewer for previewing channels and mip levels.
- **C++ library**: Modular static library (`luisa-ntc`) under `namespace luisa::ntc` (`namespace ntc = luisa::ntc;`).

---

## Comparison with NVIDIA RTX NTC

| Feature / Metric | NVIDIA RTX NTC SDK | LuisaCompute NTC (`luisa-ntc`) |
| :--- | :--- | :--- |
| **Container Format** | NTEX v3 | NTEX v3 (binary compatible) |
| **Numerical Parity** | Reference baseline | >54 dB PSNR |
| **Decompression Path** | `VK_NV_cooperative_vector` / D3D12 CoopVec (Tensor Cores) | LuisaCompute SIMT compute shader (uncompressed FP32) |
| **Latency: RTX 3090 (2K, 10-ch)** | ~0.54 ms | ~20.4 ms |
| **Latency: RTX 4090 (2K, 10-ch)** | ~0.24 ms | ~7.5 ms |
| **CUDA BCn Encoding** | Not supported (requires D3D12 / Vulkan) | Native CUDA compute shaders (BC1–BC7) |
| **BCn Transcode Speed** | ~7–10 GT/s | 14–16.8 GT/s |
| **Training Pipeline** | Python + PyTorch + custom CUDA extensions | Standalone C++ GPU trainer |
| **Dependencies** | Proprietary graphics runtime SDKs | LuisaCompute + bundled `stb` & `yyjson` |

**Notes on performance differences**:
- NVIDIA's official implementation relies on vendor-specific cooperative vector extensions (`VK_NV_cooperative_vector`) to run the MLP on Tensor Cores.
- LuisaCompute NTC uses standard SIMT compute shaders and outputs full uncompressed FP32 channels to device memory.
- NVIDIA's CLI does not provide CUDA-based BCn encoders; LuisaCompute NTC implements GPU BCn encoders directly on CUDA.

---

## Benchmark

Measurements on a 2048×2048, 10-channel material (Color, Normal, Roughness, AO, Displacement) averaged over 50 iterations:

### Decompression Throughput

| Mip Level | Resolution | RTX 3090 | RTX 4090 |
| :---: | :---: | :---: | :---: |
| 0 | 2048 × 2048 | 205.5 MT/s (20.41 ms) | 558.0 MT/s (7.52 ms) |
| 1 | 1024 × 1024 | 185.7 MT/s (5.65 ms)  | 520.0 MT/s (2.02 ms) |
| 2 | 512 × 512   | 169.5 MT/s (1.55 ms)  | 435.7 MT/s (0.60 ms) |
| 3 | 256 × 256   | 106.9 MT/s (0.61 ms)  | 405.7 MT/s (0.16 ms) |

### GPU BCn Block Compression (RTX 4090 vs RTX 3090)

| Format | Channels | RTX 3090 Latency | RTX 3090 Throughput | RTX 4090 Latency | RTX 4090 Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BC1** | RGB | 0.533 ms | 7,873 MT/s | 0.413 ms | 10,145 MT/s |
| **BC3** | RGBA | 0.842 ms | 4,981 MT/s | 0.532 ms | 7,877 MT/s |
| **BC4** | Single-channel | 0.249 ms | 16,814 MT/s | 0.290 ms | 14,487 MT/s |
| **BC5** | Two-channel | 0.550 ms | 7,628 MT/s | 0.522 ms | 8,042 MT/s |
| **BC7** | RGBA | 0.546 ms | 7,676 MT/s | 0.297 ms | 14,107 MT/s |

---

## Project Structure

```
examples/ntc/
├── CMakeLists.txt          # Target definitions
├── xmake.lua               # XMake definitions
├── include/libntc/         # Public headers
│   ├── ntc.h               # Umbrella header
│   ├── common.h            # Architecture constants & types
│   ├── manifest.h          # Material manifest parser
│   ├── ntex_container.h    # NTEX v3 container parser/writer
│   ├── feature_grid.h      # Latent grid math
│   ├── infer_engine.h      # GPU decompression engine
│   ├── bcn_encoder.h       # GPU BC1–BC7 block encoders
│   ├── trainer.h           # GPU analytical trainer
│   └── gdeflate_codec.h    # Buffer compression view
├── src/                    # Implementations
│   ├── core/               # Container, grid, manifest
│   ├── infer/              # Decompression kernels
│   ├── bcn/                # BCn compute shaders
│   └── train/              # Training kernels & Adam optimizer
└── tools/                  # Executable targets
    ├── luisa_ntc_cli.cpp   # CLI tool
    ├── luisa_ntc_bench.cpp # Benchmark tool
    ├── luisa_ntc_test.cpp  # Parity test suite
    └── luisa_ntc_viewer.cpp# Real-time viewer
```

---

## Building

### Prerequisites
- C++20 compiler (MSVC 2022 / Clang 16+ / GCC 12+)
- CMake 3.24+ or XMake 2.8+
- CUDA Toolkit 12.0+

### CMake
```bash
cmake -B build -S . -DLUISA_COMPUTE_ENABLE_GUI=ON
cmake --build build --config Release --target luisa-ntc luisa_ntc_cli luisa_ntc_bench luisa_ntc_test luisa_ntc_viewer
```

### XMake
```bash
xmake config -m release
xmake build luisa-ntc luisa_ntc_cli luisa_ntc_bench luisa_ntc_test luisa_ntc_viewer
```

---

## Usage

### Inspect Container
```bash
luisa_ntc_cli --describe path/to/texture.ntc
```

### Decompress
```bash
# Decompress to PNG images
luisa_ntc_cli -D --loadCompressed=path/to/texture.ntc -i ./output_png -B none

# Decompress and transcode to BCn DDS textures
luisa_ntc_cli -D --loadCompressed=path/to/texture.ntc -i ./output_dds
```

### Train / Compress
Train from a `manifest.json`:
```bash
luisa_ntc_cli -c --loadManifest=path/to/manifest.json -o material.ntc -S 10000
```

### Benchmark
```bash
luisa_ntc_bench --ntc=path/to/texture.ntc --backend=cuda --iters=50
```

### Run Tests
```bash
luisa_ntc_test cuda
```

### Viewer
```bash
luisa_ntc_viewer --ntc=path/to/texture.ntc --channel=1 --mip=0
```
- `A` / `D` or `←` / `→`: Switch texture channel
- `W` / `S` or `↑` / `↓`: Switch mip level
- `1` – `5`: Jump to channel
- `ESC`: Exit

---

## C++ API

```cpp
#include <libntc/ntc.h>

namespace ntc = luisa::ntc;

// Load container
ntc::NtexContainer container;
std::string error;
if (!container.load_from_file("material.ntc", error)) {
    // Handle error...
}

// Initialize inference engine
ntc::InferEngine infer(device, stream);
infer.load_container(container);

// Decompress mip level 0 to GPU device memory
uint32_t mip = 0;
uint32_t width = container.width >> mip;
uint32_t height = container.height >> mip;
luisa::compute::Buffer<float> gpu_pixels = infer.decompress_mip_device(mip, width, height);

// Encode to BC7 on GPU
ntc::BcnEncoder encoder(device, stream);
luisa::compute::Buffer<uint8_t> bc7_data = encoder.encode_bc7(gpu_pixels, width, height);
```

---

## License

Same as the parent LuisaCompute project (BSD-3-Clause).
