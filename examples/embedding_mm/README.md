# Multimodal EmbeddingGemma-2 with LuisaCompute

**EmbeddingGemma-2** (`google/embeddinggemma-2`) is Google's open multimodal embedding model built on the Gemma 4 architecture. It maps **text**, **images**, **audio**, and **video** into a shared **768-dimensional** vector space, enabling cross-modal semantic search, retrieval, and classification.

This directory provides a pure C++20 / **LuisaCompute GPU DSL** implementation (`luisa::embed_mm`) modeled 1:1 after Google's official tutorial: [**Multimodal EmbeddingGemma with Sentence Transformers**](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers). Every official example asset, prompt format, and configuration option is supported with **`1.000000` cosine similarity** against the official PyTorch `SentenceTransformer` reference across **DirectX 12 (`dx`)**, **NVIDIA CUDA (`cuda`)**, and **Vulkan (`vk`)**.

> For deep architectural details, mathematical operator specifications (`embed/ops/`), sub-model execution graphs (`embed/models/`), and binary format definitions (`.mmegm` / `.mmfeat`), see [**ARCHITECTURE.md**](./ARCHITECTURE.md).

---

## Table of Contents

1. [Setup: Build, Model Export & Feature Preprocessing](#1-setup-build-model-export--feature-preprocessing)
2. [Selective Encoder Loading (Saving GPU Memory)](#2-selective-encoder-loading-saving-gpu-memory)
3. [Generating Text Embeddings](#3-generating-text-embeddings)
4. [Working with Images](#4-working-with-images)
5. [Working with Audio](#5-working-with-audio)
6. [Working with Video](#6-working-with-video)
7. [Interleaved Multimodal Inputs](#7-interleaved-multimodal-inputs)
8. [Matryoshka Representation Learning (MRL) & GPU Benchmarks](#8-matryoshka-representation-learning-mrl--gpu-benchmarks)

---

## 1. Setup: Build, Model Export & Feature Preprocessing

### 1.1 Build the LuisaCompute Executables

Using **XMake** (recommended):
```powershell
xmake build lc-example-embedding-mm
xmake build example_embed_mm_text example_embed_mm_vision example_embed_mm_audio example_embed_mm_video example_embed_mm_bench example_embed_mm_test
```

Or using **CMake**:
```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target example_embed_mm_text example_embed_mm_vision example_embed_mm_audio example_embed_mm_video example_embed_mm_bench example_embed_mm_test
```

### 1.2 Export Model Weights (`.mmegm`)

Convert the HuggingFace `google/embeddinggemma-2` safetensors checkpoint into a memory-mapped `.mmegm` package with pre-transposed $[K, N]$ weights for coalesced GPU GEMM execution:

```powershell
python examples/embedding_mm/tools/export_model.py `
    --model-dir /path/to/embeddinggemma-2 `
    --output model.mmegm
```

### 1.3 Preprocess Media Assets with Official `AutoProcessor` (`.mmfeat`)

To keep the C++ runtime 100% pure GPU DSL without third-party CPU media codec dependencies, raw images, audio, and video are preprocessed into compact `.mmfeat` binary bundles via HuggingFace's official `EmbeddingGemma2Processor`:

```powershell
# Generate all official Google tutorial .mmfeat bundles and PyTorch golden references:
python examples/embedding_mm/tools/preprocess.py official `
    --model-dir /path/to/embeddinggemma-2 `
    --media-dir /path/to/media `
    --out-dir examples/embedding_mm/data

# Or preprocess any custom image / audio / video file on demand:
python examples/embedding_mm/tools/preprocess.py image --model-dir /path/to/embeddinggemma-2 --input photo.jpg --output photo.mmfeat --max-soft-tokens 280
python examples/embedding_mm/tools/preprocess.py audio --model-dir /path/to/embeddinggemma-2 --input clip.wav  --output clip.mmfeat
python examples/embedding_mm/tools/preprocess.py video --model-dir /path/to/embeddinggemma-2 --input clip.mp4  --output clip.mmfeat --num-frames 16 --max-soft-tokens 70 --add-timestamps
```

---

## 2. Selective Encoder Loading (Saving GPU Memory)

EmbeddingGemma-2 consists of a **Language Model** (~297M parameters), a **Vision Tower** (~170M parameters), and an **Audio Conformer Tower** (~300M parameters). If your workload only requires text or text+images, you can skip loading unused encoders to save substantial GPU VRAM and shader compilation time—matching `load_vision_encoder=False` / `load_audio_encoder=False` in Sentence Transformers:

| Configuration | Active Modules | Approx. Params | C++ `SessionConfig` |
|---|---|---|---|
| **Full Multimodal** | Language + Vision + Audio | ~767M | `enable_vision = true, enable_audio = true` |
| **Text + Vision (Image/Video)** | Language + Vision | ~467M | `enable_vision = true, enable_audio = false` |
| **Text + Audio** | Language + Audio | ~597M | `enable_vision = false, enable_audio = true` |
| **Text-Only** | Language Only | ~297M | `enable_vision = false, enable_audio = false` |

```cpp
#include "embedding_mm/embed/session.h"

luisa::embed_mm::SessionConfig config;
config.enable_vision = false; // Skip VisionTower (~170M params)
config.enable_audio  = false; // Skip AudioTower  (~300M params)
config.default_dim   = 768u;

auto session = luisa::embed_mm::Session::create(
    device, "model.mmegm", "tokenizer.json", config);
```

---

## 3. Generating Text Embeddings

For retrieval tasks, use `TaskType::SearchQuery` (`task: search result | query: `) for search queries and `TaskType::Document` (`title: none | text: `) for candidate documents.

### C++ Example ([`example_embed_mm_text.cpp`](./example_embed_mm_text.cpp))

```cpp
// Equivalent to SentenceTransformers model.encode_query(...) and model.encode_document(...)
auto q_res = session->encode_texts(
    {"Which planet is known as the Red Planet?"},
    luisa::embed_mm::TaskType::SearchQuery,
    768u);

luisa::vector<luisa::string> documents = {
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter is the largest planet in our solar system, a gas giant with a Great Red Spot.",
    "Saturn, famous for its stunning ring system, is the second-largest planet in our solar system."
};
auto d_res = session->encode_texts(
    documents,
    luisa::embed_mm::TaskType::Document,
    768u);
```

### Run the Demo

```powershell
.\bin\release\example_embed_mm_text.exe dx model.mmegm tokenizer.json
```

**Output:**
```text
Query: "Which planet is known as the Red Planet?"
----------------------------------------------------------------
  [0] sim = 0.6746  "Venus is often called Earth's twin..."
  [1] sim = 0.8482  "Mars, known for its reddish appearance, is often referred to as the Red Planet."  <-- Top-1
  [2] sim = 0.6996  "Jupiter is the largest planet in our solar system..."
  [3] sim = 0.6758  "Saturn, famous for its stunning ring system..."
```

---

## 4. Working with Images

EmbeddingGemma-2 encodes images through a 16-layer Vision Transformer (`VisionTower`) with $16 \times 16$ patches and $3 \times 3$ spatial pooling into soft tokens (`<|image|>`), mapping them into the same vector space as text.

### 4.1 Image-to-Text Similarity

Using the official tutorial images (`GoldenGate.png` and `kitchen_painting.jpg`):

```cpp
auto img0 = luisa::embed_mm::FeatureLoader::load("examples/embedding_mm/data/image_GoldenGate_280.mmfeat");
auto img1 = luisa::embed_mm::FeatureLoader::load("examples/embedding_mm/data/image_kitchen_painting_280.mmfeat");

auto emb0 = session->encode_feature(img0, 768u);
auto emb1 = session->encode_feature(img1, 768u);

auto doc_res = session->encode_texts({
    "A photo of the Golden Gate Bridge surrounded by fog",
    "A painting of a kitchen counter with fruit and wine",
    "A diagram of a transformer neural network architecture"
}, luisa::embed_mm::TaskType::Document, 768u);
```

### 4.2 Configuring Image Token Budget (`max_soft_tokens`: `280` vs. `1120`)

Just like setting `model[0].processing_kwargs = {"image": {"max_soft_tokens": 1120}}` in Sentence Transformers, you can increase `max_soft_tokens` from `280` (default) to `1120` for dense text/OCR or fine-grained visual details. The LuisaCompute `VisionTower` automatically chunks high-resolution attention dispatches (`10080` patches) to stay within GPU grid limits.

### Run the Demo

```powershell
.\bin\release\example_embed_mm_vision.exe dx model.mmegm tokenizer.json examples\embedding_mm\data
```

**Output:**
```text
--- Image [GoldenGate.png (280 soft tokens)] vs. Text Candidates ---
  [0] sim = 0.6537  "A photo of the Golden Gate Bridge surrounded by fog"  <-- Top-1
  [1] sim = 0.4534  "A painting of a kitchen counter with fruit and wine"
  [2] sim = 0.4317  "A diagram of a transformer neural network architecture"

--- Image [kitchen_painting.jpg (280 soft tokens)] vs. Text Candidates ---
  [0] sim = 0.4375  "A photo of the Golden Gate Bridge surrounded by fog"
  [1] sim = 0.6352  "A painting of a kitchen counter with fruit and wine"  <-- Top-1
  [2] sim = 0.4371  "A diagram of a transformer neural network architecture"

--- High-Resolution Budget Comparison (280 vs. 1120 Soft Tokens) ---
  GoldenGate.png       cos(280, 1120) = 0.967980 | vs PyTorch Golden (1120) = 1.000000
  kitchen_painting.jpg cos(280, 1120) = 0.961775 | vs PyTorch Golden (1120) = 1.000000
```

---

## 5. Working with Audio

EmbeddingGemma-2 processes 16 kHz mono audio waveforms into 128-bin log-mel spectrograms and encodes them via a 12-layer Conformer (`AudioTower`) into audio soft tokens (`<|audio|>`), enabling zero-shot speech-to-text retrieval without an intermediate ASR transcription step.

### C++ Example ([`example_embed_mm_audio.cpp`](./example_embed_mm_audio.cpp))

```cpp
// Official sample: roses-are.wav ("Roses are red, violets are blue, sugar is sweet, and so are you.")
auto audio_feat = luisa::embed_mm::FeatureLoader::load("examples/embedding_mm/data/audio_roses.mmfeat");
auto audio_emb  = session->encode_feature(audio_feat, 768u);

auto doc_res = session->encode_texts({
    "Roses are red, violets are blue, sugar is sweet, and so are you.",
    "The quick brown fox jumps over the lazy dog.",
    "A weather forecast predicting heavy rain tomorrow."
}, luisa::embed_mm::TaskType::Document, 768u);
```

### Run the Demo

```powershell
.\bin\release\example_embed_mm_audio.exe dx model.mmegm tokenizer.json examples\embedding_mm\data
```

**Output:**
```text
--- Audio [roses-are.wav (307 frames -> 77 soft tokens)] vs. Text Candidates ---
  [0] sim = 0.8780  "Roses are red, violets are blue, sugar is sweet, and so are you."  <-- Top-1
  [1] sim = 0.5621  "The quick brown fox jumps over the lazy dog."
  [2] sim = 0.5419  "A weather forecast predicting heavy rain tomorrow."
  Cosine similarity vs. Official PyTorch Golden = 1.000000
```

---

## 6. Working with Video

EmbeddingGemma-2 encodes video by sampling frames over time, encoding each frame through `VisionTower`, and scattering all frame soft tokens into the language model sequence (`<|video|>`).

### 6.1 Default Video Encoding vs. Custom Sampling & Timestamps

- **Default (`video_cat_default.mmfeat`)**: 1 FPS sampling (up to 32 frames, `29` frames for `cat_dozing.mp4`), `140` soft tokens per frame (`112` active for 16:9 aspect ratio $\implies 3248$ visual soft tokens).
- **Custom Timestamped (`video_cat_16f_ts.mmfeat`)**: `num_frames=16`, `max_soft_tokens=70` (`45` active per frame), `add_timestamps=True` (inserts `00:00 <|image|>00:01 <|image|>...` temporal markers).

### Run the Demo ([`example_embed_mm_video.cpp`](./example_embed_mm_video.cpp))

```powershell
.\bin\release\example_embed_mm_video.exe dx model.mmegm tokenizer.json examples\embedding_mm\data
```

**Output:**
```text
--- Video [cat_dozing.mp4 (Default 29 frames)] vs. Text Candidates ---
  [0] sim = 0.6594  "A cat sleeping peacefully on a couch"  <-- Top-1
  [1] sim = 0.4533  "A dog running across a green park"
  [2] sim = 0.4150  "A car driving down a busy highway at night"

--- Video [cat_dozing.mp4 (16 frames, 70 budget, Timestamps)] vs. Text Candidates ---
  [0] sim = 0.6775  "A cat sleeping peacefully on a couch"  <-- Top-1
  [1] sim = 0.4716  "A dog running across a green park"
  [2] sim = 0.4317  "A car driving down a busy highway at night"
  Cosine similarity vs. Official PyTorch Golden = 1.000000
```

---

## 7. Interleaved Multimodal Inputs

EmbeddingGemma-2 natively supports combining multiple modalities within a single input sequence. Both dict-style inputs (`{"text": ..., "images": [...], "audio": ...}`) and explicit placeholder strings (`<|image|>`, `<|audio|>`, `<|video|>`) are supported via indexed soft-token scattering:

1. **`interleaved_multimodal.mmfeat`**: Text (`"Landmark and interior photos with spoken narration"`) + 2 Images (`GoldenGate.png`, `kitchen_painting.jpg`) + Audio (`roses-are.wav`).
2. **`interleaved_video_images.mmfeat`**: Prompt (`"Video clip: <|video|> Reference images: <|image|> <|image|>"`) combining `cat_dozing.mp4` + 2 Images.
3. **`interleaved_media_sequence.mmfeat`**: Pure media sequence `"<|image|><|audio|><|image|>"` without text.

All three interleaved queries are evaluated and benchmarked in [`example_embed_mm_bench.cpp`](./example_embed_mm_bench.cpp) and verified in [`example_embed_mm_test.cpp`](./example_embed_mm_test.cpp) with **`cos_sim = 1.000000`**.

---

## 8. Matryoshka Representation Learning (MRL) & GPU Benchmarks

EmbeddingGemma-2 is trained with **Matryoshka Representation Learning (MRL)**, allowing you to truncate the output embedding dimension from `768` down to `512`, `256`, or `128` via the `target_dim` parameter while preserving semantic ranking quality (`MRLNormalizeKernel` truncates and re-normalizes on the GPU in a single fused pass).

### Run the Full Benchmark & Automated Verification Suite

```powershell
# Run GPU latency benchmark & MRL cross-modal evaluation:
.\bin\release\example_embed_mm_bench.exe dx model.mmegm tokenizer.json examples\embedding_mm\data

# Run the complete unit test & PyTorch 0-drift parity suite across DirectX 12, CUDA, and Vulkan:
.\bin\release\example_embed_mm_test.exe dx   model.mmegm tokenizer.json examples\embedding_mm\data
.\bin\release\example_embed_mm_test.exe cuda model.mmegm tokenizer.json examples\embedding_mm\data
.\bin\release\example_embed_mm_test.exe vk   model.mmegm tokenizer.json examples\embedding_mm\data
```

### Multi-Backend Parity Summary (NVIDIA GeForce RTX 3090)

| Modality / Test Scenario | MRL Dimensions | `dx` vs. PyTorch | `cuda` vs. PyTorch | `vk` vs. PyTorch |
|---|---|---|---|---|
| **Text (`SearchQuery` & `Document`)** | `768, 512, 256, 128` | `1.000000` | `1.000000` | `1.000000` |
| **Image (`280` & `1120` soft tokens)** | `768, 512, 256, 128` | `1.000000` | `1.000000` | `1.000000` |
| **Audio (`roses-are.wav`)** | `768, 512, 256, 128` | `1.000000` | `1.000000` | `1.000000` |
| **Video (`29f` default & `16f` timestamped)** | `768, 512, 256, 128` | `1.000000` | `1.000000` | `1.000000` |
| **Interleaved Multimodal (3 Queries)** | `768, 512, 256, 128` | `1.000000` | `1.000000` | `1.000000` |
