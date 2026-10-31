# EmbeddingGemma-2 Engine Architecture & Operator Specification (`ARCHITECTURE.md`)

Architecture, mathematical operators, binary format specifications, and development roadmap for the `luisa::embed_mm` engine in LuisaCompute, providing native GPU mega-kernel inference for Google's **EmbeddingGemma-2** (`google/embeddinggemma-2`) multimodal model.

For the user-facing quickstart and multimodal tutorial modeled after Google's official guide, see [README.md](./README.md).

---

## 1. Executive Summary & Development Milestones

Google DeepMind's **EmbeddingGemma-2** (`google/embeddinggemma-2`, ~767M total parameters) is an open multimodal representation model built upon the Gemma 4 multimodal architecture. It maps text (including source code), images, audio, and video into a unified dense vector space of **768 dimensions** with Matryoshka Representation Learning (MRL) support down to 512, 256, and 128 dimensions.

### Completed Milestones

1. **Phase 1 — Language Model & MRL Head (`LanguageModel`, ~297M params)**:
   - 24-layer hybrid-attention Transformer backbone (`hidden_size = 512`, `intermediate_size = 2048`).
   - 5:1 Local Sliding-Window Attention (`sliding_window = 512`, `head_dim = 256`, `kv_heads = 2`, $\theta = 10^4$) paired with Global Full Bidirectional Attention on layers `5, 11, 17, 23` (`head_dim = 512`, `kv_heads = 1`, $\theta = 10^6$).
   - Per-Layer Embeddings (PLE) with global projection ($512 \to 24 \times 512$), per-layer GeGLU gating, and `layer_scalar` residual scaling.
   - Sequence-mask-aware Mean Pooling (`1_Pooling`), Linear Projection ($512 \to 768$, `2_Dense`), and MRL truncation ($768 \to 512, 256, 128$) + $L_2$ Normalization.
   - Zero-dependency C++ Gemma BPE Tokenizer (`yyjson`) with byte fallback (`<0x00>`–`<0xFF>`).

2. **Phase 2 — Vision Tower & Dynamic Spatial Pooling (`VisionTower`, ~170M params)**:
   - 16-layer Vision Transformer (`Gemma4VisionModel`, `hidden_size = 768`, 12 heads, `head_dim = 64`, `intermediate_size = 3072`, GeGLU).
   - 2D Factorized Position Embedding table lookup ($2 \times 10240 \times 768$) and 2D Axial RoPE ($\theta_{\text{base}} = 100.0$).
   - Query-row-chunked full bidirectional attention supporting both standard `280` (`max_patches = 2520`) and high-resolution `1120` (`max_patches = 10080`) soft token budgets within 16-bit grid and 32-bit buffer limits.
   - 2D Spatial Average Pooling ($3 \times 3$ patch window scaled by $\sqrt{768}/9$) + Multimodal Connector (`RMSNormNoWeight(768)` + `Linear(768 -> 512)`).

3. **Phase 3 — Audio Conformer Tower (`AudioTower`, ~300M params)**:
   - 2-stage Subsampling 2D Convolution (`Conv2D 3x3 stride 2 + LayerNorm + ReLU`, $4\times$ time/frequency reduction) + Input Projection ($1024 \to 1024$).
   - 12-layer Conformer (`Gemma4AudioModel`, `hidden_size = 1024`, 8 heads, `head_dim = 128`, `intermediate_size = 4096`):
     - Dual Macaron half-step Feed-Forward Networks ($0.5\times$ residual scaling, SiLU activation, per-linear input/output clipping).
     - Chunked Local Self-Attention (`chunk_size = 12`, `context = 24` with 13 left relative sinusoidal positions, `per_dim_scale` softplus scaling, and logit softcapping at $50.0$).
     - Causal `LightConv1D` (GLU gating + depthwise 1D convolution with `kernel_size = 5` and causal left padding 4).
   - Output Projection with bias ($1024 \to 1536$) + Multimodal Connector (`RMSNormNoWeight(1536)` + `Linear(1536 -> 512)`).

4. **Phase 4 — Video & Interleaved Multimodal Pipeline**:
   - Multi-frame spatio-temporal encoding reusing `VisionTower` across $N$ sampled frames (`140` or `70` soft tokens per frame) with optional `MM:SS` timestamp prefixes (`chat_template.jinja`).
   - Non-contiguous indexed soft-token scatter (`ScatterIndexedTokensKernel`) enabling arbitrary interleaved combinations of text, multiple images, audio, and video in a single sequence.

5. **Phase 5 — Pure GPU Engine & Two-Layer Operator/Model Architecture**:
   - **Pure GPU C++ Engine**: Offloaded all heavy CPU media decoding/resampling (image bicubic resize, WAV/STFT/Mel filterbank, MP4 decoding) to a lightweight Python script ([tools/preprocess.py](./tools/preprocess.py)) powered by the official HuggingFace `AutoProcessor` (`EmbeddingGemma2Processor`), exporting compact `.mmfeat` binary bundles consumed directly by C++ [`FeatureLoader`](./embed/feature_loader.h).
   - **Two-Layer Codebase Organization**: Factored atomic GPU DSL shaders into `embed/ops/` and modality execution graphs into `embed/models/`.

---

## 2. Two-Layer Source Architecture (`embed/ops/` + `embed/models/`)

```text
examples/embedding_mm/
├── README.md                          # Official-style multimodal tutorial & quickstart guide
├── ARCHITECTURE.md                    # Engine architecture, math specs & binary format reference
├── CMakeLists.txt                     # CMake build configuration
├── xmake.lua                          # XMake build configuration
├── example_embed_mm_text.cpp          # Section 3 & 7: Text embeddings, task prompts & MRL demo
├── example_embed_mm_vision.cpp        # Section 4: Image embeddings (280 & 1120 soft tokens) demo
├── example_embed_mm_audio.cpp         # Section 5: Audio embeddings & cross-modal retrieval demo
├── example_embed_mm_video.cpp         # Section 6: Video embeddings (default & timestamped) demo
├── example_embed_mm_bench.cpp         # Section 6.5 & 8: Interleaved multimodal + GPU latency benchmark
├── example_embed_mm_test.cpp          # Automated unit & PyTorch 0-drift parity suite (dx/cuda/vk)
├── embed/
│   ├── tokenizer.{h,cpp}              # Pure C++ Gemma BPE tokenizer (yyjson)
│   ├── package.{h,cpp}                # Memory-mapped .mmegm model weight loader
│   ├── feature_loader.{h,cpp}         # Binary .mmfeat feature bundle loader (MMFT v1)
│   ├── executor.h                     # Abstract GPU execution graph interface
│   ├── session.{h,cpp}                # High-level multimodal inference session API
│   ├── ops/                           # Layer 1: Atomic LuisaCompute DSL GPU Operators
│   │   ├── ops_gemm.{h,cpp}           # 16x16 tiled shared-memory GEMM & GEMM+Bias
│   │   ├── ops_norm.{h,cpp}           # Gemma RMSNorm (1+gamma), RMSNormNoWeight, L2Normalize
│   │   ├── ops_rope.{h,cpp}           # 1D Text RoPE, 2D Vision Axial RoPE, Audio Sinusoidal RelPos
│   │   ├── ops_attention.{h,cpp}      # GQA Scores (sliding/full, chunked rows), Softmax, PV, Audio Chunked Attn
│   │   ├── ops_mlp.{h,cpp}            # GeGLU, PLEGate, ResidualAdd, ScaleAdd, AudioClamp/SiLU/GLU/AddScale
│   │   ├── ops_conv.{h,cpp}           # Audio 2D Subsample Conv0/Conv1 (fused LayerNorm+ReLU), Causal Depthwise Conv1D
│   │   ├── ops_pool.{h,cpp}           # Vision 3x3 Spatial Pool, Sequence Mask MeanPool, MRL Normalize
│   │   └── ops_embed.{h,cpp}          # EmbedLookup, ScalePixels, AddPatchPosEmbed, MaskFromPos, SoftToken Scatter
│   └── models/                        # Layer 2: Modality Sub-Models & Top-Level Orchestrator
│       ├── vision_tower.{h,cpp}       # VisionTower (model.vision_tower + model.embed_vision)
│       ├── audio_tower.{h,cpp}        # AudioTower (model.audio_tower + model.embed_audio)
│       ├── language_model.{h,cpp}     # LanguageModel (model.language_model + PLE + 1_Pooling + 2_Dense + MRL)
│       └── embedding_gemma2.{h,cpp}   # EmbeddingGemma2 orchestrator & Selective Encoder Loading
└── tools/
    ├── export_model.py                # Exports HuggingFace safetensors -> .mmegm binary package
    ├── preprocess.py                  # Official AutoProcessor CLI -> .mmfeat binary bundles & golden JSON
    └── pack_model.cpp                 # C++ model package inspector & validator
```

---

## 3. Model Hyperparameters

Derived directly from `google/embeddinggemma-2/config.json`:

| Sub-Model | Parameter | Value | Description |
|---|---|---|---|
| **Language (`text_config`)** | `vocab_size` | 262,144 | BPE vocabulary size with byte fallback tokens |
| | `hidden_size` | 512 | Core transformer hidden dimensionality ($H$) |
| | `intermediate_size` | 2,048 | GeGLU MLP intermediate dimension ($I$) |
| | `num_hidden_layers` | 24 | Total transformer blocks |
| | `sliding_layers` | 20 layers | $H_q = 4, H_{kv} = 2, D = 256, W = 512, \theta = 10^4$ |
| | `full_layers` | `[5, 11, 17, 23]` | $H_q = 4, H_{kv} = 1, D = 512, \theta = 10^6$ |
| | `per_layer_input_dim` | 512 | Per-Layer Embedding (PLE) gating dimension per block |
| | `embedding_dim` | 768 | Final dense projection dimension (`512 -> 768`) |
| **Vision (`vision_config`)** | `hidden_size` | 768 | ViT hidden dimensionality |
| | `intermediate_size` | 3,072 | ViT GeGLU intermediate dimension |
| | `num_hidden_layers` | 16 | ViT transformer blocks |
| | `num_attention_heads` | 12 | Full bidirectional MHA ($D = 64, \theta_{\text{axial}} = 100.0$) |
| | `patch_size` / `pool` | $16 \times 16$ / $3 \times 3$ | $3 \times 3$ patch pooling $\to 1$ soft token |
| **Audio (`audio_config`)** | `hidden_size` | 1,024 | Conformer hidden dimensionality |
| | `num_hidden_layers` | 12 | Conformer blocks (Macaron FFN + Chunked Attn + LightConv1D) |
| | `num_attention_heads` | 8 | Chunked local attention ($D = 128$, chunk 12, window 24) |
| | `conv_kernel_size` | 5 | Causal depthwise 1D convolution kernel size |
| | `output_proj_dims` | $1024 \to 1536 \to 512$ | Audio output projection + connector projection |

---

## 4. Mathematical Operators (`embed/ops/`)

### 4.1 Normalization (`ops_norm.h/.cpp`)
- **RMSNorm with weight (`RMSNormKernel`)**:
  $$\text{RMS}(x) = \sqrt{\frac{1}{H}\sum_{i=0}^{H-1} x_i^2 + \epsilon}, \quad y_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i$$
  *(Note: During `export_model.py`, Gemma weights that use $(1 + w)$ are pre-baked or exported directly as $\gamma$ matching the exact layer convention.)*
- **Parameterless RMSNorm (`RMSNormNoWeightKernel`)**: Used for V-norm (`v_norm` in text and vision attention) and multimodal connectors (`embed_vision`, `embed_audio`).

### 4.2 Positional Encodings (`ops_rope.h/.cpp`)
- **1D RoPE (`RoPEKernel`)**: `rotate_half` layout with $\theta \in \{10^4, 10^6\}$.
- **2D Axial RoPE (`AxialRoPEKernel`)**: Splits each 64-dim head into $[0..31]$ (X/width coordinate) and $[32..63]$ (Y/height coordinate) with $\theta = 100.0$; zeros padded patches where `pos_id < 0`.
- **Audio Sinusoidal Relative Positions (`AudioGenRelPosKernel`)**: Generates `[13, 1024]` relative key positions $p \in [12, \dots, 0]$ with $\text{inv\_ts}_i = \exp(-i \ln(10^4) / 511)$.

### 4.3 Attention (`ops_attention.h/.cpp`)
- **Bidirectional GQA/MHA (`GQAScoresKernel`, `SoftmaxRowKernel`, `GQAPvKernel`)**:
  Supports sliding-window masking ($|q_i - k_i| < 512$) and full bidirectional attention. Includes a `row_offset` parameter so high-resolution vision (`1120` soft tokens $\implies 10080$ patches $\implies 120,960$ query-head rows) is chunked into $\le 30,240$ rows per dispatch, staying well within the 65,535 thread-group limit and 32-bit buffer addressing limit across DirectX 12, Vulkan, and CUDA.
- **Audio Chunked Attention (`AudioScaleQKKernel`, `AudioChunkedAttentionKernel`)**:
  Applies learned `per_dim_scale` via $\text{softplus}(s) / \ln 2$, projects relative keys $K_{\text{rel}} \in \mathbb{R}^{13 \times 1024}$, and evaluates 12-token query chunks over a 24-token local window with tanh softcapping:
  $$\text{logit} = 50.0 \cdot \tanh\left(\frac{Q K^T + Q K_{\text{rel}}^T}{50.0}\right)$$

---

## 5. Binary Formats

### 5.1 Preprocessed Feature Bundle (`.mmfeat` — `MMFT` v1)
Generated by `tools/preprocess.py` using HuggingFace `AutoProcessor` (`EmbeddingGemma2Processor`) and loaded by [`FeatureLoader`](./embed/feature_loader.h).

| Offset | Type | Field | Description |
|---|---|---|---|
| `0` | `char[4]` | `magic` | `"MMFT"` (`0x54464D4D`) |
| `4` | `uint32_t` | `version` | `1` |
| `8` | `uint32_t` | `modality` | `1` = Image, `2` = Audio, `3` = Video, `4` = Interleaved Multimodal |
| `12` | `uint32_t` | `num_tokens` | Prompt sequence length (`input_ids` & `attention_mask`) |
| `16` | `uint32_t` | `num_frames` | Number of image/video frames ($N$) |
| `20` | `uint32_t` | `max_patches` | Padded patches per frame (`2520`, `10080`, `1260`, or `630`) |
| `24` | `uint32_t` | `patch_dim` | `768` ($16 \times 16 \times 3$) |
| `28` | `uint32_t` | `num_patches_w` | Active horizontal patches ($W_p$) |
| `32` | `uint32_t` | `num_patches_h` | Active vertical patches ($H_p$) |
| `36` | `uint32_t` | `vision_soft_tokens_per_frame` | Soft tokens per frame ($W_p/3 \times H_p/3$) |
| `40` | `uint32_t` | `total_vision_soft_tokens` | Total vision soft tokens scattered into sequence |
| `44` | `uint32_t` | `audio_frames` | Mel spectrogram frames $T_{\text{in}}$ (e.g. `307`) |
| `48` | `uint32_t` | `audio_mel_bins` | Mel bins (`128`) |
| `52` | `uint32_t` | `total_audio_soft_tokens` | Output audio soft tokens $T_{\text{out}}$ (e.g. `77`) |
| `56` | `uint32_t[2]` | `reserved` | Zero-filled (`64 bytes` total header) |

Payloads follow immediately in tight little-endian order:
1. `input_ids`: `uint32_t[num_tokens]`
2. `attention_mask`: `float[num_tokens]`
3. `vision_token_indices`: `uint32_t[total_vision_soft_tokens]`
4. `audio_token_indices`: `uint32_t[total_audio_soft_tokens]`
5. `pixel_values`: `float[num_frames * max_patches * patch_dim]`
6. `pixel_position_ids`: `int32_t[num_frames * max_patches * 2]`
7. `audio_features`: `float[audio_frames * audio_mel_bins]`
8. *(For `modality == 4` Interleaved Multi-Image)* `per_image_dims`: `uint32_t[num_frames * 4]` (`max_patches, w, h, soft_tokens` per image).

### 5.2 Model Weight Package (`.mmegm` — `EGM2` v1)
Generated by `tools/export_model.py` and memory-mapped by [`Package`](./embed/package.h). Linear weights are pre-transposed to $[K, N]$ layout for coalesced GPU reads.

---

## 6. Verification & Zero-Drift Parity Summary

Run `example_embed_mm_test` across DirectX 12 (`dx`), CUDA (`cuda`), and Vulkan (`vk`):

| Test Suite | Target / Layer | Cosine Similarity vs. Official PyTorch |
|---|---|---|
| **Text & MRL** | `SearchQuery` & `Document` @ `768, 512, 256, 128` | `1.000000` |
| **Vision Tower Layers** | `patch_embed`, `vit_layer0`, `spatial_pool`, `connector` | `1.000000` |
| **Image Embeddings** | `GoldenGate.png` & `kitchen_painting.jpg` (`280` & `1120` budgets) | `1.000000` |
| **Audio Tower Layers** | `subsample_proj`, `layer0`, `layer11`, `output_proj`, `connector` | `0.999999` – `1.000000` |
| **Audio Embeddings** | `roses-are.wav` @ `768, 512, 256, 128` | `1.000000` |
| **Video Embeddings** | `cat_dozing.mp4` (`29f` default & `16f` timestamped) @ `768..128` | `1.000000` |
| **Interleaved Multimodal** | `query_multimodal`, `query_interleaved`, `query_media_sequence` | `1.000000` |
