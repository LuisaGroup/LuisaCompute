# Multimodal Embedding Engine (`mm_embedding`)

Architecture and specification guide for the `luisa::mm_embed` engine in LuisaCompute, providing high-performance inference support for Google's **EmbeddingGemma-2** multimodal model family.

---

## 1. Executive Summary & Model Overview

Google DeepMind's **EmbeddingGemma-2** (`google/embeddinggemma-2`) is an open multimodal representation model built upon the Gemma 2 / Gemma 4 architecture. It maps text (including source code), images, video, and audio into a unified dense vector space of **768 dimensions**.

### Multi-Phase Development Strategy

1. **Phase 1 (Current Scope — Text Core Backbone ~270M)**:
   - Embedding text encoder featuring a 24-layer Transformer backbone with 512 hidden dimension.
   - Grouped-Query Attention (GQA) with 4 Query heads and 2 Key/Value heads.
   - Dual-mode Rotary Position Embedding (RoPE) supporting both sliding window (512 tokens, $\theta = 10^4$) and full global attention ($\theta = 10^6$).
   - Quadruple alternating RMSNorm per block (`input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`).
   - GeGLU feed-forward networks (`intermediate_size` $= 2048$).
   - Sequence-mask-aware Mean Pooling + Linear Projection ($512 \to 768$).
   - Matryoshka Representation Learning (MRL) dimension truncation ($768 \to 512, 256, 128$) + $L_2$ Normalization.
   - Pure C++ Gemma BPE Tokenizer using `yyjson` with byte fallback.
   - Memory-mapped model package format (`model.mmegm`).

2. **Phase 2 (Current Scope — Vision Encoder ~170M & Multimodal Pipeline)**:
   - Vision transformer tower (`Gemma4VisionModel`, 16 ViT layers, hidden dimension 768, 12 heads, intermediate size 3072, GeGLU).
   - 2D Axial RoPE ($\theta_{\text{base}} = 100.0$) and 2D Spatial Pooling ($3 \times 3$ average pooling supporting arbitrary aspect ratios).
   - Connector projection ($768 \to 512$) mapping pooled visual soft tokens into the text sequence.
   - **Selective Encoder Loading** (`enable_vision = false` skips vision tower allocation/compilation, saving ~700 MB VRAM).
   - End-to-end multimodal joint encoding (`encode_multimodal`) and standalone image encoding (`encode_image`).
   - High-fidelity Catmull-Rom bicubic downsampling aligned with PyTorch torchvision.

3. **Phase 3 (Audio Encoder ~300M)**:
   - Audio conformer tower (`Gemma4AudioModel`, subsampling conv, 12 layers, chunked attention).
   - Connector projection ($1536 \to 512$) interfacing with the shared sequence.

---

## 2. Text Model Hyperparameters (`text_config`)

Derived directly from `google/embeddinggemma-2/config.json`:

| Parameter | Value | Description |
|---|---|---|
| `vocab_size` | 262,144 | BPE vocabulary size with byte fallback tokens |
| `hidden_size` | 512 | Core transformer hidden dimensionality ($H$) |
| `intermediate_size` | 2,048 | Gated MLP intermediate dimension ($I$) |
| `num_hidden_layers` | 24 | Total transformer blocks |
| `num_attention_heads` | 4 | Number of Query attention heads ($H_q$) |
| `num_key_value_heads` | 2 | Key/Value heads in GQA layers (group ratio = 2) |
| `head_dim` | 256 | Default per-head channel dimension (256 for layers $\neq$ full) |
| `full_attention_layers`| [5, 11, 17, 23] | 0-indexed layers 5, 11, 17, 23 are full attention |
| `full_attn.head_dim` | 512 | Per-head dimension on full attention layers |
| `full_attn.kv_heads` | 1 | KV heads on full attention layers (MHA/MQA ratio = 4) |
| `sliding_window` | 512 | Window size for sliding attention layers (all layers except 5, 11, 17, 23) |
| `rope_theta_sliding` | 10,000.0 | RoPE base frequency for sliding attention |
| `rope_theta_full` | 1,000,000.0 | RoPE base frequency for global full attention |
| `rms_norm_eps` | $1 \times 10^{-6}$ | Epsilon for numerical stability in RMSNorm |
| `hidden_activation` | `gelu_pytorch_tanh` | GeGLU activation approximation |
| `embedding_dim` | 768 | Final shared multimodal vector dimension |
| `max_seq_len` | 8,192 | Standard context length (architecture ceiling 262,144) |

---

## 3. Mathematical Operations & Kernel Specifications

All compute kernels are implemented in LuisaCompute DSL and compiled into native GPU shaders across DirectX 12, Vulkan, and CUDA.

### 3.1 RMSNorm (`kernels_norm.h/.cpp`)

Following Gemma 2 conventions, Root Mean Square Normalization scales weights with a $(1.0 + \gamma)$ offset:

$$
\text{RMS}(x) = \sqrt{\frac{1}{H} \sum_{i=1}^H x_i^2 + \epsilon}
$$

$$
y_i = \frac{x_i}{\text{RMS}(x)} \times (1.0 + \gamma_i)
$$

Each thread block uses an efficient shared-memory halving tree reduction ensuring bit-identical execution across hardware backends and the fallback CPU runner.

### 3.2 Rotary Position Embedding (RoPE) (`kernels_rope.h/.cpp`)

Applied to Query ($Q$) and Key ($K$) vectors prior to dot-product attention using the `rotate_half` layout:

$$
\theta_c = \text{pos} \cdot \theta_{\text{base}}^{-\frac{2c}{D}}, \quad c \in [0, D/2)
$$

$$
q'_c = q_c \cos(\theta_c) - q_{c + D/2} \sin(\theta_c)
$$

$$
q'_{c + D/2} = q_{c + D/2} \cos(\theta_c) + q_c \sin(\theta_c)
$$

Supports runtime switching between $\theta_{\text{base}} = 10^4$ (sliding layers) and $\theta_{\text{base}} = 10^6$ (full layers).

### 3.3 Grouped-Query Attention (GQA) (`kernels_attention.h/.cpp`)

- $H_q = 4, H_{kv} = 2$ on sliding layers ($g = 2$); $H_q = 4, H_{kv} = 1$ on full layers ($g = 4$).
- For each Query head $h_q$, Key/Value vectors are fetched from group $h_{kv} = \lfloor h_q / g \rfloor$.

**Attention logit calculation:**

$$
\text{logits}_{qi, ki} = \frac{Q_{qi, h_q} \cdot K_{ki, h_{kv}}}{\sqrt{D}} + \text{mask}_{qi, ki}
$$

**Sliding window masking:**

$$
\text{mask}_{qi, ki} = \begin{cases} 0 & \text{if } |qi - ki| \le W \text{ and } \text{valid}[ki] = 1 \\ -\infty & \text{otherwise} \end{cases}
$$

**Softmax and projection:**

$$
A_{qi, ki} = \text{softmax}(\text{logits})_{qi, ki}, \quad O_{qi, h_q} = \sum_{ki} A_{qi, ki} \cdot V_{ki, h_{kv}}
$$

### 3.4 GeGLU Feed-Forward Network (`kernels_mlp.h/.cpp`)

Gemma utilizes Gated GELU with tanh approximation:

$$
\text{GELU}_{\text{tanh}}(z) = 0.5 \cdot z \cdot \left(1.0 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot (z + 0.044715 \cdot z^3)\right)\right)
$$

$$
\text{FFN}(x) = W_{\text{down}} \Big(\text{GELU}_{\text{tanh}}(W_{\text{gate}} x) \odot (W_{\text{up}} x)\Big)
$$

where $W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$ correspond to the gate, up, and down projection weights (`gate_proj`, `up_proj`, `down_proj`).

### 3.5 Mask-Aware Mean Pooling, Projection & MRL (`kernels_pool.h/.cpp`)

1. **Mask-Aware Mean Pooling**:

$$
\bar{h}_b = \frac{\sum_{s=0}^{S-1} \text{mask}_{b, s} \cdot h_{b, s}}{\max\left(\sum_{s=0}^{S-1} \text{mask}_{b, s}, 1\right)}
$$

2. **Dense Projection**:

$$
p_b = \bar{h}_b \cdot W_{\text{proj}} \quad (\text{shape: } [1, 512] \times [512, 768] \to [1, 768])
$$

3. **Matryoshka Representation Learning (MRL) Truncation**:
   Supports dynamic truncation to $d \in \{768, 512, 256, 128\}$.

4. **$L_2$ Normalization**:

$$
\hat{e}_b = \frac{p_b[0:d]}{\max(\|p_b[0:d]\|_2, 10^{-12})}
$$

### 3.6 Vision Transformer & Spatial Pooling Kernels (`kernels_vision.h/.cpp`, `image_processor.h/.cpp`)

1. **Catmull-Rom Bicubic Resampling & Normalization**:
   High-quality image preprocessing via Catmull-Rom cubic splines (`STBIR_FILTER_CATMULLROM`) mathematically aligned with PyTorch torchvision's antialiased bicubic interpolation. The input pixel values $[0, 255]$ are normalized to $[-1.0, 1.0]$:

$$
x_{\text{norm}} = \frac{x}{127.5} - 1.0
$$

2. **2D Axial Rotary Position Embedding**:
   ViT patches of size $16 \times 16$ are projected into 768 dimensions. For an arbitrary 2D grid $(H_p \times W_p)$, position frequencies along vertical ($h$) and horizontal ($w$) axes are computed with $\theta_{\text{base}} = 100.0$:

$$
\theta_{h, c} = h \cdot \theta_{\text{base}}^{-\frac{4c}{D}}, \quad \theta_{w, c} = w \cdot \theta_{\text{base}}^{-\frac{4c}{D}}, \quad c \in [0, D/4)
$$

   Half of the channel budget encodes row coordinates, and the remaining half encodes column coordinates.

3. **2D Dynamic Spatial Average Pooling**:
   To compress visual sequence length while preserving 2D topological locality, each non-overlapping $3 \times 3$ grid of patch representations is averaged into a single visual soft token:

$$
T_{\text{soft}}(y, x) = \frac{1}{9} \sum_{dy=0}^2 \sum_{dx=0}^2 T_{\text{patch}}(3y + dy, 3x + dx)
$$

   For an image of arbitrary dimensions ($W \times H$), $W_p = W / 16$, $H_p = H / 16$, yielding $W_{\text{soft}} = W_p / 3$ and $H_{\text{soft}} = H_p / 3$.

4. **Connector Projection & Multimodal Token Scatter**:
   Soft tokens are mapped into language hidden dimension ($512$) via a linear projection $W_{\text{connector}} \in \mathbb{R}^{768 \times 512}$, then scattered into the language backbone sequence buffer at designated `<image>` token locations.

---

## 4. Binary Model Package Format (`model.mmegm`)

To eliminate runtime serialization overhead and Python dependencies, weights are packaged into an aligned binary format loaded directly via OS memory mapping.

### File Header (`32 bytes`)
| Offset | Size | Type | Name | Description |
|---|---|---|---|---|
| 0 | 4 | `char[4]` | `magic` | `'E', 'G', 'M', '2'` |
| 4 | 4 | `uint32_t` | `version` | Format version (currently `1`) |
| 8 | 4 | `uint32_t` | `precision` | `0` = fp32, `1` = bf16, `2` = fp16 |
| 12 | 4 | `uint32_t` | `num_tensors` | Number of tensor descriptors |
| 16 | 4 | `uint32_t` | `num_layers` | 24 |
| 20 | 4 | `uint32_t` | `hidden_size` | 512 |
| 24 | 4 | `uint32_t` | `intermediate_size`| 2048 |
| 28 | 4 | `uint32_t` | `embedding_dim` | 768 |

### Tensor Descriptor Table (`48 bytes` per entry)
| Field | Type | Description |
|---|---|---|
| `name_hash` | `uint64_t` | FNV-1a 64-bit hash of canonical tensor name |
| `offset` | `uint64_t` | Byte offset from file start (256-byte aligned) |
| `size_bytes` | `uint64_t` | Tensor byte size |
| `rank` | `uint32_t` | Number of dimensions (1 or 2) |
| `dims[4]` | `uint32_t[4]`| Dimension lengths (outermost first) |

### Canonical Tensor Names
- **Text Backbone**:
  - `emb.word`: Token embeddings `[262144, 512]`
  - `layer{i}.input_norm`: RMSNorm weight `[512]`
  - `layer{i}.q_proj`: Query weight `[512, 1024]` (or `[512, 2048]` for full attention)
  - `layer{i}.k_proj`: Key weight `[512, 512]`
  - `layer{i}.v_proj`: Value weight `[512, 512]`
  - `layer{i}.out_proj`: Output projection `[1024, 512]` (or `[2048, 512]`)
  - `layer{i}.post_attn_norm`: RMSNorm weight `[512]`
  - `layer{i}.pre_ffn_norm`: RMSNorm weight `[512]`
  - `layer{i}.gate_proj`: Gated MLP weight `[512, 2048]`
  - `layer{i}.up_proj`: Gated MLP weight `[512, 2048]`
  - `layer{i}.down_proj`: Gated MLP down weight `[2048, 512]`
  - `layer{i}.post_ffn_norm`: RMSNorm weight `[512]`
  - `final_norm`: Post-backbone RMSNorm weight `[512]`
  - `embedding_proj`: Output projection `[512, 768]`
- **Vision Tower (Phase 2)**:
  - `vision.emb.patch`: Linear patch embedding projection `[768, 768]`
  - `vision.layer{i}.input_norm`: ViT layer RMSNorm `[768]`
  - `vision.layer{i}.q_proj`: ViT query projection `[768, 768]`
  - `vision.layer{i}.k_proj`: ViT key projection `[768, 768]`
  - `vision.layer{i}.v_proj`: ViT value projection `[768, 768]`
  - `vision.layer{i}.out_proj`: ViT attention output projection `[768, 768]`
  - `vision.layer{i}.post_attn_norm`: ViT post-attention RMSNorm `[768]`
  - `vision.layer{i}.gate_proj`: ViT gated MLP weight `[768, 3072]`
  - `vision.layer{i}.up_proj`: ViT gated MLP weight `[768, 3072]`
  - `vision.layer{i}.down_proj`: ViT gated MLP down weight `[3072, 768]`
  - `vision.final_norm`: ViT post-encoder RMSNorm `[768]`
  - `vision.connector`: Linear projection mapping vision hidden dim to language hidden dim `[768, 512]`

All linear weights are pre-transposed during offline packaging ($B[K, N]$ layout) so GPU matrix multiply reads coalesce consecutively along $N$.

---

## 5. Tokenizer Architecture

A zero-external-dependency, pure C++ Gemma BPE Tokenizer (`tokenizer.h/.cpp`):
- Uses embedded `yyjson` for fast parsing of HuggingFace `tokenizer.json`.
- Implements Byte-Pair Encoding (BPE) with prefix merge tables.
- Supports byte-level fallback tokens (`<0x00>` through `<0xFF>`).
- Prepends special token `<bos>` (id `2`) and manages padding (`<pad>` id `0`).

---

## 6. Directory Structure

```
examples/mm_embedding/
├── README.md                          # Architecture specification & usage documentation
├── CMakeLists.txt                     # CMake build configuration
├── xmake.lua                          # XMake build configuration
├── example_mm_embed_text.cpp          # Interactive text embedding demo
├── example_mm_embed_vision.cpp        # Interactive image and multimodal query demo
├── example_mm_embed_test.cpp          # Automated unit & regression test suite
├── example_mm_embed_bench.cpp         # Multimodal vision evaluation & benchmark harness
├── embed/
│   ├── executor.h / .cpp              # Forward pass graph recorder & memory manager
│   ├── package.h / .cpp               # Memory-mapped model.mmegm package loader
│   ├── session.h / .cpp               # High-level thread-safe Session API & Selective Loading
│   ├── tokenizer.h / .cpp             # Pure C++ Gemma BPE tokenizer
│   ├── image_processor.h / .cpp       # Catmull-Rom bicubic image downsampler & normalizer
│   └── kernels/
│       ├── kernels_norm.h / .cpp      # RMSNorm and L2Norm kernels
│       ├── kernels_rope.h / .cpp      # RoPE rotary embedding kernels (1D & 2D Axial)
│       ├── kernels_mlp.h / .cpp       # GeGLU activation & residual add kernels
│       ├── kernels_attention.h / .cpp # GQA attention kernels (scores, softmax, PV)
│       ├── kernels_pool.h / .cpp      # Mask-aware Mean Pooling + MRL + L2Norm
│       ├── kernels_vision.h / .cpp    # ViT pixel scaling, 2D pooling & soft token scatter
│       └── kernels_gemm.h / .cpp      # Tiled & register-blocked GEMM kernels
└── tools/
    ├── pack_model.py                  # Offline safetensors-to-EGM converter
    ├── dump_golden.py                 # PyTorch text golden generation script
    ├── dump_vision_golden.py          # PyTorch multimodal vision golden generation script
    ├── build_diverse_dataset.py       # Diverse 8-domain multimodal benchmark dataset builder
    ├── generate_test_multimodal_data.py # Multimodal synthetic test fixture generator
    ├── pack_model.cpp                 # C++ packaging & validation tool
    └── dump_golden.cpp                # C++ golden intermediate verification tool
```

---

## 7. Build and Execution

### Using XMake
```bash
# Configure (Windows x64 with MSVC toolchain)
xmake f -p windows -a x64 -m release

# Build engine library, demos, tests, and benchmark harness
xmake build lc-example-mm-embedding example_mm_embed_text example_mm_embed_vision example_mm_embed_test example_mm_embed_bench

# Run automated unit and regression test suite
xmake run example_mm_embed_test dx model.mmegm tokenizer.json

# Run text embedding demo
xmake run example_mm_embed_text dx model.mmegm tokenizer.json

# Run interactive image and multimodal embedding demo
xmake run example_mm_embed_vision dx model.mmegm tokenizer.json input.jpg "A high resolution photo"

# Run multimodal vision benchmark harness
xmake run example_mm_embed_bench dx model.mmegm tokenizer.json data/diverse_dataset.json data/golden_diverse_vision.json
```

### Using CMake
```bash
cmake -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build build --target example_mm_embed_text example_mm_embed_vision example_mm_embed_test example_mm_embed_bench
```

### Selective Encoder Loading

In resource-constrained environments or text-only inference services, disable the vision encoder to save approximately **700 MB of VRAM** and avoid ViT kernel compilation overhead:

```cpp
luisa::mm_embed::SessionConfig config;
config.enable_vision = false; // Disables vision tower allocation and kernel compilation

luisa::mm_embed::Session session(context, device, stream, package, tokenizer, config);
// Full text embedding capabilities remain available with zero numerical difference
```

---

## 8. Verification & Multi-Backend Parity

The implementation has been strictly benchmarked and verified across **DirectX 12**, **CUDA**, and **Vulkan** against official PyTorch `SentenceTransformer` outputs:

- **Mathematical Parity**: Average cosine similarity $> 0.9997$ across diverse visual domains (minimum $> 0.9982$).
- **Hard-Negative Discrimination**: Semantic margin $\Delta(\text{pos}, \text{neg}) \ge +0.1455$ on challenging cross-domain distractors.
- **Bilingual Cross-Lingual Alignment**: Multilingual consistency score of $0.8397$ between English and Chinese queries.
- **Selective Loading Invariance**: Text embedding cosine difference between `enable_vision = true` and `enable_vision = false` is strictly $0.0$.


