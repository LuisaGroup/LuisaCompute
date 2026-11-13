#pragma once
// OCR engine -- shared Phase 7 vertical-slice configuration (device-safe).
//
// Plain geometry/config data for the Phase 7 custom dense/layout model
// variants (TeleOCR, GLM-OCR, MinerU, HunyuanOCR). The EXECUTOR that consumes
// this data lives in runtime/p7_vlm_executor.*; each model module owns its own
// identity/tensor-name/output contract and config parser and instantiates the
// shared graph through an explicit configuration -- never by guessing from an
// outer model_type string.
//
// This header pulls NO api.h/model_recipe.h (host AttentionMaskKind clash) and
// NO ops headers, so both host and device TUs may include it.
//
// HONESTY (plan §1/§6): every dimension here is a DECLARED SYNTHETIC CONTRACT.
// No official checkpoint is pinned; real artifacts fail closed through the
// per-model gates. See MODELS.md.
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/models/qwen2_5_vl.h"// DenseDecoderConfig / RopePairLayout / ExpectedTensor
#include "ocr/package.h"
#include "ocr/runtime/memory_budget.h"
#include "ocr/status.h"

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// Vision tower conventions (explicit, never inferred)
// ---------------------------------------------------------------------------

// Vision attention mask convention. WINDOW_GLOBAL builds a host custom block
// mask: queries in a GLOBAL layer see every patch of the image; queries in a
// WINDOW layer see only patches of their own spatial window block. This is a
// BIDIRECTIONAL vision convention (never causal); the decoder stays causal.
enum class P7VisionMaskKind : uint32_t {
    bidirectional = 0,
    window_global = 1,
};

// Vision rotary convention.
//   none   : no RoPE (absolute/learned position embeddings instead).
//   axial2d: adjacent-pair 2D RoPE over (height, width) patch positions.
enum class P7VisionRopeKind : uint32_t {
    none = 0,
    axial2d = 1,
};

// Vision tower geometry. All fields concrete before use. The block body is:
// pre-LayerNorm -> QKV (bias per qkv_bias) -> optional per-head QK-Norm ->
// optional 2D RoPE -> masked bidirectional attention -> o-proj (+bias) +
// residual -> post-LayerNorm -> fc1 (+bias, activation) -> fc2 (+bias) +
// residual. The MLP activation is explicit (GELU-erf vs SiLU).
struct P7VisionConfig {
    uint32_t num_layers{0};
    uint32_t hidden_size{0};
    uint32_t num_heads{0};
    uint32_t head_dim{0};// heads * head_dim == hidden_size
    uint32_t intermediate_size{0};
    uint32_t patch_size{0};
    uint32_t channels{3};
    float layer_norm_eps{1e-6f};
    bool qkv_bias{true};
    bool qk_norm{false};
    // 0 = GELU-erf, 1 = SiLU (the vision MLP activation; explicit).
    uint32_t mlp_act{0};
    P7VisionRopeKind rope_kind{P7VisionRopeKind::axial2d};
    float rope_theta{10000.0f};
    uint32_t rope_height_pairs{0};// axial2d: pairs using the height axis
    P7VisionMaskKind mask_kind{P7VisionMaskKind::bidirectional};
    // window_global convention: the window block edge in PATCH units and the
    // 0-based indices of the GLOBAL layers (every other layer is windowed).
    uint32_t window_patch_size{0};
    luisa::vector<uint32_t> global_layers;
    // Optional learned absolute position embedding added after patch embed.
    // 0 disables it; otherwise the table has position_rows rows of hidden_size
    // and the patch grid must fit (grid_h*grid_w <= position_rows).
    uint32_t position_rows{0};
    // Additive Phase 8: decomposed relative-position bias (ViTDet/SAM, GOT).
    // When nonzero, each attention layer adds a learned per-head bias
    // B[h, dy+window-1, dx+window-1] indexed by the patch grid delta. The table
    // has num_heads * (2*window-1)^2 entries. This is DISTINCT from RoPE and
    // from absolute position embeddings; it is never substituted for them.
    uint32_t rel_pos_window{0};

    [[nodiscard]] uint32_t patch_dim() const noexcept { return channels * patch_size * patch_size; }
    [[nodiscard]] Status validate() const;
};

// ---------------------------------------------------------------------------
// Connector conventions (explicit, never inferred)
// ---------------------------------------------------------------------------

//   mlp         : LayerNorm + 2-layer MLP over the patch rows (no reduction).
//   spatial_mlp : spatial m x m token packing + LayerNorm + 2-layer MLP
//                 (Qwen-style merger / TeleOCR aligner).
//   unshuffle_mlp : 2x2 pixel-unshuffle (NCHW channel-fold) + LayerNorm +
//                 2-layer MLP (MinerU). NOT Paddle's adaptive pool.
//   adaptive_mlp : adaptive average pool to a token-budget grid + LayerNorm +
//                 2-layer MLP (Hunyuan). Distinct from spatial/unshuffle.
//   downsample_mlp : learned strided token downsampling (conv-style gather of
//                 stride x stride spatial neighbors -> linear) + LayerNorm +
//                 2-layer MLP (GLM CogViT connector).
enum class P7ConnectorKind : uint32_t {
    mlp = 0,
    spatial_mlp = 1,
    unshuffle_mlp = 2,
    adaptive_mlp = 3,
    downsample_mlp = 4,
};

struct P7ConnectorConfig {
    P7ConnectorKind kind{P7ConnectorKind::mlp};
    uint32_t intermediate_size{0};
    float layer_norm_eps{1e-6f};
    // spatial_mlp / unshuffle_mlp / downsample_mlp: the spatial factor m
    // (unshuffle is fixed 2x2 by the MinerU contract; downsample stride).
    uint32_t spatial_factor{1};
    // adaptive_mlp: the visual-token budget for the adaptive pool.
    uint32_t max_visual_tokens{0};
    // Additive Phase 8: connector normalization convention. false = LayerNorm
    // (the Phase 7 default, weight+bias); true = RMSNorm (weight only, no bias)
    // -- the Qwen2.5-VL merger convention (olmOCR-2). The two are DISTINCT and
    // never substituted for each other.
    bool rms_norm{false};

    [[nodiscard]] Status validate() const;
};

// ---------------------------------------------------------------------------
// Packing (native-resolution, aspect-preserving; shared host math)
// ---------------------------------------------------------------------------

struct P7PackingSpec {
    uint32_t patch_size{0};
    uint64_t min_pixels{0};
    uint64_t max_pixels{0};
    // The connector's spatial factor: the patch grid must be divisible by it
    // per axis for spatial/unshuffle/downsample connectors (1 = no constraint).
    uint32_t grid_multiple{1};

    [[nodiscard]] Status validate() const;
};

struct P7PackedImage {
    uint32_t source_width{0};
    uint32_t source_height{0};
    uint32_t resized_width{0};
    uint32_t resized_height{0};
    uint32_t grid_h{0};
    uint32_t grid_w{0};
    uint64_t patch_count{0};
    // Connector output token grid + count (filled by the packing helper from
    // the connector kind).
    uint32_t out_h{0};
    uint32_t out_w{0};
    uint64_t visual_tokens{0};
    float scale_x{1.0f};
    float scale_y{1.0f};
};

// Pack one image size (no pixels touched). Fails closed on degenerate sizes,
// invalid spec, or a pixel budget that cannot be satisfied. The output token
// grid is derived from the connector kind + spatial factor / token budget.
[[nodiscard]] Result<P7PackedImage> p7_pack_image(uint32_t source_width, uint32_t source_height,
                                                  const P7PackingSpec &spec, const P7ConnectorConfig &connector);

// Vision 2D RoPE positions for one packed image: [patch_count * 2] int32
// (height, width) per patch, row-major. Empty when the vision rope is none.
[[nodiscard]] luisa::vector<int32_t> p7_vision_positions(const P7PackedImage &packed);
[[nodiscard]] luisa::vector<uint32_t> p7_vision_axis_of_pair(const P7VisionConfig &vision);

// One image input: decoded HWC uint8 RGB source pixels + packing metadata.
// Device-safe plain data (shared by the executor and the host preprocessing
// bridge, which must NOT include the device executor header).
struct P7ImageInput {
    luisa::vector<uint8_t> pixels;
    uint32_t source_width{0};
    uint32_t source_height{0};
    P7PackedImage packed;
};

// ---------------------------------------------------------------------------
// Text decoder position convention (explicit, never inferred)
// ---------------------------------------------------------------------------

//   rope1d           : plain 1D text positions (GLM partial-rotary supported).
//   mrope_adjacent   : adjacent-pair M-RoPE (Qwen2-VL / MinerU Qwen2 decoder).
//   mrope_interleaved: interleaved M-RoPE (Qwen3 / TeleOCR decoder).
//   xd               : XD-RoPE over (text, height, width, time) (Hunyuan).
enum class P7TextPosKind : uint32_t {
    rope1d = 0,
    mrope_adjacent = 1,
    mrope_interleaved = 2,
    xd = 3,
};

// The full slice configuration consumed by P7VlmExecutor.
struct P7SliceConfig {
    P7VisionConfig vision;
    P7ConnectorConfig connector;
    DenseDecoderConfig text;
    bool tie_word_embeddings{true};
    P7PackingSpec packing;
    P7TextPosKind text_pos{P7TextPosKind::rope1d};
    // M-RoPE / XD-RoPE per-pair axis sections. For mrope_*: [t,h,w] summing to
    // head_dim/2. For xd: [text,h,w,time] summing to head_dim/2. Ignored for
    // rope1d.
    luisa::vector<uint32_t> pos_sections;
    // GLM partial rotary factor (text only; 1/1 = full rotary).
    uint32_t partial_rotary_num{1};
    uint32_t partial_rotary_den{1};
    // Additive Phase 8: explicit Qwen2.5-VL text contract switch. When true, the
    // text decoder uses QKV bias + NO QK-Norm even with rotate-half M-RoPE
    // (the Qwen2.5-VL / olmOCR-2 convention). When false (default), the Phase 7
    // behavior is unchanged: mrope_interleaved selects the Qwen3 QK-Norm/no-bias
    // contract. This is NEVER inferred from a model_type string.
    bool text_qwen2_5_contract{false};
    // Additive Phase 8: per-linear decoder weight storage (W4A32). Default fp32
    // reproduces the exact Phase 7 path. int4_* selects packed INT4 decoder
    // linears + a standalone packed lm_head (tied embeddings refused). The
    // vision tower / connector / embeddings / norms / biases stay FP32.
    DenseLinearStorage linear_storage{DenseLinearStorage::fp32};
    uint32_t linear_group_size{0};
    // Weight component tags.
    luisa::string vision_component{"vision"};
    luisa::string decoder_component{"decoder"};
    // Normalization (RGB, (pixel/255 - mean)/std).
    luisa::vector<float> mean{0.5f, 0.5f, 0.5f};
    luisa::vector<float> std_dev{0.5f, 0.5f, 0.5f};
    uint32_t max_vision_patches{4096};
    uint32_t prefill_chunk_rows{256};
    // Opt-in dense residual+RMSNorm fusion A/B. Default preserves the
    // reference command sequence; forwarded to the embedded text executor.
    bool enable_fused_residual_rms_norm{false};
    // Hidden-sized cooperative RMSNorm; false restores the reference kernel.
    bool enable_cooperative_norm{true};
    MemoryLimits memory;

    // Derived text RoPE axis count.
    [[nodiscard]] uint32_t text_rope_axes() const noexcept {
        return text_pos == P7TextPosKind::xd ? 4u
               : text_pos == P7TextPosKind::rope1d ? 1u
                                                   : 3u;
    }
    // Rotated elements per head for the GLM partial-rotary factor.
    [[nodiscard]] uint32_t rotary_dim() const noexcept {
        return partial_rotary_den == 0u ? 0u
               : static_cast<uint32_t>((static_cast<uint64_t>(text.head_dim) * partial_rotary_num) / partial_rotary_den);
    }
    // axis_of_pair map for the text decoder (length head_dim/2).
    [[nodiscard]] luisa::vector<uint32_t> text_axis_of_pair() const;
    [[nodiscard]] Status validate() const;
};

// Build the multimodal prompt position ids for a prompt whose image rows are
// [mm_start, mm_start+mm_count) with the packed image grid. Returns
// [prompt_len * axes] int32 plus the FIRST generated token's axis vector.
// The convention is explicit per text_pos kind (see the .cpp for the exact
// per-axis rules); it is NEVER inferred from a model_type string.
struct P7PositionIds {
    luisa::vector<int32_t> ids;// [seq_len * axes]
    uint32_t seq_len{0};
    luisa::vector<int32_t> next_position;// [axes]
};
[[nodiscard]] Result<P7PositionIds> p7_build_position_ids(
    const P7SliceConfig &config, uint32_t prompt_len, uint32_t mm_start, uint32_t mm_count,
    const P7PackedImage &packed);

// ---------------------------------------------------------------------------
// Canonical tensor-name contract (device-free; shared by the executor and the
// host model modules). Every model sets its OWN prefixes. Linear weights are
// packed [K,N] (`kn`).
// ---------------------------------------------------------------------------

struct P7TensorNames {
    DenseDecoderTensorNames text;
    luisa::string visual_prefix{"visual."};

    [[nodiscard]] Status validate() const;
    [[nodiscard]] luisa::string patch_embed_weight() const;
    [[nodiscard]] luisa::string patch_embed_bias() const;
    [[nodiscard]] luisa::string position_embed() const;
    // Additive Phase 8: decomposed relative-position bias table for layer l
    // (ViTDet/SAM). Present only when vision.rel_pos_window != 0.
    [[nodiscard]] luisa::string block_rel_pos_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string block_prefix(uint32_t layer) const;
    [[nodiscard]] luisa::string block_norm1_weight(uint32_t layer) const;
    [[nodiscard]] luisa::string block_norm1_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string block_norm2_weight(uint32_t layer) const;
    [[nodiscard]] luisa::string block_norm2_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string block_q_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string block_k_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string block_v_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string block_o_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string block_q_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string block_k_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string block_v_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string block_o_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string block_q_norm(uint32_t layer) const;
    [[nodiscard]] luisa::string block_k_norm(uint32_t layer) const;
    [[nodiscard]] luisa::string block_fc1(uint32_t layer) const;
    [[nodiscard]] luisa::string block_fc1_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string block_fc2(uint32_t layer) const;
    [[nodiscard]] luisa::string block_fc2_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string connector_prefix() const;
    [[nodiscard]] luisa::string connector_ln_weight() const;
    [[nodiscard]] luisa::string connector_ln_bias() const;
    [[nodiscard]] luisa::string connector_fc1() const;
    [[nodiscard]] luisa::string connector_fc1_bias() const;
    [[nodiscard]] luisa::string connector_fc2() const;
    [[nodiscard]] luisa::string connector_fc2_bias() const;
    [[nodiscard]] luisa::string connector_down_weight() const;
    [[nodiscard]] luisa::string connector_down_bias() const;
};

// Additive Phase 8 review (S2/GOT): checked host footprint plan for ONE slice
// (vision + connector + decoder weights, combined workspace arenas, KV cache).
// Mirrors the executor arenas exactly (256-byte region alignment) so model
// preflights can be asserted EQUAL to measured device bytes. FP32 or W4A32
// decoder linears; the tied-LM-head transpose is counted as decoder weight
// bytes (it is executor-owned residency). No allocation occurs here.
struct P7SliceFootprint {
    uint64_t vision_weight_bytes{0};
    uint64_t decoder_weight_bytes{0};
    uint64_t workspace_bytes{0};// vision arena + decoder arena (combined)
    uint64_t kv_cache_bytes{0}; // includes DecodeState metadata (16 B, batch 1)
    uint64_t total_device_bytes{0};
};
[[nodiscard]] Result<P7SliceFootprint> p7_plan_slice_footprint(const P7SliceConfig &config, uint32_t max_seq_len);

// Full expected tensor inventory (vision + connector + text decoder). The text
// part follows the ACTIVE decoder contract (Qwen2-style QKV bias vs Qwen3-style
// QK-Norm/no-bias), selected by the slice config.
[[nodiscard]] luisa::vector<ExpectedTensor> p7_expected_tensors(const P7SliceConfig &config,
                                                               const P7TensorNames &names);
[[nodiscard]] Status p7_validate_tensors(luisa::span<const PackageTensor> tensors,
                                         luisa::span<const ExpectedTensor> expected);

}// namespace luisa::ocr
