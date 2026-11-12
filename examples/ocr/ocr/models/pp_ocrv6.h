#pragma once
// OCR engine -- PP-OCRv6 CNN graph contract (Phase 5). Host-only.
//
// PP-OCRv6 is a conventional detector -> crop -> recognizer -> CTC pipeline:
//   detector:   LCNetV4 backbone + RepLKFPN neck + DB head
//   recognizer: LCNetV4 backbone + LightSVTR sequence head + CTC
// This module owns the model-specific geometry, the exact per-size tensor
// maps (tiny/small/medium are SEPARATE contracts validated independently),
// and the inference-only RepLK / Conv+BN folding helpers.
//
// INFERENCE ONLY. There is NO NRTR attention decoder, NO dropout, NO loss,
// and NO training-time reparameterization branch anywhere. RepLK blocks are
// folded to a single large-kernel conv at pack/load time; Conv+BN is folded
// to a single conv. The folding helpers below are pure math over host weight
// spans and are tested against an independent two-step (conv-then-BN) oracle.
//
// HONESTY (plan §1/§6): NO official PP-OCRv6 checkpoint has been pinned,
// inspected, or run. Every dimension / tensor name here is a DECLARED
// SYNTHETIC CONTRACT exercised by the synthetic CNN fixture
// (runtime/cnn_executor.*). Real packages fail closed through the registry /
// Session and pp_ocrv6_validate_real_artifact(). The official gates are
// recorded in MODELS.md. Nothing here validates a real artifact, and no
// checkpoint hash / official tensor name is invented as a fact.
//
// Host-only header: no DSL/runtime device includes.
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/models/pp_ocr_size.h"
#include "ocr/models/qwen2_5_vl.h"// ExpectedTensor + validate_decoder_weight_desc
#include "ocr/package.h"
#include "ocr/status.h"

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// Geometry (declared synthetic contract; per-size, never shared blindly)
// ---------------------------------------------------------------------------

// One folded 2D convolution stage (Conv+BN already folded). NCHW weights
// [out_c, in_c/groups, kh, kw]; bias [out_c]. `act` is the post-conv
// activation: 0=none, 1=relu, 2=relu6, 3=hard_swish.
struct PpConvStage {
    uint32_t in_c{0};
    uint32_t out_c{0};
    uint32_t kh{0};
    uint32_t kw{0};
    uint32_t stride_h{0};
    uint32_t stride_w{0};
    uint32_t pad_h{0};
    uint32_t pad_w{0};
    uint32_t dil_h{1};
    uint32_t dil_w{1};
    uint32_t groups{1};
    uint32_t act{0};

    [[nodiscard]] Status validate() const;
    [[nodiscard]] uint64_t weight_elements() const noexcept {
        return static_cast<uint64_t>(out_c) * (in_c / groups) * kh * kw;
    }
};

// Detector: LCNetV4-style backbone (a short list of conv stages, including an
// asymmetric-stride stage and a large depthwise stage), a RepLKFPN neck
// (lateral 1x1 + upsample + fusion + a folded RepLK large-kernel conv), and a
// DB head (conv -> sigmoid probability map).
struct PpDetectorConfig {
    uint32_t in_channels{3};
    // Backbone stages, applied in order. The LAST stage's output feeds the
    // FPN; an earlier stage provides the lateral skip (see lateral_stage).
    luisa::vector<PpConvStage> backbone;
    uint32_t lateral_stage{0};// index into backbone providing the FPN skip
    // RepLKFPN neck.
    PpConvStage lateral_conv;// 1x1 on the skip feature
    PpConvStage replk_conv;  // folded large-kernel conv on the fused feature
    // DB head.
    PpConvStage db_conv;// 1x1 -> 1 channel logit map (sigmoid applied after)
    // The detector runs at a fixed input size (multiple of the total stride).
    uint32_t input_width{0};
    uint32_t input_height{0};

    [[nodiscard]] Status validate() const;
    // Total output stride of the backbone (product of stage strides).
    [[nodiscard]] uint32_t stride_h() const noexcept;
    [[nodiscard]] uint32_t stride_w() const noexcept;
};

// Recognizer: LCNetV4-style backbone over a fixed-height line crop, a
// LightSVTR-like sequence reshape (collapse the height axis into channels,
// keep the width axis as the time/frame axis), a small SVTR mixing block
// (channel MLP applied per frame), and a CTC linear head.
struct PpRecognizerConfig {
    uint32_t in_channels{3};
    luisa::vector<PpConvStage> backbone;
    // Height-collapse factor r: the backbone output [C, H', W'] is reshaped
    // LightSVTR-style to [W', C*r] by grouping r height rows into the channel
    // axis (pixel-unshuffle along height only). H' must be divisible by r.
    // r == 1 is a plain height-collapse (H' must be 1).
    uint32_t height_collapse{1};
    // LightSVTR sequence mixing: a per-frame channel MLP (two linear layers
    // with a GELU between). Input width = collapsed_channels.
    uint32_t svtr_intermediate{0};
    // CTC head: linear collapsed_channels -> num_classes.
    uint32_t num_classes{0};
    uint32_t blank_id{0};
    // Fixed line-crop input size.
    uint32_t input_width{0};
    uint32_t input_height{0};

    [[nodiscard]] Status validate() const;
    [[nodiscard]] uint32_t stride_h() const noexcept;
    [[nodiscard]] uint32_t stride_w() const noexcept;
    // Backbone output spatial size for the fixed input.
    [[nodiscard]] Result<uint32_t> feature_height() const;
    [[nodiscard]] Result<uint32_t> feature_width() const;
    // Channel width after the height-collapse reshape (the SVTR/CTC input).
    [[nodiscard]] Result<uint32_t> collapsed_channels() const;
    // Frames produced for input_width (the CTC time axis == feature width).
    [[nodiscard]] Result<uint32_t> num_frames() const;
};

// Full per-size PP-OCRv6 config. tiny/small/medium each get their OWN
// geometry; nothing is shared across sizes without explicit validation.
struct PpOcrv6Config {
    PpOcrSize size{PpOcrSize::none};
    PpDetectorConfig detector;
    PpRecognizerConfig recognizer;

    [[nodiscard]] Status validate() const;
};

// The declared SMALL SYNTHETIC per-size configs. NOT real checkpoints.
[[nodiscard]] PpOcrv6Config pp_ocrv6_synthetic_config(PpOcrSize size);

// ---------------------------------------------------------------------------
// Tensor-name contract (per-size; exact inventory)
// ---------------------------------------------------------------------------

// Tensor names for the detector + recognizer components. Every name is a
// declared synthetic contract name (prefix "pp_ocrv6."). Linear weights are
// stored PACKED [K,N] (layout kn); conv weights are conv_oihw.
struct PpOcrv6TensorNames {
    luisa::string prefix{"pp_ocrv6."};

    [[nodiscard]] Status validate() const;
    // Detector backbone stage i: weight/bias (folded Conv+BN).
    [[nodiscard]] luisa::string det_backbone_w(uint32_t i) const;
    [[nodiscard]] luisa::string det_backbone_b(uint32_t i) const;
    [[nodiscard]] luisa::string det_lateral_w() const;
    [[nodiscard]] luisa::string det_lateral_b() const;
    [[nodiscard]] luisa::string det_replk_w() const;
    [[nodiscard]] luisa::string det_replk_b() const;
    [[nodiscard]] luisa::string det_db_w() const;
    [[nodiscard]] luisa::string det_db_b() const;
    // Recognizer backbone stage i.
    [[nodiscard]] luisa::string rec_backbone_w(uint32_t i) const;
    [[nodiscard]] luisa::string rec_backbone_b(uint32_t i) const;
    // LightSVTR per-frame MLP (packed [K,N]).
    [[nodiscard]] luisa::string rec_svtr_fc1_w() const;
    [[nodiscard]] luisa::string rec_svtr_fc1_b() const;
    [[nodiscard]] luisa::string rec_svtr_fc2_w() const;
    [[nodiscard]] luisa::string rec_svtr_fc2_b() const;
    // CTC head (packed [K,N]).
    [[nodiscard]] luisa::string rec_ctc_w() const;
    [[nodiscard]] luisa::string rec_ctc_b() const;
};

// Full expected tensor inventory for one component ("detector"/"recognizer").
[[nodiscard]] luisa::vector<ExpectedTensor> pp_ocrv6_expected_tensors(
    const PpOcrv6Config &config, const PpOcrv6TensorNames &names, luisa::string_view component);

// DEVICE-descriptor view of the same inventory: conv weights appear as
// row_major because the runtime's logical_desc maps every non-kn/nk package
// layout to row_major. Use THIS for installed-weight checks and
// pp_ocrv6_expected_tensors for ON-DISK package checks.
[[nodiscard]] luisa::vector<ExpectedTensor> pp_ocrv6_device_expected_tensors(
    const PpOcrv6Config &config, const PpOcrv6TensorNames &names, luisa::string_view component);

// Validate a package component's tensor table against the expected inventory
// (exact name/shape/layout; fp32 only for the Phase 5 baseline).
[[nodiscard]] Status pp_ocrv6_validate_tensors(luisa::span<const PackageTensor> tensors,
                                               luisa::span<const ExpectedTensor> expected);

// ---------------------------------------------------------------------------
// Inference-only folding helpers (pure host math; tested vs two-step oracle)
// ---------------------------------------------------------------------------

// Fold Conv+BN into a single conv. `weight` [out_c, in_c/groups, kh, kw],
// `bias` [out_c] (empty = zeros), BN params [out_c]. Returns the folded
// weight/bias:
//   scale[c] = gamma[c] / sqrt(var[c] + eps)
//   w'[c,...] = w[c,...] * scale[c]
//   b'[c] = (b[c] - mean[c]) * scale[c] + beta[c]
// All arithmetic is double, rounded to float. Validates sizes; var must be
// finite and >= 0, eps > 0.
[[nodiscard]] Status pp_fold_conv_bn(luisa::span<const float> weight, luisa::span<const float> bias,
                                     luisa::span<const float> bn_mean, luisa::span<const float> bn_var,
                                     luisa::span<const float> bn_gamma, luisa::span<const float> bn_beta,
                                     uint32_t out_c, uint64_t per_channel, float eps,
                                     luisa::vector<float> &out_weight, luisa::vector<float> &out_bias);

// Fold a RepLK structural block (large kernel + small kernel + identity BN
// branches) into ONE large-kernel conv. The small kernel is zero-padded into
// the center of the large kernel; the identity branch contributes a centered
// impulse scaled by its BN gamma. Inputs are the ALREADY Conv+BN-folded
// branches:
//   large_weight [C,1,K,K] (depthwise), large_bias [C]
//   small_weight [C,1,k,k] (depthwise, k<=K, odd), small_bias [C]
//   identity: BN-only branch -> impulse of weight gamma_id[c] at the center,
//             bias beta_id[c] (mean/var folded into the impulse scale).
// The folded weight is large_weight + pad(small_weight) + identity_impulse;
// the folded bias is large_bias + small_bias + identity_bias.
// Returns the merged large-kernel weight/bias. k and K must be odd, k<=K.
[[nodiscard]] Status pp_fold_replk(luisa::span<const float> large_weight, luisa::span<const float> large_bias,
                                   luisa::span<const float> small_weight, luisa::span<const float> small_bias,
                                   luisa::span<const float> identity_gamma, luisa::span<const float> identity_beta,
                                   uint32_t channels, uint32_t large_k, uint32_t small_k,
                                   luisa::vector<float> &out_weight, luisa::vector<float> &out_bias);

// ---------------------------------------------------------------------------
// Real-artifact gate (always fail-closed in Phase 5)
// ---------------------------------------------------------------------------

// ALWAYS returns unvalidated: no official PP-OCRv6 checkpoint/config has been
// qualified. The synthetic path is exercised only through the explicit
// synthetic config above (tests). Real packages never load through this gate.
[[nodiscard]] Status pp_ocrv6_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                     luisa::string_view config_fingerprint);

// Training-node guard: the PP-OCRv6 contract must never carry NRTR / dropout
// / loss / training reparameterization components. Checks a component-role
// list for forbidden markers (contract-level assertion, not a runtime filter).
[[nodiscard]] Status pp_ocrv6_check_no_training_nodes(luisa::span<const luisa::string> component_roles);

}// namespace luisa::ocr