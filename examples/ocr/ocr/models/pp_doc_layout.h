#pragma once
// OCR engine -- PP-DocLayoutV2/V3 layout detector contract (Phase 6).
//
// DEVICE-SAFE header: the layout detector is a small anchor-free convolutional
// detector (backbone convs -> class head + box head), so this header pulls only
// conv/CTC ops (NO ops_attention.h) and NO api.h/model_recipe.h. Both host and
// device TUs may include it. The public RecognizeResult is assembled by a
// host-only bridge from the plain scheduler.h types.
//
// PP-DocLayoutV2 and V3 are SEPARATE recipes/tensor/config contracts: different
// class counts, input sizes, and head geometry. They are never collapsed into
// one config; the version is an explicit field and the tensor inventory is
// version-specific.
//
// HONESTY (plan §1/§6): NO official PP-DocLayout checkpoint has been pinned,
// inspected, or run. Every dimension / tensor name here is a DECLARED SYNTHETIC
// CONTRACT exercised by the synthetic layout fixture. Real packages fail closed
// through pp_doc_layout_validate_real_artifact(). The official gates (mAP,
// reading-order accuracy) are recorded in MODELS.md. Nothing here validates a
// real artifact, and no checkpoint hash / official tensor name is invented.
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/models/qwen2_5_vl.h"// ExpectedTensor + validate_decoder_weight_desc
#include "ocr/models/pp_ocrv6.h"  // PpConvStage (folded conv contract)
#include "ocr/output/db_postprocess.h"
#include "ocr/package.h"
#include "ocr/runtime/scheduler.h"
#include "ocr/status.h"

namespace luisa::ocr {

// Explicit PP-DocLayout version. V2 and V3 are distinct contracts.
enum class PpDocLayoutVersion : uint32_t {
    v2 = 0,
    v3 = 1,
};

[[nodiscard]] luisa::string_view pp_doc_layout_version_name(PpDocLayoutVersion version) noexcept;

// One layout class. `name` is a stable contract tag; `block_type` maps it to the
// public block schema. The class list is version-specific caller data.
struct PpDocLayoutClass {
    luisa::string name;
    LayoutBlockType block_type{LayoutBlockType::text};
};

// Anchor-free layout detector geometry. The backbone is a short list of folded
// conv stages (shared PpConvStage contract); the class head is a 1x1 conv to
// num_classes (sigmoid per class), the box head a 1x1 conv to 4 (center offsets
// + log-scale sizes, decoded host-side). Both heads operate at the backbone
// output resolution.
struct PpDocLayoutConfig {
    PpDocLayoutVersion version{PpDocLayoutVersion::v2};
    uint32_t in_channels{3};
    luisa::vector<PpConvStage> backbone;
    PpConvStage class_head;// 1x1 -> num_classes
    PpConvStage box_head;  // 1x1 -> 4
    uint32_t input_width{0};
    uint32_t input_height{0};
    luisa::vector<PpDocLayoutClass> classes;

    [[nodiscard]] uint32_t num_classes() const noexcept { return static_cast<uint32_t>(classes.size()); }
    [[nodiscard]] Status validate() const;
    // Backbone output spatial size (== head map size).
    [[nodiscard]] Result<uint32_t> map_height() const;
    [[nodiscard]] Result<uint32_t> map_width() const;
};

// The declared SMALL SYNTHETIC per-version configs. NOT real checkpoints.
[[nodiscard]] PpDocLayoutConfig pp_doc_layout_synthetic_config(PpDocLayoutVersion version);

// Tensor-name contract (version-specific prefix). Conv weights conv_oihw,
// biases row_major. The inventory is exact; unexpected tensors are rejected.
struct PpDocLayoutTensorNames {
    luisa::string prefix{"pp_doclayout."};

    [[nodiscard]] Status validate() const;
    [[nodiscard]] luisa::string backbone_w(uint32_t i) const;
    [[nodiscard]] luisa::string backbone_b(uint32_t i) const;
    [[nodiscard]] luisa::string class_head_w() const;
    [[nodiscard]] luisa::string class_head_b() const;
    [[nodiscard]] luisa::string box_head_w() const;
    [[nodiscard]] luisa::string box_head_b() const;
};

// Full expected tensor inventory (ON-DISK layout tags).
[[nodiscard]] luisa::vector<ExpectedTensor> pp_doc_layout_expected_tensors(
    const PpDocLayoutConfig &config, const PpDocLayoutTensorNames &names);
// DEVICE-descriptor view (conv weights read row_major after the runtime map).
[[nodiscard]] luisa::vector<ExpectedTensor> pp_doc_layout_device_expected_tensors(
    const PpDocLayoutConfig &config, const PpDocLayoutTensorNames &names);
// Validate a package table against the inventory (exact name/shape/layout,
// fp32 only, no unexpected tensors).
[[nodiscard]] Status pp_doc_layout_validate_tensors(luisa::span<const PackageTensor> tensors,
                                                    luisa::span<const ExpectedTensor> expected);

// ---------------------------------------------------------------------------
// CPU decode (host algorithm; device-safe types)
// ---------------------------------------------------------------------------

// Decode options: score threshold, NMS / class-aware NMS, region caps.
struct PpDocLayoutDecodeOptions {
    // Minimum class sigmoid score to keep a detection, in [0, 1).
    float score_threshold{0.3f};
    // Axis-aligned IoU NMS threshold in (0, 1]; 0 disables NMS.
    float nms_iou_threshold{0.5f};
    // When true, NMS only suppresses detections of the SAME class (class-aware
    // NMS); when false, NMS is class-agnostic. Both are tested.
    bool class_aware_nms{true};
    // Maximum detections returned after NMS (0 = unlimited).
    uint32_t max_regions{64};
    // Minimum box side in SOURCE pixels (> 0). Degenerate boxes are dropped.
    float min_box_side{1.0f};
    // Box log-scale clamp: decoded size = stride * exp(clamp(logit, -cap, cap)).
    float box_log_cap{4.0f};

    [[nodiscard]] Status validate() const;
};

// One decoded layout region in ORIGINAL source coordinates. `score` is the
// detector class score (internal/test confidence source, explicitly labeled).
struct PpDocLayoutRegion {
    uint32_t class_index{0};
    LayoutBlockType block_type{LayoutBlockType::text};
    float score{0.0f};
    output::DbRect bounds;
    luisa::vector<output::DbPoint> polygon;// axis-aligned quad (TL,TR,BR,BL)
    uint32_t reading_order{0};           // assigned by order_regions
};

// Axis-aligned IoU of two boxes (shared with DB NMS semantics).
[[nodiscard]] float pp_doc_layout_iou(const output::DbRect &a, const output::DbRect &b) noexcept;

// Greedy NMS over detections sorted by score descending. `class_aware` restricts
// suppression to the same class. Ties break on the original input index (stable).
// Returns the kept indices in score order. Exposed for unit testing.
[[nodiscard]] luisa::vector<uint32_t> pp_doc_layout_nms(luisa::span<const PpDocLayoutRegion> regions,
                                                        float iou_threshold, bool class_aware);

// Decode the class/box head maps into source-coordinate regions.
//   class_probs [num_classes, map_h, map_w] sigmoid scores (row-major per class)
//   box_deltas  [4, map_h, map_w] (dx, dy, log_w, log_h) per cell
//   transform   maps detector-map pixels -> ORIGINAL source pixels
// The box at cell (cx, cy) is centered at ((cx+0.5+dx)*stride, (cy+0.5+dy)*stride)
// in MAP units with size stride*exp(clamp(log)). After decode: score threshold,
// NMS/class-aware NMS, region cap, source transform, reading order. An empty or
// all-background map yields ZERO regions. Fails closed on malformed inputs.
[[nodiscard]] Result<luisa::vector<PpDocLayoutRegion>> pp_doc_layout_decode(
    luisa::span<const float> class_probs, luisa::span<const float> box_deltas,
    uint32_t map_width, uint32_t map_height, uint32_t num_classes, float stride,
    const output::DbSourceTransform &transform, const PpDocLayoutConfig &config,
    const PpDocLayoutDecodeOptions &options = {});

// Assign a stable reading order to decoded regions (reuse the Phase 5
// multi-column reading-order algorithm). Returns regions in reading order with
// reading_order set (0-based, dense). Empty input -> empty output.
[[nodiscard]] Result<luisa::vector<PpDocLayoutRegion>> pp_doc_layout_order_regions(
    luisa::span<const PpDocLayoutRegion> regions);

// Real-artifact gate: ALWAYS fail-closed in Phase 6. No official PP-DocLayout
// checkpoint/config has been qualified; any real artifact is refused with
// `unvalidated` regardless of its contents.
[[nodiscard]] Status pp_doc_layout_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                          luisa::string_view config_fingerprint);

}// namespace luisa::ocr