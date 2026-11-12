#pragma once
// OCR engine -- OpenDoc-0.1B two-stage recognizer contract (Phase 6).
//
// DEVICE-SAFE header. OpenDoc is a TWO-STAGE pipeline, NOT a generic dense VLM:
//   stage 1: PP-DocLayoutV2 layout detection (models/pp_doc_layout.*),
//   stage 2: UniRec-0.1B region recognizer -- a conventional CNN + CTC
//            recognizer (the shared CnnExecutor recognizer graph) applied per
//            region, with TYPE-SPECIFIC output adapters (text / formula /
//            table). There is NO autoregressive KV decoding and NO dense VLM
//            executor on this path; forcing OpenDoc into one is a contract
//            violation this module explicitly avoids.
//
// This header pulls NO api.h/model_recipe.h and NO ops_attention.h, so both
// host and device TUs may include it. The device orchestration lives in
// runtime/layout_vlm_executor.* (OpenDocRecognizer); host composition + the
// public result bridge live in open_doc_result.* (host-only).
//
// HONESTY (plan §1/§6): NO official OpenDoc/UniRec checkpoint has been pinned,
// inspected, or run. Every dimension / tensor name here is a DECLARED SYNTHETIC
// CONTRACT exercised by the synthetic two-stage fixture. Real packages fail
// closed through open_doc_validate_real_artifact(). The official gates are
// recorded in MODELS.md.
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/models/pp_doc_layout.h"
#include "ocr/models/pp_ocrv6.h"// PpRecognizerConfig (UniRec CNN+CTC geometry)
#include "ocr/output/ctc_decode.h"
#include "ocr/package.h"
#include "ocr/runtime/scheduler.h"
#include "ocr/status.h"

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// UniRec recognizer geometry (CNN + CTC; NOT a VLM)
// ---------------------------------------------------------------------------

// UniRec-0.1B region recognizer. The geometry reuses the shared PpRecognizerConfig
// (LCNetV4-style backbone + LightSVTR height-collapse + per-frame SVTR MLP + CTC
// head) because UniRec is a conventional sequence recognizer, not a transformer
// decoder. The contract is DISTINCT from PP-OCRv6 only in identity/tensor map and
// the type-specific output adapters below; nothing here claims a real checkpoint.
struct UniRecConfig {
    PpRecognizerConfig recognizer;
    // Character dictionary (one entry per non-blank class). The synthetic
    // fixture uses a small multilingual dictionary incl. LaTeX/OTSL tokens.
    output::CtcDictionary dictionary;

    [[nodiscard]] Status validate() const;
};

// The declared SMALL SYNTHETIC UniRec config. NOT a real checkpoint.
[[nodiscard]] UniRecConfig uni_rec_synthetic_config();

// Tensor-name contract for the UniRec recognizer component.
struct UniRecTensorNames {
    luisa::string prefix{"unirec."};

    [[nodiscard]] Status validate() const;
    [[nodiscard]] luisa::string backbone_w(uint32_t i) const;
    [[nodiscard]] luisa::string backbone_b(uint32_t i) const;
    [[nodiscard]] luisa::string svtr_fc1_w() const;
    [[nodiscard]] luisa::string svtr_fc1_b() const;
    [[nodiscard]] luisa::string svtr_fc2_w() const;
    [[nodiscard]] luisa::string svtr_fc2_b() const;
    [[nodiscard]] luisa::string ctc_w() const;
    [[nodiscard]] luisa::string ctc_b() const;
};

// Expected tensor inventory (ON-DISK + device views) for the recognizer.
[[nodiscard]] luisa::vector<ExpectedTensor> uni_rec_expected_tensors(
    const UniRecConfig &config, const UniRecTensorNames &names);
[[nodiscard]] luisa::vector<ExpectedTensor> uni_rec_device_expected_tensors(
    const UniRecConfig &config, const UniRecTensorNames &names);
[[nodiscard]] Status uni_rec_validate_tensors(luisa::span<const PackageTensor> tensors,
                                              luisa::span<const ExpectedTensor> expected);

// ---------------------------------------------------------------------------
// Full OpenDoc config
// ---------------------------------------------------------------------------

struct OpenDocConfig {
    // Stage 1 layout: PP-DocLayoutV2 (the OpenDoc contract uses V2, not V3).
    PpDocLayoutConfig layout;
    PpDocLayoutTensorNames layout_names;
    // Stage 2 recognizer: UniRec.
    UniRecConfig recognizer;
    UniRecTensorNames recognizer_names;

    [[nodiscard]] Status validate() const;
};

// The declared SMALL SYNTHETIC OpenDoc config (PP-DocLayoutV2 + UniRec).
[[nodiscard]] OpenDocConfig open_doc_synthetic_config();

// ---------------------------------------------------------------------------
// Type-specific output adapters live in open_doc_result.h (HOST-ONLY), because
// the formula adapter uses check_latex (output/markdown.h -> api.h). The device
// orchestration produces raw CTC text; the host bridge adapts it per type.
// ---------------------------------------------------------------------------

// Real-artifact gate: ALWAYS fail-closed in Phase 6.
[[nodiscard]] Status open_doc_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                     luisa::string_view config_fingerprint);

}// namespace luisa::ocr