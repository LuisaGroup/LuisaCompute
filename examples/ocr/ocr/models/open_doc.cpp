// OCR engine -- OpenDoc-0.1B two-stage contract (Phase 6). Device-safe TU.
// Config/tensor-inventory/gate only; device orchestration lives in
// runtime/layout_vlm_executor.* and host adapters/composition in open_doc_result.*.
#include "ocr/models/open_doc.h"

#include <utility>

#include <luisa/core/stl/format.h>

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// UniRec config
// ---------------------------------------------------------------------------

[[nodiscard]] Status UniRecConfig::validate() const {
    if (auto s = recognizer.validate(); !s.ok()) { return s; }
    if (auto s = dictionary.validate(); !s.ok()) { return s; }
    if (dictionary.num_classes() != recognizer.num_classes) {
        return Status::error(StatusCode::invalid_argument,
                             "UniRec dictionary class count must match the CTC head", "dictionary");
    }
    if (dictionary.blank_id != recognizer.blank_id) {
        return Status::error(StatusCode::invalid_argument,
                             "UniRec dictionary blank id must match the recognizer", "dictionary");
    }
    return Status::success();
}

[[nodiscard]] UniRecConfig uni_rec_synthetic_config() {
    UniRecConfig cfg;
    // Reuse the small synthetic PP-OCRv6 recognizer geometry (tiny) as the
    // UniRec CNN+CTC contract. This is a DECLARED synthetic geometry, not an
    // official UniRec checkpoint. The recognizer is a conventional sequence
    // model -- there is no transformer decoder / KV cache on this path.
    const auto pp = pp_ocrv6_synthetic_config(PpOcrSize::tiny);
    cfg.recognizer = pp.recognizer;
    // Small multilingual dictionary incl. LaTeX/OTSL tokens for the type-specific
    // adapters. 15 non-blank entries (num_classes 16, blank id 0).
    cfg.dictionary.characters = {
        "a", "b", "c", "x", "y", "z", "0", "1", "2", "+", "=",
        "\\", "frac", "{", "}"};
    cfg.dictionary.blank_id = 0;
    return cfg;
}

// ---------------------------------------------------------------------------
// UniRec tensor names
// ---------------------------------------------------------------------------

[[nodiscard]] Status UniRecTensorNames::validate() const {
    if (prefix.empty()) {
        return Status::error(StatusCode::invalid_argument, "tensor prefix must not be empty", "prefix");
    }
    return Status::success();
}

[[nodiscard]] luisa::string UniRecTensorNames::backbone_w(uint32_t i) const {
    return prefix + "backbone." + luisa::format("{}", i) + ".weight";
}
[[nodiscard]] luisa::string UniRecTensorNames::backbone_b(uint32_t i) const {
    return prefix + "backbone." + luisa::format("{}", i) + ".bias";
}
[[nodiscard]] luisa::string UniRecTensorNames::svtr_fc1_w() const { return prefix + "svtr.fc1.weight"; }
[[nodiscard]] luisa::string UniRecTensorNames::svtr_fc1_b() const { return prefix + "svtr.fc1.bias"; }
[[nodiscard]] luisa::string UniRecTensorNames::svtr_fc2_w() const { return prefix + "svtr.fc2.weight"; }
[[nodiscard]] luisa::string UniRecTensorNames::svtr_fc2_b() const { return prefix + "svtr.fc2.bias"; }
[[nodiscard]] luisa::string UniRecTensorNames::ctc_w() const { return prefix + "ctc.weight"; }
[[nodiscard]] luisa::string UniRecTensorNames::ctc_b() const { return prefix + "ctc.bias"; }

namespace {

[[nodiscard]] ExpectedTensor conv_weight(const luisa::string &name, const PpConvStage &s) {
    ExpectedTensor t;
    t.name = name;
    t.shape = {s.out_c, s.in_c / s.groups, s.kh, s.kw};
    t.layout = TensorLayout::conv_oihw;
    return t;
}
[[nodiscard]] ExpectedTensor bias_tensor(const luisa::string &name, uint32_t c) {
    ExpectedTensor t;
    t.name = name;
    t.shape = {c};
    t.layout = TensorLayout::row_major;
    return t;
}
[[nodiscard]] ExpectedTensor kn_weight(const luisa::string &name, uint32_t k, uint32_t n) {
    ExpectedTensor t;
    t.name = name;
    t.shape = {k, n};
    t.layout = TensorLayout::kn;
    return t;
}

}// namespace

[[nodiscard]] luisa::vector<ExpectedTensor> uni_rec_expected_tensors(
    const UniRecConfig &config, const UniRecTensorNames &names) {
    luisa::vector<ExpectedTensor> out;
    const auto &rec = config.recognizer;
    for (uint32_t i = 0; i < rec.backbone.size(); i++) {
        out.push_back(conv_weight(names.backbone_w(i), rec.backbone[i]));
        out.push_back(bias_tensor(names.backbone_b(i), rec.backbone[i].out_c));
    }
    uint32_t collapsed = 0;
    if (auto cc = rec.collapsed_channels(); cc.ok()) { collapsed = *cc.value; }
    out.push_back(kn_weight(names.svtr_fc1_w(), collapsed, rec.svtr_intermediate));
    out.push_back(bias_tensor(names.svtr_fc1_b(), rec.svtr_intermediate));
    out.push_back(kn_weight(names.svtr_fc2_w(), rec.svtr_intermediate, collapsed));
    out.push_back(bias_tensor(names.svtr_fc2_b(), collapsed));
    out.push_back(kn_weight(names.ctc_w(), collapsed, rec.num_classes));
    out.push_back(bias_tensor(names.ctc_b(), rec.num_classes));
    return out;
}

[[nodiscard]] luisa::vector<ExpectedTensor> uni_rec_device_expected_tensors(
    const UniRecConfig &config, const UniRecTensorNames &names) {
    auto out = uni_rec_expected_tensors(config, names);
    for (auto &e : out) {
        if (e.layout == TensorLayout::conv_oihw || e.layout == TensorLayout::conv_ohwi) {
            e.layout = TensorLayout::row_major;
        }
    }
    return out;
}

[[nodiscard]] Status uni_rec_validate_tensors(luisa::span<const PackageTensor> tensors,
                                              luisa::span<const ExpectedTensor> expected) {
    for (const auto &exp : expected) {
        const PackageTensor *found = nullptr;
        for (const auto &t : tensors) {
            if (t.name == exp.name) { found = &t; break; }
        }
        if (found == nullptr) {
            return Status::error(StatusCode::not_found,
                                 luisa::format("required unirec tensor '{}' is missing", exp.name), exp.name);
        }
        if (found->shape != exp.shape) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("unirec tensor '{}' has the wrong shape", exp.name), exp.name);
        }
        if (found->layout != exp.layout) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("unirec tensor '{}' must use layout '{}'", exp.name,
                                               tensor_layout_name(exp.layout)), exp.name);
        }
        if (found->encoding != TensorEncoding::fp32) {
            return Status::error(StatusCode::unsupported,
                                 luisa::format("unirec tensor '{}' must be fp32 in the Phase 6 baseline", exp.name), exp.name);
        }
    }
    for (const auto &t : tensors) {
        bool known = false;
        for (const auto &exp : expected) {
            if (exp.name == t.name) { known = true; break; }
        }
        if (!known) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("unexpected unirec tensor '{}'", t.name), t.name);
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Full OpenDoc config
// ---------------------------------------------------------------------------

[[nodiscard]] Status OpenDocConfig::validate() const {
    if (layout.version != PpDocLayoutVersion::v2) {
        return Status::error(StatusCode::invalid_argument,
                             "the OpenDoc contract uses PP-DocLayoutV2 (not V3)", "layout.version");
    }
    if (auto s = layout.validate(); !s.ok()) { return s; }
    if (auto s = layout_names.validate(); !s.ok()) { return s; }
    if (auto s = recognizer.validate(); !s.ok()) { return s; }
    if (auto s = recognizer_names.validate(); !s.ok()) { return s; }
    return Status::success();
}

[[nodiscard]] OpenDocConfig open_doc_synthetic_config() {
    OpenDocConfig cfg;
    cfg.layout = pp_doc_layout_synthetic_config(PpDocLayoutVersion::v2);
    cfg.layout_names = PpDocLayoutTensorNames{};
    cfg.recognizer = uni_rec_synthetic_config();
    cfg.recognizer_names = UniRecTensorNames{};
    return cfg;
}

// ---------------------------------------------------------------------------
// Real-artifact gate
// ---------------------------------------------------------------------------

[[nodiscard]] Status open_doc_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                     luisa::string_view config_fingerprint) {
    (void)checkpoint_revision;
    (void)config_fingerprint;
    return Status::error(StatusCode::unvalidated,
                         "no official OpenDoc/UniRec checkpoint/config has been qualified; "
                         "real artifacts fail closed (synthetic two-stage path only)",
                         "open_doc");
}

}// namespace luisa::ocr