// OCR engine -- PP-OCRv6 CNN graph contract implementation (Phase 5).
// Host-only. See the header for the honesty/synthetic-contract boundary.
#include "ocr/models/pp_ocrv6.h"

#include <cmath>

#include <luisa/core/stl/format.h>

#include "ocr/ops/ops_conv.h"// conv2d_out_dim (host geometry helper)

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// PpConvStage
// ---------------------------------------------------------------------------

[[nodiscard]] Status PpConvStage::validate() const {
    if (in_c == 0u || out_c == 0u || kh == 0u || kw == 0u ||
        stride_h == 0u || stride_w == 0u || dil_h == 0u || dil_w == 0u || groups == 0u) {
        return Status::error(StatusCode::invalid_argument, "conv stage dimensions must be nonzero", "stage");
    }
    if (in_c % groups != 0u || out_c % groups != 0u) {
        return Status::error(StatusCode::invalid_argument, "conv stage groups must divide in_c and out_c", "groups");
    }
    if (act > 3u) {
        return Status::error(StatusCode::invalid_argument, "conv stage activation must be 0..3", "act");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// PpDetectorConfig
// ---------------------------------------------------------------------------

[[nodiscard]] uint32_t PpDetectorConfig::stride_h() const noexcept {
    uint32_t s = 1;
    for (const auto &stage : backbone) { s *= stage.stride_h; }
    return s;
}
[[nodiscard]] uint32_t PpDetectorConfig::stride_w() const noexcept {
    uint32_t s = 1;
    for (const auto &stage : backbone) { s *= stage.stride_w; }
    return s;
}

[[nodiscard]] Status PpDetectorConfig::validate() const {
    if (in_channels == 0u || backbone.empty()) {
        return Status::error(StatusCode::invalid_argument, "detector needs channels and a backbone", "detector");
    }
    if (input_width == 0u || input_height == 0u) {
        return Status::error(StatusCode::invalid_argument, "detector input size must be nonzero", "input");
    }
    if (lateral_stage >= backbone.size()) {
        return Status::error(StatusCode::invalid_argument, "lateral_stage out of range", "lateral_stage");
    }
    uint32_t c = in_channels;
    uint32_t h = input_height, w = input_width;
    uint32_t skip_h = 0, skip_w = 0;
    for (size_t i = 0; i < backbone.size(); i++) {
        const auto &s = backbone[i];
        if (auto st = s.validate(); !st.ok()) { return st; }
        if (s.in_c != c) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("detector backbone[{}] channel mismatch ({} vs {})", i, s.in_c, c),
                                 "backbone");
        }
        auto oh = conv2d_out_dim(h, s.kh, s.stride_h, s.pad_h, s.dil_h);
        auto ow = conv2d_out_dim(w, s.kw, s.stride_w, s.pad_w, s.dil_w);
        if (!oh.ok() || !ow.ok()) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("detector backbone[{}] degenerate geometry", i), "backbone");
        }
        h = *oh.value;
        w = *ow.value;
        c = s.out_c;
        if (i == lateral_stage) {
            skip_h = h;
            skip_w = w;
        }
    }
    // The FPN neck upsamples the lateral skip to the backbone output
    // resolution and concatenates them. A skip LARGER than the backbone
    // output would need a downsample the neck does not implement, and the
    // executor's lateral/upsample/fusion arenas are sized at the backbone
    // output resolution -- so reject it here, BEFORE any allocation.
    if (skip_h > h || skip_w > w) {
        return Status::error(StatusCode::invalid_argument,
                             luisa::format("detector lateral skip ({}x{}) is larger than the backbone output "
                                           "({}x{}); the FPN neck only upsamples the skip",
                                           skip_w, skip_h, w, h),
                             "lateral_stage");
    }
    if (auto st = lateral_conv.validate(); !st.ok()) { return st; }
    if (auto st = replk_conv.validate(); !st.ok()) { return st; }
    if (auto st = db_conv.validate(); !st.ok()) { return st; }
    if (db_conv.out_c != 1u) {
        return Status::error(StatusCode::invalid_argument, "DB head must output 1 channel", "db_conv");
    }
    // The lateral conv consumes the skip stage's channels; the RepLK conv
    // consumes the fused (main + lateral) channels; the DB head consumes the
    // RepLK output channels.
    if (lateral_conv.in_c != backbone[lateral_stage].out_c) {
        return Status::error(StatusCode::invalid_argument, "lateral conv channel mismatch", "lateral_conv");
    }
    if (replk_conv.in_c != backbone.back().out_c + lateral_conv.out_c) {
        return Status::error(StatusCode::invalid_argument,
                             "replk conv must consume main+lateral channels", "replk_conv");
    }
    if (db_conv.in_c != replk_conv.out_c) {
        return Status::error(StatusCode::invalid_argument, "db conv channel mismatch", "db_conv");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// PpRecognizerConfig
// ---------------------------------------------------------------------------

[[nodiscard]] uint32_t PpRecognizerConfig::stride_h() const noexcept {
    uint32_t s = 1;
    for (const auto &stage : backbone) { s *= stage.stride_h; }
    return s;
}
[[nodiscard]] uint32_t PpRecognizerConfig::stride_w() const noexcept {
    uint32_t s = 1;
    for (const auto &stage : backbone) { s *= stage.stride_w; }
    return s;
}

[[nodiscard]] Result<uint32_t> PpRecognizerConfig::feature_height() const {
    uint32_t h = input_height;
    for (const auto &s : backbone) {
        auto oh = conv2d_out_dim(h, s.kh, s.stride_h, s.pad_h, s.dil_h);
        if (!oh.ok()) { return Result<uint32_t>::failure(oh.status); }
        h = *oh.value;
    }
    return Result<uint32_t>::success(h);
}
[[nodiscard]] Result<uint32_t> PpRecognizerConfig::feature_width() const {
    uint32_t w = input_width;
    for (const auto &s : backbone) {
        auto ow = conv2d_out_dim(w, s.kw, s.stride_w, s.pad_w, s.dil_w);
        if (!ow.ok()) { return Result<uint32_t>::failure(ow.status); }
        w = *ow.value;
    }
    return Result<uint32_t>::success(w);
}
[[nodiscard]] Result<uint32_t> PpRecognizerConfig::collapsed_channels() const {
    if (backbone.empty()) {
        return Result<uint32_t>::failure(StatusCode::invalid_argument, "recognizer backbone empty", "backbone");
    }
    const uint64_t c = static_cast<uint64_t>(backbone.back().out_c) * height_collapse;
    auto cu = checked_u32(c, "collapsed channels");
    if (!cu.ok()) { return Result<uint32_t>::failure(cu.status); }
    return Result<uint32_t>::success(*cu.value);
}
[[nodiscard]] Result<uint32_t> PpRecognizerConfig::num_frames() const {
    // The LightSVTR height-collapse folds the height axis into channels in
    // blocks of `height_collapse`, so EVERY remaining height row becomes its
    // own frame: frames = feature_width * (feature_height / height_collapse).
    // (Ignoring the height factor would under-size the sequence/logits
    // arenas and the per-frame GEMMs whenever feature_height > collapse.)
    auto fh = feature_height();
    if (!fh.ok()) { return fh; }
    auto fw = feature_width();
    if (!fw.ok()) { return fw; }
    if (height_collapse == 0u) {
        return Result<uint32_t>::failure(StatusCode::invalid_argument,
                                         "height_collapse must be >= 1", "height_collapse");
    }
    if (*fh.value % height_collapse != 0u) {
        return Result<uint32_t>::failure(StatusCode::invalid_argument,
                                         "feature height must be divisible by height_collapse",
                                         "height_collapse");
    }
    const uint64_t frames = static_cast<uint64_t>(*fw.value) * (*fh.value / height_collapse);
    auto f32 = checked_u32(frames, "recognizer frame count");
    if (!f32.ok()) { return Result<uint32_t>::failure(f32.status); }
    return Result<uint32_t>::success(*f32.value);
}

[[nodiscard]] Status PpRecognizerConfig::validate() const {
    if (in_channels == 0u || backbone.empty()) {
        return Status::error(StatusCode::invalid_argument, "recognizer needs channels and a backbone", "recognizer");
    }
    if (input_width == 0u || input_height == 0u) {
        return Status::error(StatusCode::invalid_argument, "recognizer input size must be nonzero", "input");
    }
    if (height_collapse == 0u) {
        return Status::error(StatusCode::invalid_argument, "height_collapse must be >= 1", "height_collapse");
    }
    if (svtr_intermediate == 0u || num_classes < 2u) {
        return Status::error(StatusCode::invalid_argument, "recognizer needs svtr/ctc widths", "svtr/ctc");
    }
    if (blank_id >= num_classes) {
        return Status::error(StatusCode::invalid_argument, "blank_id outside class range", "blank_id");
    }
    uint32_t c = in_channels;
    for (size_t i = 0; i < backbone.size(); i++) {
        const auto &s = backbone[i];
        if (auto st = s.validate(); !st.ok()) { return st; }
        if (s.in_c != c) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("recognizer backbone[{}] channel mismatch", i), "backbone");
        }
        c = s.out_c;
    }
    auto fh = feature_height();
    auto fw = feature_width();
    if (!fh.ok() || !fw.ok()) {
        return Status::error(StatusCode::invalid_argument, "recognizer degenerate geometry", "backbone");
    }
    if (*fh.value == 0u || *fw.value == 0u) {
        return Status::error(StatusCode::invalid_argument, "recognizer feature size is zero", "feature");
    }
    if (*fh.value % height_collapse != 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "feature height must be divisible by height_collapse", "height_collapse");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// PpOcrv6Config
// ---------------------------------------------------------------------------

[[nodiscard]] Status PpOcrv6Config::validate() const {
    if (size == PpOcrSize::none) {
        return Status::error(StatusCode::invalid_argument, "pp_ocrv6 config needs an explicit size", "size");
    }
    if (auto s = detector.validate(); !s.ok()) { return s; }
    if (auto s = recognizer.validate(); !s.ok()) { return s; }
    return Status::success();
}

namespace {

// Synthetic per-size geometry. Each size is a DISTINCT contract (different
// channel widths / kernel sizes / input sizes), validated independently.
// These are declared synthetic values, NOT official PP-OCRv6 dimensions.
[[nodiscard]] PpOcrv6Config make_synthetic(PpOcrSize size) {
    PpOcrv6Config cfg;
    cfg.size = size;
    // Channel base scales with size: tiny=4, small=6, medium=8.
    const uint32_t base = size == PpOcrSize::tiny ? 4u : size == PpOcrSize::small ? 6u : 8u;
    const uint32_t det_in = size == PpOcrSize::tiny ? 64u : size == PpOcrSize::small ? 96u : 128u;
    const uint32_t rec_w = size == PpOcrSize::tiny ? 64u : size == PpOcrSize::small ? 96u : 128u;

    auto &det = cfg.detector;
    det.in_channels = 3;
    det.input_width = det_in;
    det.input_height = det_in;
    // Backbone: 3x3 stride-2 conv, then asymmetric (2,1) depthwise 3x3, then
    // a large 5x5 depthwise conv (stride 1). Total stride (4,2).
    PpConvStage s0;
    s0.in_c = 3;
    s0.out_c = base;
    s0.kh = s0.kw = 3;
    s0.stride_h = s0.stride_w = 2;
    s0.pad_h = s0.pad_w = 1;
    s0.act = 1;// relu
    PpConvStage s1;
    s1.in_c = base;
    s1.out_c = base;
    s1.kh = s1.kw = 3;
    s1.stride_h = 2;
    s1.stride_w = 1;// asymmetric stride (2,1)
    s1.pad_h = s1.pad_w = 1;
    s1.groups = base;// depthwise
    s1.act = 1;
    PpConvStage s2;
    s2.in_c = base;
    s2.out_c = base * 2;
    s2.kh = s2.kw = 5;// large depthwise conv
    s2.stride_h = s2.stride_w = 1;
    s2.pad_h = s2.pad_w = 2;
    s2.groups = base;// depthwise, in_c=base -> out_c must be multiple of base
    s2.act = 3;// hard_swish
    det.backbone = {s0, s1, s2};
    det.lateral_stage = 1;// skip from the asymmetric stage (stride (4,1))
    // Lateral 1x1 on the skip (base channels -> base channels).
    det.lateral_conv.in_c = base;
    det.lateral_conv.out_c = base;
    det.lateral_conv.kh = det.lateral_conv.kw = 1;
    det.lateral_conv.stride_h = det.lateral_conv.stride_w = 1;
    det.lateral_conv.act = 0;
    // RepLK folded large-kernel conv on the fused (2*base + base) channels.
    det.replk_conv.in_c = base * 3;
    det.replk_conv.out_c = base * 2;
    det.replk_conv.kh = det.replk_conv.kw = 7;// large kernel (folded RepLK)
    det.replk_conv.stride_h = det.replk_conv.stride_w = 1;
    det.replk_conv.pad_h = det.replk_conv.pad_w = 3;
    det.replk_conv.act = 1;
    // DB head 1x1 -> 1 channel.
    det.db_conv.in_c = base * 2;
    det.db_conv.out_c = 1;
    det.db_conv.kh = det.db_conv.kw = 1;
    det.db_conv.stride_h = det.db_conv.stride_w = 1;
    det.db_conv.act = 0;

    auto &rec = cfg.recognizer;
    rec.in_channels = 3;
    rec.input_width = rec_w;
    rec.input_height = 16;
    // Backbone: 3x3 stride-(2,2) conv, then 3x3 stride-(2,1) conv (asymmetric),
    // then a 3x3 depthwise conv stride-(2,1). Height 16 -> 8 -> 4 -> 2.
    PpConvStage r0;
    r0.in_c = 3;
    r0.out_c = base;
    r0.kh = r0.kw = 3;
    r0.stride_h = r0.stride_w = 2;
    r0.pad_h = r0.pad_w = 1;
    r0.act = 1;
    PpConvStage r1;
    r1.in_c = base;
    r1.out_c = base * 2;
    r1.kh = r1.kw = 3;
    r1.stride_h = 2;
    r1.stride_w = 1;
    r1.pad_h = r1.pad_w = 1;
    r1.act = 1;
    PpConvStage r2;
    r2.in_c = base * 2;
    r2.out_c = base * 2;
    r2.kh = r2.kw = 3;
    r2.stride_h = 2;
    r2.stride_w = 1;
    r2.pad_h = r2.pad_w = 1;
    r2.groups = base * 2;// depthwise
    r2.act = 3;
    rec.backbone = {r0, r1, r2};
    rec.height_collapse = 2;// feature height 2 -> collapse to 1, channels *2
    rec.svtr_intermediate = base * 4;
    rec.num_classes = 16;// small synthetic vocabulary (incl. blank)
    rec.blank_id = 0;
    return cfg;
}

}// namespace

[[nodiscard]] PpOcrv6Config pp_ocrv6_synthetic_config(PpOcrSize size) {
    return make_synthetic(size);
}

// ---------------------------------------------------------------------------
// Tensor names
// ---------------------------------------------------------------------------

[[nodiscard]] Status PpOcrv6TensorNames::validate() const {
    if (prefix.empty()) {
        return Status::error(StatusCode::invalid_argument, "tensor prefix must not be empty", "prefix");
    }
    return Status::success();
}

[[nodiscard]] luisa::string PpOcrv6TensorNames::det_backbone_w(uint32_t i) const {
    return prefix + "detector.backbone." + luisa::format("{}", i) + ".weight";
}
[[nodiscard]] luisa::string PpOcrv6TensorNames::det_backbone_b(uint32_t i) const {
    return prefix + "detector.backbone." + luisa::format("{}", i) + ".bias";
}
[[nodiscard]] luisa::string PpOcrv6TensorNames::det_lateral_w() const { return prefix + "detector.lateral.weight"; }
[[nodiscard]] luisa::string PpOcrv6TensorNames::det_lateral_b() const { return prefix + "detector.lateral.bias"; }
[[nodiscard]] luisa::string PpOcrv6TensorNames::det_replk_w() const { return prefix + "detector.replk.weight"; }
[[nodiscard]] luisa::string PpOcrv6TensorNames::det_replk_b() const { return prefix + "detector.replk.bias"; }
[[nodiscard]] luisa::string PpOcrv6TensorNames::det_db_w() const { return prefix + "detector.db.weight"; }
[[nodiscard]] luisa::string PpOcrv6TensorNames::det_db_b() const { return prefix + "detector.db.bias"; }
[[nodiscard]] luisa::string PpOcrv6TensorNames::rec_backbone_w(uint32_t i) const {
    return prefix + "recognizer.backbone." + luisa::format("{}", i) + ".weight";
}
[[nodiscard]] luisa::string PpOcrv6TensorNames::rec_backbone_b(uint32_t i) const {
    return prefix + "recognizer.backbone." + luisa::format("{}", i) + ".bias";
}
[[nodiscard]] luisa::string PpOcrv6TensorNames::rec_svtr_fc1_w() const { return prefix + "recognizer.svtr.fc1.weight"; }
[[nodiscard]] luisa::string PpOcrv6TensorNames::rec_svtr_fc1_b() const { return prefix + "recognizer.svtr.fc1.bias"; }
[[nodiscard]] luisa::string PpOcrv6TensorNames::rec_svtr_fc2_w() const { return prefix + "recognizer.svtr.fc2.weight"; }
[[nodiscard]] luisa::string PpOcrv6TensorNames::rec_svtr_fc2_b() const { return prefix + "recognizer.svtr.fc2.bias"; }
[[nodiscard]] luisa::string PpOcrv6TensorNames::rec_ctc_w() const { return prefix + "recognizer.ctc.weight"; }
[[nodiscard]] luisa::string PpOcrv6TensorNames::rec_ctc_b() const { return prefix + "recognizer.ctc.bias"; }

namespace {

[[nodiscard]] ExpectedTensor conv_weight(const luisa::string &name, const PpConvStage &s) {
    ExpectedTensor t;
    t.name = name;
    t.shape = {s.out_c, s.in_c / s.groups, s.kh, s.kw};
    // ON-DISK contract: conv weights carry the conv_oihw package tag. The
    // DEVICE-descriptor view is derived separately by
    // pp_ocrv6_device_expected_tensors (the runtime maps conv_oihw ->
    // row_major in weights.cpp logical_desc).
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

[[nodiscard]] luisa::vector<ExpectedTensor> pp_ocrv6_expected_tensors(
    const PpOcrv6Config &config, const PpOcrv6TensorNames &names, luisa::string_view component) {
    luisa::vector<ExpectedTensor> out;
    if (component == "detector") {
        const auto &det = config.detector;
        for (uint32_t i = 0; i < det.backbone.size(); i++) {
            out.push_back(conv_weight(names.det_backbone_w(i), det.backbone[i]));
            out.push_back(bias_tensor(names.det_backbone_b(i), det.backbone[i].out_c));
        }
        out.push_back(conv_weight(names.det_lateral_w(), det.lateral_conv));
        out.push_back(bias_tensor(names.det_lateral_b(), det.lateral_conv.out_c));
        out.push_back(conv_weight(names.det_replk_w(), det.replk_conv));
        out.push_back(bias_tensor(names.det_replk_b(), det.replk_conv.out_c));
        out.push_back(conv_weight(names.det_db_w(), det.db_conv));
        out.push_back(bias_tensor(names.det_db_b(), det.db_conv.out_c));
    } else if (component == "recognizer") {
        const auto &rec = config.recognizer;
        for (uint32_t i = 0; i < rec.backbone.size(); i++) {
            out.push_back(conv_weight(names.rec_backbone_w(i), rec.backbone[i]));
            out.push_back(bias_tensor(names.rec_backbone_b(i), rec.backbone[i].out_c));
        }
        // collapsed_channels is config-derived; compute it defensively.
        uint32_t collapsed = 0;
        if (auto cc = rec.collapsed_channels(); cc.ok()) { collapsed = *cc.value; }
        out.push_back(kn_weight(names.rec_svtr_fc1_w(), collapsed, rec.svtr_intermediate));
        out.push_back(bias_tensor(names.rec_svtr_fc1_b(), rec.svtr_intermediate));
        out.push_back(kn_weight(names.rec_svtr_fc2_w(), rec.svtr_intermediate, collapsed));
        out.push_back(bias_tensor(names.rec_svtr_fc2_b(), collapsed));
        out.push_back(kn_weight(names.rec_ctc_w(), collapsed, rec.num_classes));
        out.push_back(bias_tensor(names.rec_ctc_b(), rec.num_classes));
    }
    return out;
}

[[nodiscard]] luisa::vector<ExpectedTensor> pp_ocrv6_device_expected_tensors(
    const PpOcrv6Config &config, const PpOcrv6TensorNames &names, luisa::string_view component) {
    // The runtime's logical_desc maps every non-kn/nk PACKAGE layout to
    // row_major, so an installed conv weight's logical descriptor reads
    // row_major even though its on-disk tag is conv_oihw. Derive the device
    // view from the on-disk inventory instead of weakening the on-disk gate.
    auto out = pp_ocrv6_expected_tensors(config, names, component);
    for (auto &e : out) {
        if (e.layout == TensorLayout::conv_oihw || e.layout == TensorLayout::conv_ohwi) {
            e.layout = TensorLayout::row_major;
        }
    }
    return out;
}

[[nodiscard]] Status pp_ocrv6_validate_tensors(luisa::span<const PackageTensor> tensors,
                                               luisa::span<const ExpectedTensor> expected) {
    // Every expected tensor present with exact shape/layout; fp32 only.
    for (const auto &exp : expected) {
        const PackageTensor *found = nullptr;
        for (const auto &t : tensors) {
            if (t.name == exp.name) {
                found = &t;
                break;
            }
        }
        if (found == nullptr) {
            return Status::error(StatusCode::not_found,
                                 luisa::format("required pp_ocrv6 tensor '{}' is missing", exp.name), exp.name);
        }
        if (found->shape != exp.shape) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("pp_ocrv6 tensor '{}' has the wrong shape for the config", exp.name),
                                 exp.name);
        }
        if (found->layout != exp.layout) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("pp_ocrv6 tensor '{}' must use layout '{}'", exp.name,
                                               tensor_layout_name(exp.layout)),
                                 exp.name);
        }
        if (found->encoding != TensorEncoding::fp32) {
            return Status::error(StatusCode::unsupported,
                                 luisa::format("pp_ocrv6 tensor '{}' uses encoding '{}'; the Phase 5 CNN baseline "
                                               "requires fp32",
                                               exp.name, tensor_encoding_name(found->encoding)),
                                 exp.name);
        }
    }
    // No unexpected tensors in this component.
    for (const auto &t : tensors) {
        bool known = false;
        for (const auto &exp : expected) {
            if (exp.name == t.name) {
                known = true;
                break;
            }
        }
        if (!known) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("unexpected pp_ocrv6 tensor '{}'", t.name), t.name);
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Folding helpers
// ---------------------------------------------------------------------------

[[nodiscard]] Status pp_fold_conv_bn(luisa::span<const float> weight, luisa::span<const float> bias,
                                     luisa::span<const float> bn_mean, luisa::span<const float> bn_var,
                                     luisa::span<const float> bn_gamma, luisa::span<const float> bn_beta,
                                     uint32_t out_c, uint64_t per_channel, float eps,
                                     luisa::vector<float> &out_weight, luisa::vector<float> &out_bias) {
    if (out_c == 0u || per_channel == 0u) {
        return Status::error(StatusCode::invalid_argument, "fold_conv_bn dimensions must be nonzero", "dims");
    }
    if (!std::isfinite(eps) || eps <= 0.0f) {
        return Status::error(StatusCode::invalid_argument, "fold_conv_bn eps must be > 0", "eps");
    }
    const uint64_t total = static_cast<uint64_t>(out_c) * per_channel;
    if (weight.size() != total) {
        return Status::error(StatusCode::invalid_argument, "fold_conv_bn weight size mismatch", "weight");
    }
    if (!bias.empty() && bias.size() != out_c) {
        return Status::error(StatusCode::invalid_argument, "fold_conv_bn bias size mismatch", "bias");
    }
    if (bn_mean.size() != out_c || bn_var.size() != out_c || bn_gamma.size() != out_c || bn_beta.size() != out_c) {
        return Status::error(StatusCode::invalid_argument, "fold_conv_bn BN param size mismatch", "bn");
    }
    for (uint32_t c = 0; c < out_c; c++) {
        if (!std::isfinite(bn_var[c]) || bn_var[c] < 0.0f) {
            return Status::error(StatusCode::invalid_argument, "fold_conv_bn variance must be finite and >= 0", "bn_var");
        }
        if (!std::isfinite(bn_mean[c]) || !std::isfinite(bn_gamma[c]) || !std::isfinite(bn_beta[c])) {
            return Status::error(StatusCode::invalid_argument, "fold_conv_bn BN params must be finite", "bn");
        }
    }
    out_weight.resize(static_cast<size_t>(total));
    out_bias.resize(out_c);
    for (uint32_t c = 0; c < out_c; c++) {
        const double scale = static_cast<double>(bn_gamma[c]) /
                             std::sqrt(static_cast<double>(bn_var[c]) + static_cast<double>(eps));
        const double b = bias.empty() ? 0.0 : static_cast<double>(bias[c]);
        out_bias[c] = static_cast<float>((b - static_cast<double>(bn_mean[c])) * scale +
                                         static_cast<double>(bn_beta[c]));
        const uint64_t base = static_cast<uint64_t>(c) * per_channel;
        for (uint64_t j = 0; j < per_channel; j++) {
            out_weight[static_cast<size_t>(base + j)] =
                static_cast<float>(static_cast<double>(weight[static_cast<size_t>(base + j)]) * scale);
        }
    }
    return Status::success();
}

[[nodiscard]] Status pp_fold_replk(luisa::span<const float> large_weight, luisa::span<const float> large_bias,
                                   luisa::span<const float> small_weight, luisa::span<const float> small_bias,
                                   luisa::span<const float> identity_gamma, luisa::span<const float> identity_beta,
                                   uint32_t channels, uint32_t large_k, uint32_t small_k,
                                   luisa::vector<float> &out_weight, luisa::vector<float> &out_bias) {
    if (channels == 0u || large_k == 0u || small_k == 0u) {
        return Status::error(StatusCode::invalid_argument, "fold_replk dimensions must be nonzero", "dims");
    }
    if (large_k % 2u == 0u || small_k % 2u == 0u || small_k > large_k) {
        return Status::error(StatusCode::invalid_argument, "fold_replk kernels must be odd and small<=large", "kernel");
    }
    const uint64_t large_total = static_cast<uint64_t>(channels) * large_k * large_k;
    const uint64_t small_total = static_cast<uint64_t>(channels) * small_k * small_k;
    if (large_weight.size() != large_total || small_weight.size() != small_total) {
        return Status::error(StatusCode::invalid_argument, "fold_replk weight size mismatch", "weight");
    }
    if (large_bias.size() != channels || small_bias.size() != channels ||
        identity_gamma.size() != channels || identity_beta.size() != channels) {
        return Status::error(StatusCode::invalid_argument, "fold_replk bias/identity size mismatch", "bias");
    }
    out_weight.assign(large_weight.begin(), large_weight.end());
    out_bias.resize(channels);
    const uint32_t offset = (large_k - small_k) / 2u;
    const uint32_t center = large_k / 2u;
    for (uint32_t c = 0; c < channels; c++) {
        // Add the zero-padded small kernel into the center of the large one.
        for (uint32_t ky = 0; ky < small_k; ky++) {
            for (uint32_t kx = 0; kx < small_k; kx++) {
                const uint64_t li = (static_cast<uint64_t>(c) * large_k + (ky + offset)) * large_k + (kx + offset);
                const uint64_t si = (static_cast<uint64_t>(c) * small_k + ky) * small_k + kx;
                out_weight[static_cast<size_t>(li)] = static_cast<float>(
                    static_cast<double>(out_weight[static_cast<size_t>(li)]) +
                    static_cast<double>(small_weight[static_cast<size_t>(si)]));
            }
        }
        // Identity BN branch: centered impulse scaled by gamma.
        const uint64_t ci = (static_cast<uint64_t>(c) * large_k + center) * large_k + center;
        out_weight[static_cast<size_t>(ci)] = static_cast<float>(
            static_cast<double>(out_weight[static_cast<size_t>(ci)]) +
            static_cast<double>(identity_gamma[c]));
        out_bias[c] = static_cast<float>(static_cast<double>(large_bias[c]) +
                                         static_cast<double>(small_bias[c]) +
                                         static_cast<double>(identity_beta[c]));
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Real-artifact gate + training guard
// ---------------------------------------------------------------------------

[[nodiscard]] Status pp_ocrv6_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                     luisa::string_view config_fingerprint) {
    (void)checkpoint_revision;
    (void)config_fingerprint;
    return Status::error(StatusCode::unvalidated,
                         "no official PP-OCRv6 checkpoint has been qualified; real artifacts fail closed "
                         "(only the declared synthetic contract is exercised in Phase 5)",
                         "pp_ocrv6");
}

[[nodiscard]] Status pp_ocrv6_check_no_training_nodes(luisa::span<const luisa::string> component_roles) {
    for (const auto &role : component_roles) {
        // Forbidden training/NRTR markers (case-insensitive substring match).
        luisa::string lower;
        lower.reserve(role.size());
        for (char ch : role) {
            lower += (ch >= 'A' && ch <= 'Z') ? static_cast<char>(ch - 'A' + 'a') : ch;
        }
        static constexpr const char *kForbidden[] = {"nrtr", "dropout", "loss", "train", "grpo", "reparam_train"};
        for (const auto *marker : kForbidden) {
            if (lower.find(marker) != luisa::string::npos) {
                return Status::error(StatusCode::invalid_argument,
                                     luisa::format("pp_ocrv6 contract carries a forbidden training/NRTR role '{}'", role),
                                     role);
            }
        }
    }
    return Status::success();
}

}// namespace luisa::ocr