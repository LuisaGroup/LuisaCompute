// OCR engine -- Surya device executor (Phase 9). Detector (EfficientViT-
// Segformer-style CNN + DB head) -> line crops -> vision merger -> hybrid GDN
// decode. Reuses the shared CNN conv/activation/sigmoid ops + output/db
// postprocess, and the HybridVlmGraph (vision + hybrid decode) per line crop.
// SYNTHETIC ONLY; real artifacts fail closed in models/surya.*.
#include "ocr/runtime/surya_executor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>

#include "ocr/quantization.h"

namespace luisa::ocr {

using namespace luisa::compute;

namespace {

// Host STRETCH-resize + normalize of an HWC uint8 page to CHW float at the
// detector input size (independent x/y scales, half-pixel centers, clamp edges;
// NO aspect preservation, NO padding). Declared SYNTHETIC convention (same
// bilinear math as the CNN detector preprocessing).
[[nodiscard]] luisa::vector<float> preprocess_page_host(const SuryaPageInput &page, uint32_t dw, uint32_t dh,
                                                        luisa::span<const float> mean, luisa::span<const float> std_dev) {
    const uint32_t sw = page.source_width, sh = page.source_height;
    luisa::vector<float> out(static_cast<size_t>(3) * dw * dh);
    const float scale_x = static_cast<float>(sw) / static_cast<float>(dw);
    const float scale_y = static_cast<float>(sh) / static_cast<float>(dh);
    const auto sample = [&](uint32_t c, float sx, float sy) -> float {
        const float x0f = std::floor(sx);
        const float y0f = std::floor(sy);
        const int32_t x0 = static_cast<int32_t>(x0f);
        const int32_t y0 = static_cast<int32_t>(y0f);
        const float fx = sx - x0f;
        const float fy = sy - y0f;
        const auto at = [&](int32_t px, int32_t py) -> float {
            const int32_t cx = std::clamp(px, 0, static_cast<int32_t>(sw) - 1);
            const int32_t cy = std::clamp(py, 0, static_cast<int32_t>(sh) - 1);
            return static_cast<float>(page.pixels[(static_cast<size_t>(cy) * sw + cx) * 3u + c]);
        };
        const float v00 = at(x0, y0);
        const float v10 = at(x0 + 1, y0);
        const float v01 = at(x0, y0 + 1);
        const float v11 = at(x0 + 1, y0 + 1);
        const float top = v00 + (v10 - v00) * fx;
        const float bottom = v01 + (v11 - v01) * fx;
        return top + (bottom - top) * fy;
    };
    for (uint32_t y = 0; y < dh; y++) {
        for (uint32_t x = 0; x < dw; x++) {
            const float sx = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const float sy = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
            for (uint32_t c = 0; c < 3u; c++) {
                const float v = sample(c, sx, sy) / 255.0f;
                const float normalized = (v - mean[c]) / std_dev[c];
                out[static_cast<size_t>(c) * dw * dh + static_cast<size_t>(y) * dw + x] = normalized;
            }
        }
    }
    return out;
}

// Crop an axis-aligned rect (source coordinates, clamped) from an HWC uint8 page
// into a new HWC uint8 buffer. Enforces a minimum side so the vision tower always
// has a non-degenerate crop. Returns the cropped pixels + actual size.
struct LineCrop {
    luisa::vector<uint8_t> pixels;
    uint32_t width{0};
    uint32_t height{0};
};
[[nodiscard]] LineCrop crop_line(const SuryaPageInput &page, const output::DbRect &bounds) {
    const int32_t sw = static_cast<int32_t>(page.source_width);
    const int32_t sh = static_cast<int32_t>(page.source_height);
    int32_t x0 = static_cast<int32_t>(std::floor(bounds.x0));
    int32_t y0 = static_cast<int32_t>(std::floor(bounds.y0));
    int32_t x1 = static_cast<int32_t>(std::ceil(bounds.x1));
    int32_t y1 = static_cast<int32_t>(std::ceil(bounds.y1));
    x0 = std::clamp(x0, 0, sw - 1);
    y0 = std::clamp(y0, 0, sh - 1);
    x1 = std::clamp(x1, x0 + 1, sw);
    y1 = std::clamp(y1, y0 + 1, sh);
    // Enforce a minimum 4x4 crop (expand around the center, clamped to source).
    const int32_t min_side = 4;
    if (x1 - x0 < min_side) {
        const int32_t need = min_side - (x1 - x0);
        x0 = std::max(0, x0 - need / 2);
        x1 = std::min(sw, x0 + min_side);
        x0 = std::max(0, x1 - min_side);
    }
    if (y1 - y0 < min_side) {
        const int32_t need = min_side - (y1 - y0);
        y0 = std::max(0, y0 - need / 2);
        y1 = std::min(sh, y0 + min_side);
        y0 = std::max(0, y1 - min_side);
    }
    LineCrop crop;
    crop.width = static_cast<uint32_t>(x1 - x0);
    crop.height = static_cast<uint32_t>(y1 - y0);
    crop.pixels.resize(static_cast<size_t>(crop.width) * crop.height * 3u);
    for (uint32_t y = 0; y < crop.height; y++) {
        for (uint32_t x = 0; x < crop.width; x++) {
            const size_t src = (static_cast<size_t>(y0 + y) * page.source_width + (x0 + x)) * 3u;
            const size_t dst = (static_cast<size_t>(y) * crop.width + x) * 3u;
            crop.pixels[dst] = page.pixels[src];
            crop.pixels[dst + 1] = page.pixels[src + 1];
            crop.pixels[dst + 2] = page.pixels[src + 2];
        }
    }
    return crop;
}

}// namespace

// ---------------------------------------------------------------------------
// create / resolve / reserve / compile
// ---------------------------------------------------------------------------

[[nodiscard]] Status SuryaExecutor::create(Device &device, Stream &stream,
                                           const SuryaConfig &config, const WeightStore &weights) {
    auto valid = config.validate();
    if (!valid.ok()) { return valid; }
    // S2 (Phase 9 review): Surya is a TEXT recognizer. Without a tokenizer every
    // line's text would be silently empty, so a text-capable run must never be
    // created in that state. Require a borrowed tokenizer here (config.validate()
    // stays a pure contract check so parse/signature/preflight work without one).
    if (config.tokenizer == nullptr) {
        return Status::error(StatusCode::invalid_argument,
                             "Surya requires a borrowed tokenizer (SuryaConfig::tokenizer) to produce line text; "
                             "it must outlive create() and every recognize() call",
                             "tokenizer");
    }
    if (config.tokenizer->vocab_size() < config.decoder.decoder.vocab_size) {
        return Status::error(StatusCode::invalid_argument,
                             luisa::format("Surya tokenizer vocabulary {} is smaller than the decoder vocabulary {}",
                                           config.tokenizer->vocab_size(), config.decoder.decoder.vocab_size),
                             "tokenizer");
    }
    _config = config;
    _device = &device;
    _det_ws = DeviceWorkspace{device};
    _conv_cache = TypedShaderCache<Conv2dKernel>{device, _shader_registry};
    _act_cache = TypedShaderCache<CnnActKernel>{device, _shader_registry};
    _sigmoid_cache = TypedShaderCache<ElementwiseSigmoidKernel>{device, _shader_registry};
    if (auto s = resolve_weights(weights); !s.ok()) { return s; }
    if (auto s = reserve_workspace(); !s.ok()) { return s; }
    if (auto s = compile_shaders(); !s.ok()) { return s; }
    // Vision + hybrid decode graph for line crops.
    HybridVlmGraphConfig gc;
    gc.vision = config.vision;
    gc.vision_names = config.vision_names;
    gc.decoder = config.decoder;
    gc.decoder_names = config.decoder_names;
    gc.max_seq_len = config.max_seq_len;
    gc.memory = config.memory;
    if (auto s = _line_graph.create(device, stream, gc, weights); !s.ok()) { return s; }
    _weight_bytes = weights.component_bytes("detector") + _line_graph.weight_bytes();
    return Status::success();
}

[[nodiscard]] Status SuryaExecutor::resolve_weights(const WeightStore &weights) {
    const auto &dn = _config.detector_names;
    const auto &det = _config.detector;
    const auto expected = surya_expected_tensors(_config);
    luisa::unordered_map<luisa::string, const ExpectedTensor *> expected_by_name;
    for (const auto &e : expected) {
        if (e.name.compare(0, dn.prefix.size(), dn.prefix) == 0) { expected_by_name.emplace(e.name, &e); }
    }
    for (const auto &name : weights.tensor_names("detector")) {
        if (expected_by_name.find(name) == expected_by_name.end()) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("unexpected detector tensor '{}'", name), name);
        }
    }
    auto view_of = [&](const luisa::string &name, BufferView<float> &out) -> Status {
        const auto *w = weights.find("detector", name);
        if (w == nullptr) {
            return Status::error(StatusCode::not_found, luisa::format("detector weight '{}' not found", name), name);
        }
        if (w->encoding != TensorEncoding::fp32 || !w->dense) {
            return Status::error(StatusCode::unsupported, luisa::format("detector weight '{}' must be dense fp32", name), name);
        }
        const auto it = expected_by_name.find(name);
        if (it == expected_by_name.end()) {
            return Status::error(StatusCode::invalid_argument, luisa::format("detector weight '{}' has no contract", name), name);
        }
        if (auto s = validate_decoder_weight_desc(w->logical, w->element_count, *it->second); !s.ok()) { return s; }
        out = w->dense.view();
        return Status::success();
    };
    if (auto s = view_of(dn.stem_weight(), _stem.weight); !s.ok()) { return s; }
    if (auto s = view_of(dn.stem_bias(), _stem.bias); !s.ok()) { return s; }
    if (auto s = view_of(dn.stage1_weight(), _stage1.weight); !s.ok()) { return s; }
    if (auto s = view_of(dn.stage1_bias(), _stage1.bias); !s.ok()) { return s; }
    if (auto s = view_of(dn.stage2_weight(), _stage2.weight); !s.ok()) { return s; }
    if (auto s = view_of(dn.stage2_bias(), _stage2.bias); !s.ok()) { return s; }
    if (auto s = view_of(dn.db_weight(), _db.weight); !s.ok()) { return s; }
    if (auto s = view_of(dn.db_bias(), _db.bias); !s.ok()) { return s; }
    (void)det;
    return Status::success();
}

[[nodiscard]] Status SuryaExecutor::reserve_workspace() {
    const auto &det = _config.detector;
    // Compute each stage's output size; the ping-pong buffers hold the largest
    // stage output. The probability map is the DB head output.
    uint32_t h = det.input_height, w = det.input_width;
    uint64_t max_stage_elems = 0;
    uint32_t map_h = 0, map_w = 0;
    const SuryaConvStage *stages[] = {&det.stem, &det.stage1, &det.stage2, &det.db_head};
    for (const auto *s : stages) {
        auto oh = conv2d_out_dim(h, s->k, s->stride, s->pad, 1u);
        auto ow = conv2d_out_dim(w, s->k, s->stride, s->pad, 1u);
        if (!oh.ok() || !ow.ok()) {
            return Status::error(StatusCode::invalid_argument, "detector stage output dim invalid", "stage");
        }
        h = *oh.value;
        w = *ow.value;
        const uint64_t elems = static_cast<uint64_t>(s->out_c) * h * w;
        max_stage_elems = std::max(max_stage_elems, elems);
    }
    map_h = h;
    map_w = w;
    const uint64_t pixel_elems = static_cast<uint64_t>(3) * det.input_height * det.input_width;
    const uint64_t prob_elems = static_cast<uint64_t>(map_h) * map_w;
    struct Spec { WorkspaceAllocation *out; uint64_t elems; };
    const Spec specs[] = {
        {&_det_pixels, pixel_elems},
        {&_det_a, max_stage_elems},
        {&_det_b, max_stage_elems},
        {&_det_prob, prob_elems},
    };
    const auto align = [](uint64_t bytes) { return (bytes + kWorkspaceAlignment - 1u) & ~(kWorkspaceAlignment - 1u); };
    uint64_t total = 0;
    for (const auto &s : specs) {
        uint64_t bytes = 0;
        if (!checked_mul(s.elems, 4ull, bytes) || !checked_add(total, align(bytes), total)) {
            return Status::error(StatusCode::out_of_memory, "detector workspace overflows", "workspace");
        }
    }
    if (auto s = _det_ws.reserve(total); !s.ok()) { return s; }
    for (const auto &s : specs) {
        auto r = _det_ws.allocate(s.elems * 4ull);
        if (!r.ok()) { return r.status; }
        *s.out = *r.value;
    }
    return Status::success();
}

[[nodiscard]] Status SuryaExecutor::compile_shaders() {
    const auto get = [](auto &cache, const luisa::string &key, auto factory, auto *&out) -> Status {
        auto r = cache.get_or_compile(key, factory);
        if (!r.ok()) { return r.status; }
        out = *r.value;
        return Status::success();
    };
    if (auto s = get(_conv_cache, conv_variant_key("conv2d"),
                     [] { return make_conv2d_kernel(); }, _sh_conv); !s.ok()) { return s; }
    for (uint32_t act = 0; act < 4u; act++) {
        if (auto s = get(_act_cache, ctc_variant_key("cnn_act") + luisa::format("_{}", act),
                         [act] { return make_cnn_act_kernel(act); }, _sh_act[act]); !s.ok()) { return s; }
    }
    if (auto s = get(_sigmoid_cache, gdn_variant_key("elementwise_sigmoid"),
                     [] { return make_elementwise_sigmoid_kernel(); }, _sh_sigmoid); !s.ok()) { return s; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// detector recording
// ---------------------------------------------------------------------------

[[nodiscard]] Status SuryaExecutor::record_conv(CommandList &list, const SuryaConvStage &stage,
                                                const ConvWeights &w, BufferView<float> in,
                                                BufferView<float> out, uint32_t in_h, uint32_t in_w,
                                                uint32_t &out_h, uint32_t &out_w) {
    auto oh = conv2d_out_dim(in_h, stage.k, stage.stride, stage.pad, 1u);
    auto ow = conv2d_out_dim(in_w, stage.k, stage.stride, stage.pad, 1u);
    if (!oh.ok()) { return oh.status; }
    if (!ow.ok()) { return ow.status; }
    out_h = *oh.value;
    out_w = *ow.value;
    if (auto s = validate_conv2d_inputs(1u, stage.in_c, in_h, in_w, stage.out_c, stage.k, stage.k,
                                        stage.stride, stage.stride, 1u, 1u, 1u); !s.ok()) { return s; }
    auto plan = plan_conv2d_dispatch(1u, stage.out_c, out_h, out_w);
    if (!plan.ok()) { return plan.status; }
    if (auto cmd = dispatch_1d((*_sh_conv)(in, w.weight, w.bias, out, 1u, stage.in_c, in_h, in_w,
                                           stage.out_c, stage.k, stage.k, stage.stride, stage.stride,
                                           stage.pad, stage.pad, 1u, 1u, 1u, out_h, out_w, 1u), *plan.value)) {
        list << std::move(cmd);
    }
    if (stage.act != 0u) {
        const uint64_t count = static_cast<uint64_t>(stage.out_c) * out_h * out_w;
        auto act_plan = plan_cnn_act_dispatch(count);
        if (!act_plan.ok()) { return act_plan.status; }
        auto count32 = checked_u32(count, "surya act count");
        if (!count32.ok()) { return count32.status; }
        if (auto cmd = dispatch_1d((*_sh_act[stage.act])(out, out, *count32.value), *act_plan.value)) {
            list << std::move(cmd);
        }
    }
    return Status::success();
}

[[nodiscard]] Status SuryaExecutor::record_detector(CommandList &list, BufferView<float> pixels,
                                                    uint32_t &map_h, uint32_t &map_w) {
    const auto &det = _config.detector;
    uint32_t h = det.input_height, w = det.input_width;
    uint32_t oh = 0, ow = 0;
    // stem: pixels -> _det_a
    if (auto s = record_conv(list, det.stem, _stem, pixels, dws(_det_a), h, w, oh, ow); !s.ok()) { return s; }
    h = oh; w = ow;
    // stage1: _det_a -> _det_b
    if (auto s = record_conv(list, det.stage1, _stage1, dws(_det_a), dws(_det_b), h, w, oh, ow); !s.ok()) { return s; }
    h = oh; w = ow;
    // stage2: _det_b -> _det_a
    if (auto s = record_conv(list, det.stage2, _stage2, dws(_det_b), dws(_det_a), h, w, oh, ow); !s.ok()) { return s; }
    h = oh; w = ow;
    // db_head: _det_a -> _det_prob (1 channel logit)
    if (auto s = record_conv(list, det.db_head, _db, dws(_det_a), dws(_det_prob), h, w, oh, ow); !s.ok()) { return s; }
    // sigmoid -> probability map
    const uint64_t count = static_cast<uint64_t>(oh) * ow;
    auto plan = plan_gdn_elementwise_dispatch(count);
    if (!plan.ok()) { return plan.status; }
    auto count32 = checked_u32(count, "surya sigmoid count");
    if (!count32.ok()) { return count32.status; }
    if (auto cmd = dispatch_1d((*_sh_sigmoid)(dws(_det_prob), dws(_det_prob), *count32.value), *plan.value)) {
        list << std::move(cmd);
    }
    map_h = oh;
    map_w = ow;
    return Status::success();
}

// ---------------------------------------------------------------------------
// recognize: detector -> line crops -> vision -> hybrid decode
// ---------------------------------------------------------------------------

[[nodiscard]] Result<SuryaRunResult> SuryaExecutor::recognize(Stream &stream, luisa::span<const SuryaPageInput> pages) {
    using R = Result<SuryaRunResult>;
    if (_device == nullptr) { return R::failure(StatusCode::invalid_argument, "surya executor not created", "executor"); }
    // S2: the borrowed tokenizer was required at create(); re-check it here so a
    // caller that cleared it cannot get silently empty text.
    if (_config.tokenizer == nullptr) {
        return R::failure(StatusCode::invalid_argument,
                          "Surya recognize requires the borrowed tokenizer to remain set (lifetime contract)",
                          "tokenizer");
    }
    SuryaRunResult result;
    result.weight_bytes = _weight_bytes;
    result.workspace_bytes = _det_ws.capacity_bytes();
    uint64_t cache_peak = 0;
    for (const auto &page : pages) {
        if (page.pixels.size() != static_cast<size_t>(page.source_width) * page.source_height * 3u) {
            return R::failure(StatusCode::invalid_argument, "page pixel buffer size mismatch", "pixels");
        }
        if (page.source_width == 0u || page.source_height == 0u) {
            return R::failure(StatusCode::invalid_argument, "degenerate page size", "size");
        }
        SuryaPageResult pr;
        pr.page_index = page.page_index;
        pr.source_width = page.source_width;
        pr.source_height = page.source_height;
        // 1. Preprocess + detector.
        auto pixels_chw = preprocess_page_host(page, _config.detector.input_width, _config.detector.input_height,
                                               _config.detector.mean, _config.detector.std_dev);
        uint32_t map_h = 0, map_w = 0;
        {
            auto pixels_view = dws(_det_pixels);
            auto staging = luisa::make_unique<luisa::vector<float>>(std::move(pixels_chw));
            auto *stage_ptr = staging.get();
            auto list = CommandList::create();
            list << pixels_view.copy_from(luisa::span<const float>{*stage_ptr});
            if (auto s = record_detector(list, pixels_view, map_h, map_w); !s.ok()) { return R::failure(s); }
            list.add_callback([staging = std::move(staging)]() mutable { staging.reset(); });
            stream << list.commit() << synchronize();
        }
        _last_map_h = map_h;
        _last_map_w = map_w;
        pr.det_map_width = map_w;
        pr.det_map_height = map_h;
        // Download the probability map.
        pr.det_probabilities.resize(static_cast<size_t>(map_h) * map_w);
        stream << dws(_det_prob).copy_to(luisa::span<float>{pr.det_probabilities}) << synchronize();
        // 2. DB postprocess -> boxes in SOURCE coordinates.
        output::DbSourceTransform transform;
        transform.scale_x = static_cast<float>(page.source_width) / static_cast<float>(map_w);
        transform.scale_y = static_cast<float>(page.source_height) / static_cast<float>(map_h);
        transform.offset_x = 0.0f;
        transform.offset_y = 0.0f;
        auto db = output::db_postprocess(pr.det_probabilities, map_w, map_h, transform, _config.db);
        if (!db.ok()) { return R::failure(db.status); }
        pr.boxes = std::move(db.value->boxes);
        // Bound the number of lines (largest-area first is handled by db max_boxes;
        // here we cap to max_lines_per_page in detection order).
        if (pr.boxes.size() > _config.max_lines_per_page) {
            pr.boxes.resize(_config.max_lines_per_page);
        }
        // 3. Per-line: crop -> vision -> hybrid decode.
        for (const auto &box : pr.boxes) {
            auto crop = crop_line(page, box.bounds);
            P7ImageInput img;
            img.pixels = std::move(crop.pixels);
            img.source_width = crop.width;
            img.source_height = crop.height;
            auto packed = p7_pack_image(crop.width, crop.height, _config.vision.packing, _config.vision.connector);
            if (!packed.ok()) { return R::failure(packed.status); }
            img.packed = *packed.value;
            const uint32_t vt = static_cast<uint32_t>(img.packed.visual_tokens);
            // Prompt: [prefix][visual rows][suffix].
            luisa::vector<uint32_t> prompt = {1u};
            const uint32_t mm_start = 1u;
            for (uint32_t i = 0; i < vt; i++) { prompt.push_back(0u); }
            prompt.push_back(2u);
            HybridDecodeOptions opts;
            opts.sampling.greedy = true;
            opts.max_new_tokens = _config.max_new_tokens;
            // EOS comes from the DECLARED decoder contract (never hardcoded);
            // the controller also honors the tokenizer's own EOS.
            opts.eos_ids = _config.decoder.decoder.eos_token_ids;
            // S2: pass the BORROWED tokenizer so the line is actually
            // detokenized. Lifetime: it outlives create() and this call (checked
            // at create); the graph never retains it past generate().
            auto run = _line_graph.run(stream, luisa::span<const P7ImageInput>{&img, 1}, prompt, mm_start, opts,
                                       _config.tokenizer);
            if (!run.ok()) { return R::failure(run.status); }
            cache_peak = std::max(cache_peak, run.value->cache_bytes);
            // A text-capable run must never silently emit empty content: if the
            // decoder produced tokens but detokenization yielded nothing, that
            // is a real failure (tokenizer/contract mismatch), not a blank line.
            // (A genuinely token-less line -- EOS on the first step -- is valid
            // and stays empty; blank PAGES have no lines at all.)
            if (!run.value->decode.tokens.empty() && run.value->decode.text.empty() &&
                !run.value->decode.hit_eos && !run.value->decode.hit_stop_string) {
                return R::failure(StatusCode::internal_error,
                                  luisa::format("Surya line {} produced {} tokens but no text; "
                                                "the tokenizer does not match the decoder vocabulary",
                                                pr.lines.size(), run.value->decode.tokens.size()),
                                  "tokenizer");
            }
            SuryaLine line;
            line.text = run.value->decode.text;
            line.tokens = run.value->decode.tokens;
            line.page_index = page.page_index;
            line.bounds = box.bounds;
            line.polygon = box.polygon;
            // Confidence is the DETECTOR mean probability (an explicitly defined
            // reference quantity), NOT a generative probability.
            line.has_confidence = true;
            line.confidence = box.score;
            line.truncated = run.value->decode.truncated;
            line.hit_eos = run.value->decode.hit_eos;
            pr.lines.push_back(std::move(line));
        }
        // 4. Reading order: top-to-bottom, then left-to-right (deterministic).
        std::stable_sort(pr.lines.begin(), pr.lines.end(), [](const SuryaLine &a, const SuryaLine &b) {
            if (a.bounds.y0 != b.bounds.y0) { return a.bounds.y0 < b.bounds.y0; }
            return a.bounds.x0 < b.bounds.x0;
        });
        for (uint32_t i = 0; i < pr.lines.size(); i++) { pr.lines[i].reading_order = i; }
        // 5. Canonical joined text.
        for (uint32_t i = 0; i < pr.lines.size(); i++) {
            if (i != 0u) { pr.text += "\n"; }
            pr.text += pr.lines[i].text;
        }
        result.pages.push_back(std::move(pr));
    }
    result.cache_bytes = cache_peak;
    return R::success(std::move(result));
}

[[nodiscard]] Status SuryaExecutor::download_detector_prob(Stream &stream, luisa::vector<float> &out) const {
    if (_last_map_h == 0u || _last_map_w == 0u) {
        return Status::error(StatusCode::invalid_argument, "no detector run to download", "detector");
    }
    out.resize(static_cast<size_t>(_last_map_h) * _last_map_w);
    stream << dws(_det_prob).copy_to(luisa::span<float>{out});
    return Status::success();
}

}// namespace luisa::ocr
