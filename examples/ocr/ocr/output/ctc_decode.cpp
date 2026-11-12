// OCR engine -- CTC greedy decode implementation (Phase 5). Host-only.
#include "ocr/output/ctc_decode.h"

#include <cmath>

#include <luisa/core/stl/format.h>

namespace luisa::ocr::output {

[[nodiscard]] Status CtcDictionary::validate() const {
    if (characters.empty()) {
        return Status::error(StatusCode::invalid_argument,
                             "ctc dictionary must contain at least one non-blank character", "characters");
    }
    if (blank_id >= num_classes()) {
        return Status::error(StatusCode::invalid_argument, "ctc blank id outside the class range", "blank_id");
    }
    for (const auto &c : characters) {
        if (c.empty()) {
            return Status::error(StatusCode::invalid_argument,
                                 "ctc dictionary entries must be non-empty", "characters");
        }
    }
    return Status::success();
}

[[nodiscard]] const luisa::string *CtcDictionary::entry(uint32_t id) const noexcept {
    if (id >= num_classes() || id == blank_id) { return nullptr; }
    // Class ids below the blank map directly; ids above it shift down by one
    // (the dictionary excludes the blank entry).
    const uint32_t index = id < blank_id ? id : id - 1u;
    return &characters[index];
}

[[nodiscard]] Status CtcDecodeOptions::validate() const {
    if (!std::isfinite(temperature) || temperature <= 0.0f) {
        return Status::error(StatusCode::invalid_argument,
                             "ctc decode temperature must be finite and strictly positive", "temperature");
    }
    return Status::success();
}

[[nodiscard]] Result<CtcDecodeResult> ctc_decode_greedy(luisa::span<const int32_t> ids,
                                                        luisa::span<const float> probs,
                                                        const CtcDictionary &dict) {
    using R = Result<CtcDecodeResult>;
    if (auto s = dict.validate(); !s.ok()) { return R::failure(s); }
    if (ids.size() != probs.size()) {
        return R::failure(StatusCode::invalid_argument,
                          "ctc ids/probs frame counts differ", "ids/probs");
    }
    CtcDecodeResult result;
    const uint32_t classes = dict.num_classes();
    double prob_sum = 0.0;
    bool have_prev = false;
    int32_t prev_id = -1;
    for (size_t t = 0; t < ids.size(); t++) {
        const int32_t raw = ids[t];
        if (raw < 0 || static_cast<uint32_t>(raw) >= classes) {
            return R::failure(StatusCode::invalid_argument,
                              luisa::format("ctc frame {} has invalid class id {} (classes {})", t, raw, classes),
                              "ids");
        }
        const float p = probs[t];
        if (!std::isfinite(p) || p < 0.0f || p > 1.0f) {
            return R::failure(StatusCode::invalid_argument,
                              luisa::format("ctc frame {} has invalid probability {}", t, p), "probs");
        }
        const uint32_t id = static_cast<uint32_t>(raw);
        // Consecutive duplicate collapse on the RAW sequence (before blank
        // removal): a frame equal to the previous RAW frame is dropped even
        // when that previous frame was blank (blank runs collapse too).
        const bool duplicate = have_prev && raw == prev_id;
        have_prev = true;
        prev_id = raw;
        if (duplicate || id == dict.blank_id) { continue; }
        const auto *entry = dict.entry(id);
        if (entry == nullptr) {
            return R::failure(StatusCode::internal_error, "ctc dictionary lookup failed", "characters");
        }
        result.text += *entry;
        result.ids.push_back(id);
        result.kept_frames.push_back(CtcKeptFrame{static_cast<uint32_t>(t), id, p});
        prob_sum += static_cast<double>(p);
    }
    if (!result.ids.empty()) {
        result.has_confidence = true;
        result.confidence = static_cast<float>(prob_sum / static_cast<double>(result.ids.size()));
    }
    return R::success(std::move(result));
}

[[nodiscard]] Result<CtcDecodeResult> ctc_decode_logits(luisa::span<const float> logits,
                                                        uint32_t frames,
                                                        const CtcDictionary &dict,
                                                        const CtcDecodeOptions &options) {
    using R = Result<CtcDecodeResult>;
    if (auto s = dict.validate(); !s.ok()) { return R::failure(s); }
    if (auto s = options.validate(); !s.ok()) { return R::failure(s); }
    const uint32_t classes = dict.num_classes();
    if (frames != 0u && logits.size() != static_cast<size_t>(frames) * classes) {
        return R::failure(StatusCode::invalid_argument, "ctc logits span size mismatch", "logits");
    }
    if (frames == 0u) {
        return R::success(CtcDecodeResult{});
    }
    luisa::vector<int32_t> ids(frames);
    luisa::vector<float> probs(frames);
    const double temperature = static_cast<double>(options.temperature);
    for (uint32_t t = 0; t < frames; t++) {
        const size_t base = static_cast<size_t>(t) * classes;
        // Validate finiteness first: NaN/inf logits are invalid model output.
        for (uint32_t c = 0; c < classes; c++) {
            if (!std::isfinite(logits[base + c])) {
                return R::failure(StatusCode::invalid_argument,
                                  luisa::format("ctc frame {} class {} has a non-finite logit", t, c), "logits");
            }
        }
        uint32_t best = 0;
        double best_v = static_cast<double>(logits[base]) / temperature;
        double max_v = best_v;
        for (uint32_t c = 1; c < classes; c++) {
            const double v = static_cast<double>(logits[base + c]) / temperature;
            if (v > best_v) {
                best_v = v;
                best = c;
            }
            max_v = std::max(max_v, v);
        }
        double sum = 0.0;
        for (uint32_t c = 0; c < classes; c++) {
            sum += std::exp(static_cast<double>(logits[base + c]) / temperature - max_v);
        }
        ids[t] = static_cast<int32_t>(best);
        probs[t] = static_cast<float>(std::exp(best_v - max_v) / sum);
    }
    return ctc_decode_greedy(ids, probs, dict);
}

[[nodiscard]] luisa::string join_line_texts(luisa::span<const luisa::string> lines) {
    luisa::string text;
    bool first = true;
    for (const auto &line : lines) {
        if (line.empty()) { continue; }
        if (!first) { text += '\n'; }
        text += line;
        first = false;
    }
    return text;
}

}// namespace luisa::ocr::output