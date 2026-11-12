// OCR engine -- full RecognizeResult JSON serialization (Phase 1).
// yyjson with correct copy-vs-borrow ownership; non-finite floats rejected.
#include "ocr/output/json_output.h"

#include <cmath>
#include <cstdio>
#include <utility>

#include <yyjson.h>

namespace luisa::ocr::output {
namespace {

// RAII mutable document.
struct MutDoc {
    yyjson_mut_doc *doc{nullptr};
    explicit MutDoc() noexcept : doc{yyjson_mut_doc_new(nullptr)} {}
    ~MutDoc() noexcept {
        if (doc != nullptr) { yyjson_mut_doc_free(doc); }
    }
    MutDoc(const MutDoc &) = delete;
    MutDoc &operator=(const MutDoc &) = delete;
};

// All dynamic strings are COPIED into the document (strncpy). Keys are string
// literals (static storage, safe to borrow).
void add_str(yyjson_mut_doc *doc, yyjson_mut_val *obj, const char *key, luisa::string_view value) {
    yyjson_mut_obj_add_strncpy(doc, obj, key, value.data(), value.size());
}

// Stable policy names (the policy enums live in api.h; these render them).
[[nodiscard]] luisa::string_view cache_policy_name(CachePolicy policy) noexcept {
    switch (policy) {
        case CachePolicy::automatic: return "automatic";
        case CachePolicy::contiguous: return "contiguous";
        case CachePolicy::paged: return "paged";
        default: return "unknown";
    }
}
[[nodiscard]] luisa::string_view offload_policy_name(OffloadPolicy policy) noexcept {
    switch (policy) {
        case OffloadPolicy::disabled: return "disabled";
        case OffloadPolicy::explicit_host: return "explicit_host";
        default: return "unknown";
    }
}
[[nodiscard]] luisa::string_view graph_policy_name(GraphPolicy policy) noexcept {
    switch (policy) {
        case GraphPolicy::disabled: return "disabled";
        case GraphPolicy::automatic: return "automatic";
        case GraphPolicy::enabled: return "enabled";
        default: return "unknown";
    }
}
[[nodiscard]] luisa::string_view fast_path_policy_name(FastPathPolicy policy) noexcept {
    switch (policy) {
        case FastPathPolicy::disabled: return "disabled";
        case FastPathPolicy::automatic: return "automatic";
        case FastPathPolicy::enabled: return "enabled";
        default: return "unknown";
    }
}

[[nodiscard]] bool add_finite_real(yyjson_mut_doc *doc, yyjson_mut_val *obj, const char *key, double value) {
    if (!std::isfinite(value)) { return false; }
    yyjson_mut_obj_add_real(doc, obj, key, value);
    return true;
}

[[nodiscard]] bool add_source_region(yyjson_mut_doc *doc, yyjson_mut_val *obj, const char *key,
                                     const SourceRegion &region) {
    auto *region_obj = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_uint(doc, region_obj, "page_index", region.page_index);
    auto *box = yyjson_mut_obj(doc);
    if (!add_finite_real(doc, box, "x0", region.box.x0)) { return false; }
    if (!add_finite_real(doc, box, "y0", region.box.y0)) { return false; }
    if (!add_finite_real(doc, box, "x1", region.box.x1)) { return false; }
    if (!add_finite_real(doc, box, "y1", region.box.y1)) { return false; }
    yyjson_mut_obj_add_val(doc, region_obj, "box", box);
    auto *polygon = yyjson_mut_arr(doc);
    for (const auto &point : region.polygon) {
        auto *p = yyjson_mut_obj(doc);
        if (!add_finite_real(doc, p, "x", point.x)) { return false; }
        if (!add_finite_real(doc, p, "y", point.y)) { return false; }
        yyjson_mut_arr_append(polygon, p);
    }
    yyjson_mut_obj_add_val(doc, region_obj, "polygon", polygon);
    yyjson_mut_obj_add_val(doc, obj, key, region_obj);
    return true;
}

[[nodiscard]] bool add_metadata(yyjson_mut_doc *doc, yyjson_mut_val *obj, const ModelMetadata &meta) {
    auto *m = yyjson_mut_obj(doc);
    add_str(doc, m, "model_id", model_id_name(meta.model_id));
    add_str(doc, m, "pp_size", pp_ocr_size_name(meta.pp_size));
    add_str(doc, m, "target_name", meta.target_name);
    add_str(doc, m, "checkpoint_revision", meta.checkpoint_revision);
    add_str(doc, m, "package_hash", meta.package_hash);
    add_str(doc, m, "recipe_version", meta.recipe_version);
    add_str(doc, m, "backend", meta.backend);
    add_str(doc, m, "weight_precision", precision_name(meta.weight_precision));
    add_str(doc, m, "activation_precision", precision_name(meta.activation_precision));
    add_str(doc, m, "cache_precision", precision_name(meta.cache_precision));
    add_str(doc, m, "cache_policy", cache_policy_name(meta.cache_policy));
    add_str(doc, m, "offload_policy", offload_policy_name(meta.offload_policy));
    add_str(doc, m, "graph_policy", graph_policy_name(meta.graph_policy));
    add_str(doc, m, "fast_path_policy", fast_path_policy_name(meta.fast_path_policy));
    yyjson_mut_obj_add_uint(doc, m, "max_overview_pixels", meta.max_overview_pixels);
    yyjson_mut_obj_add_uint(doc, m, "max_crop_pixels", meta.max_crop_pixels);
    yyjson_mut_obj_add_uint(doc, m, "max_output_tokens_per_region", meta.max_output_tokens_per_region);
    yyjson_mut_obj_add_uint(doc, m, "max_output_tokens_per_page", meta.max_output_tokens_per_page);
    auto *optimizations = yyjson_mut_arr(doc);
    for (const auto &opt : meta.enabled_optimizations) {
        yyjson_mut_arr_add_strncpy(doc, optimizations, opt.data(), opt.size());
    }
    yyjson_mut_obj_add_val(doc, m, "enabled_optimizations", optimizations);
    add_str(doc, m, "support_state", support_state_name(meta.support_state));
    yyjson_mut_obj_add_bool(doc, m, "validated", meta.validated);
    auto *router = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_bool(doc, router, "used", meta.router.used);
    add_str(doc, router, "router_id", meta.router.router_id);
    add_str(doc, router, "initial_model", model_id_name(meta.router.initial_model));
    add_str(doc, router, "final_model", model_id_name(meta.router.final_model));
    add_str(doc, router, "reason", meta.router.reason);
    yyjson_mut_obj_add_val(doc, m, "router", router);
    yyjson_mut_obj_add_val(doc, obj, "metadata", m);
    return true;
}

[[nodiscard]] bool add_counters(yyjson_mut_doc *doc, yyjson_mut_val *obj, const PerformanceCounters &counters) {
    auto *c = yyjson_mut_obj(doc);
    if (!add_finite_real(doc, c, "load_ms", counters.load_ms)) { return false; }
    if (!add_finite_real(doc, c, "preprocess_ms", counters.preprocess_ms)) { return false; }
    if (!add_finite_real(doc, c, "vision_ms", counters.vision_ms)) { return false; }
    if (!add_finite_real(doc, c, "decode_ms", counters.decode_ms)) { return false; }
    if (!add_finite_real(doc, c, "postprocess_ms", counters.postprocess_ms)) { return false; }
    if (!add_finite_real(doc, c, "total_ms", counters.total_ms)) { return false; }
    yyjson_mut_obj_add_uint(doc, c, "pages_processed", counters.pages_processed);
    yyjson_mut_obj_add_uint(doc, c, "regions_processed", counters.regions_processed);
    yyjson_mut_obj_add_uint(doc, c, "patches_processed", counters.patches_processed);
    yyjson_mut_obj_add_uint(doc, c, "output_tokens", counters.output_tokens);
    yyjson_mut_obj_add_uint(doc, c, "peak_device_bytes", counters.peak_device_bytes);
    yyjson_mut_obj_add_uint(doc, c, "peak_host_bytes", counters.peak_host_bytes);
    yyjson_mut_obj_add_uint(doc, c, "resident_weight_bytes", counters.resident_weight_bytes);
    yyjson_mut_obj_add_val(doc, obj, "counters", c);
    return true;
}

[[nodiscard]] Result<luisa::string> write_doc(yyjson_mut_doc *doc, const JsonOutputOptions &options) {
    using R = Result<luisa::string>;
    const auto flags = options.pretty ? YYJSON_WRITE_PRETTY_TWO_SPACES : YYJSON_WRITE_NOFLAG;
    size_t length = 0;
    char *json = yyjson_mut_write(doc, flags, &length);
    if (json == nullptr) {
        return R::failure(StatusCode::out_of_memory, "cannot serialize JSON output", "json");
    }
    luisa::string out{json, length};
    std::free(json);// yyjson default allocator
    return R::success(std::move(out));
}

[[nodiscard]] Status write_text(const luisa::string &text, const luisa::filesystem::path &path,
                                bool to_stderr) {
    if (path.empty()) {
        std::FILE *stream = to_stderr ? stderr : stdout;
        if (std::fputs(text.c_str(), stream) < 0) {
            return Status::error(StatusCode::io_error, "failed to write JSON output", "output");
        }
        return Status::success();
    }
    const auto path_text = luisa::to_string(path);
    std::FILE *file = std::fopen(path_text.c_str(), "wb");
    if (file == nullptr) {
        return Status::error(StatusCode::io_error,
                             luisa::string{"cannot open output file '"} + path_text + "'", "output");
    }
    const bool wrote = std::fwrite(text.data(), 1, text.size(), file) == text.size();
    const bool closed = std::fclose(file) == 0;
    if (!wrote || !closed) {
        return Status::error(StatusCode::io_error,
                             luisa::string{"failed to write output file '"} + path_text + "'", "output");
    }
    return Status::success();
}

}// namespace

[[nodiscard]] Result<luisa::string> serialize_result(const RecognizeResult &result,
                                                     const JsonOutputOptions &options) {
    using R = Result<luisa::string>;
    MutDoc holder;
    if (holder.doc == nullptr) {
        return R::failure(StatusCode::out_of_memory, "cannot allocate JSON document", "json");
    }
    auto *doc = holder.doc;
    auto *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_bool(doc, root, "ok", true);
    add_str(doc, root, "text", result.text);

    auto *lines = yyjson_mut_arr(doc);
    for (const auto &line : result.lines) {
        auto *l = yyjson_mut_obj(doc);
        add_str(doc, l, "text", line.text);
        if (!add_source_region(doc, l, "source", line.source)) {
            return R::failure(StatusCode::invalid_argument, "non-finite line coordinate", "lines");
        }
        yyjson_mut_obj_add_uint(doc, l, "reading_order", line.reading_order);
        yyjson_mut_obj_add_bool(doc, l, "has_confidence", line.has_confidence);
        if (line.has_confidence && !add_finite_real(doc, l, "confidence", line.confidence)) {
            return R::failure(StatusCode::invalid_argument, "non-finite line confidence", "lines");
        }
        yyjson_mut_arr_append(lines, l);
    }
    yyjson_mut_obj_add_val(doc, root, "lines", lines);

    auto *blocks = yyjson_mut_arr(doc);
    for (const auto &block : result.blocks) {
        auto *b = yyjson_mut_obj(doc);
        add_str(doc, b, "type", block_type_name(block.type));
        add_str(doc, b, "content", block.content);
        if (!add_source_region(doc, b, "source", block.source)) {
            return R::failure(StatusCode::invalid_argument, "non-finite block coordinate", "blocks");
        }
        yyjson_mut_obj_add_uint(doc, b, "reading_order", block.reading_order);
        yyjson_mut_obj_add_bool(doc, b, "has_confidence", block.has_confidence);
        if (block.has_confidence && !add_finite_real(doc, b, "confidence", block.confidence)) {
            return R::failure(StatusCode::invalid_argument, "non-finite block confidence", "blocks");
        }
        yyjson_mut_arr_append(blocks, b);
    }
    yyjson_mut_obj_add_val(doc, root, "blocks", blocks);

    if (!add_metadata(doc, root, result.metadata)) {
        return R::failure(StatusCode::invalid_argument, "non-finite metadata value", "metadata");
    }

    auto *diagnostics = yyjson_mut_arr(doc);
    for (const auto &diag : result.diagnostics) {
        auto *d = yyjson_mut_obj(doc);
        add_str(doc, d, "severity", diagnostic_severity_name(diag.severity));
        add_str(doc, d, "code", diag.code);
        add_str(doc, d, "message", diag.message);
        yyjson_mut_arr_append(diagnostics, d);
    }
    yyjson_mut_obj_add_val(doc, root, "diagnostics", diagnostics);

    if (!add_counters(doc, root, result.counters)) {
        return R::failure(StatusCode::invalid_argument, "non-finite performance counter", "counters");
    }
    return write_doc(doc, options);
}

[[nodiscard]] Result<luisa::string> serialize_status(const Status &status,
                                                     const JsonOutputOptions &options) {
    using R = Result<luisa::string>;
    MutDoc holder;
    if (holder.doc == nullptr) {
        return R::failure(StatusCode::out_of_memory, "cannot allocate JSON document", "json");
    }
    auto *doc = holder.doc;
    auto *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_bool(doc, root, "ok", status.ok());
    add_str(doc, root, "code", status_code_name(status.code));
    add_str(doc, root, "message", status.message);
    add_str(doc, root, "detail", status.detail);
    return write_doc(doc, options);
}

[[nodiscard]] Status write_result(const RecognizeResult &result,
                                  const luisa::filesystem::path &path,
                                  const JsonOutputOptions &options) {
    auto json = serialize_result(result, options);
    if (!json.ok()) { return json.status; }
    auto out = *json.value;
    out.push_back('\n');
    return write_text(out, path, false);
}

[[nodiscard]] Status write_status(const Status &status,
                                  const luisa::filesystem::path &path,
                                  const JsonOutputOptions &options) {
    auto json = serialize_status(status, options);
    if (!json.ok()) { return json.status; }
    auto out = *json.value;
    out.push_back('\n');
    return write_text(out, path, true);
}

}// namespace luisa::ocr::output