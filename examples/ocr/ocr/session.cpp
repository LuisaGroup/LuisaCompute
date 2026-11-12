// OCR engine -- session implementation: argument validation plus fail-closed
// refusal (Phase 0 contract, unchanged in Phase 1). No device work, no package
// reads, no inference, and no mock output. Phase 1 adds host artifact readers
// (ocr/package.* etc.) but the session does not consume them until a recipe
// passes its validation gates in a later phase.
#include "ocr/session.h"

#include <cmath>
#include <utility>

namespace luisa::ocr {

[[nodiscard]] luisa::string_view block_type_name(BlockType type) noexcept {
    switch (type) {
        case BlockType::text: return "text";
        case BlockType::title: return "title";
        case BlockType::table: return "table";
        case BlockType::formula: return "formula";
        case BlockType::figure: return "figure";
        case BlockType::header: return "header";
        case BlockType::footer: return "footer";
        case BlockType::list: return "list";
        case BlockType::other: return "other";
    }
    return "unknown";
}

namespace {

[[nodiscard]] bool valid_precision(Precision precision) noexcept {
    return precision_name(precision) != "unknown";
}

}// namespace

[[nodiscard]] Status validate_recognize_options(const RecognizeOptions &options) {
    if (recognize_mode_name(options.mode) == "unknown") {
        return Status::error(StatusCode::invalid_argument, "unknown recognize mode", "mode");
    }
    if (options.max_patches_per_vision_batch == 0) {
        return Status::error(StatusCode::invalid_argument, "max_patches_per_vision_batch must be > 0", "max_patches_per_vision_batch");
    }
    if (options.max_regions_per_batch == 0) {
        return Status::error(StatusCode::invalid_argument, "max_regions_per_batch must be > 0", "max_regions_per_batch");
    }
    if (options.max_decode_batch == 0) {
        return Status::error(StatusCode::invalid_argument, "max_decode_batch must be > 0", "max_decode_batch");
    }
    if (options.max_output_tokens_per_region == 0) {
        return Status::error(StatusCode::invalid_argument, "max_output_tokens_per_region must be > 0", "max_output_tokens_per_region");
    }
    if (options.max_output_tokens_per_page == 0) {
        return Status::error(StatusCode::invalid_argument, "max_output_tokens_per_page must be > 0", "max_output_tokens_per_page");
    }
    if (options.max_output_tokens_per_page < options.max_output_tokens_per_region) {
        return Status::error(StatusCode::invalid_argument,
                             "max_output_tokens_per_page must be >= max_output_tokens_per_region",
                             "max_output_tokens_per_page");
    }
    if (!options.sampling.greedy) {
        if (!(options.sampling.temperature > 0.0f) || !std::isfinite(options.sampling.temperature)) {
            return Status::error(StatusCode::invalid_argument,
                                 "non-greedy sampling requires a finite temperature > 0", "sampling.temperature");
        }
        if (!(options.sampling.top_p > 0.0f) || options.sampling.top_p > 1.0f || !std::isfinite(options.sampling.top_p)) {
            return Status::error(StatusCode::invalid_argument,
                                 "sampling.top_p must be in (0, 1]", "sampling.top_p");
        }
    }
    if (!valid_precision(options.precision.weights) ||
        !valid_precision(options.precision.activations) ||
        !valid_precision(options.precision.cache)) {
        return Status::error(StatusCode::invalid_argument, "unknown precision in policy", "precision");
    }
    if (options.cache_policy != CachePolicy::automatic &&
        options.cache_policy != CachePolicy::contiguous &&
        options.cache_policy != CachePolicy::paged) {
        return Status::error(StatusCode::invalid_argument, "unknown cache policy", "cache_policy");
    }
    if (options.offload_policy != OffloadPolicy::disabled &&
        options.offload_policy != OffloadPolicy::explicit_host) {
        return Status::error(StatusCode::invalid_argument, "unknown offload policy", "offload_policy");
    }
    if (options.graph_policy != GraphPolicy::disabled &&
        options.graph_policy != GraphPolicy::automatic &&
        options.graph_policy != GraphPolicy::enabled) {
        return Status::error(StatusCode::invalid_argument, "unknown graph policy", "graph_policy");
    }
    if (options.fast_path_policy != FastPathPolicy::disabled &&
        options.fast_path_policy != FastPathPolicy::automatic &&
        options.fast_path_policy != FastPathPolicy::enabled) {
        return Status::error(StatusCode::invalid_argument, "unknown fast-path policy", "fast_path_policy");
    }
    if (options.max_wall_time_ms < 0.0 || !std::isfinite(options.max_wall_time_ms)) {
        return Status::error(StatusCode::invalid_argument,
                             "max_wall_time_ms must be finite and >= 0", "max_wall_time_ms");
    }
    return Status::success();
}

[[nodiscard]] EffectivePixelBounds effective_pixel_bounds(const RecognizeOptions &options) noexcept {
    EffectivePixelBounds bounds{options.max_overview_pixels, options.max_crop_pixels};
    if (options.max_pixels != 0) {
        bounds.overview = bounds.overview == 0 ? options.max_pixels
                                               : (bounds.overview < options.max_pixels ? bounds.overview : options.max_pixels);
        bounds.crop = bounds.crop == 0 ? options.max_pixels
                                       : (bounds.crop < options.max_pixels ? bounds.crop : options.max_pixels);
    }
    return bounds;
}

[[nodiscard]] Status validate_session_config(const SessionConfig &config) {
    if (config.backend.empty()) {
        return Status::error(StatusCode::invalid_argument, "backend must be set", "backend");
    }
    for (auto c : config.backend) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
        if (!ok) {
            return Status::error(StatusCode::invalid_argument,
                                 "backend must be a lowercase plugin name (e.g. cuda, vk, dx, fallback)", "backend");
        }
    }
    if (config.package.empty() && config.manifest.empty()) {
        return Status::error(StatusCode::invalid_argument,
                             "a package or manifest path is required", "package");
    }
    if (!config.model.empty()) {
        const auto target = parse_model_target(config.model);
        if (!target.ok()) { return target.status; }
    }
    if (!valid_precision(config.precision.weights) ||
        !valid_precision(config.precision.activations) ||
        !valid_precision(config.precision.cache)) {
        return Status::error(StatusCode::invalid_argument, "unknown precision in session policy", "precision");
    }
    return Status::success();
}

[[nodiscard]] Status Session::validate_request(
    luisa::span<const PageInput> pages, const RecognizeOptions &options) {
    if (pages.empty()) {
        return Status::error(StatusCode::invalid_argument, "at least one input page is required", "pages");
    }
    for (size_t i = 0; i < pages.size(); i++) {
        if (pages[i].path.empty()) {
            return Status::error(StatusCode::invalid_argument,
                                 "input page path must not be empty", "pages");
        }
    }
    auto status = validate_recognize_options(options);
    if (!status.ok()) { return status; }
    if (options.mode == RecognizeMode::extract && options.schema.empty() && options.prompt.empty()) {
        return Status::error(StatusCode::invalid_argument,
                             "extract mode requires a schema or prompt", "schema");
    }
    return Status::success();
}

[[nodiscard]] Status Session::preflight_request(
    luisa::span<const PageInput> pages, const RecognizeOptions &options) {
    // Validation precedes cancellation: bad input is reported as
    // invalid_argument even if the caller would also cancel.
    auto status = validate_request(pages, options);
    if (!status.ok()) { return status; }
    if (options.cancel_callback && options.cancel_callback()) {
        return Status::error(StatusCode::cancelled, "request cancelled before start", "cancel");
    }
    return Status::success();
}

Session::Session() noexcept = default;
Session::~Session() noexcept = default;

[[nodiscard]] Result<luisa::unique_ptr<Session>> Session::create(const SessionConfig &config) {
    auto status = validate_session_config(config);
    if (!status.ok()) {
        return Result<luisa::unique_ptr<Session>>::failure(std::move(status));
    }
    // Fail closed: no recipe is validated, so no loader path is enabled. Even
    // the unsafe experimental override cannot make a model runnable; it only
    // relaxes registry matching, never inference.
    return Result<luisa::unique_ptr<Session>>::failure(
        StatusCode::unvalidated,
        "no OCR model is validated yet: session creation is refused; see examples/ocr/MODELS.md",
        "model");
}

[[nodiscard]] Result<RecognizeResult> Session::recognize(
    luisa::span<const PageInput> pages, const RecognizeOptions &options) {
    auto status = preflight_request(pages, options);
    if (!status.ok()) {
        return Result<RecognizeResult>::failure(std::move(status));
    }
    // Fail closed: no validated model exists, so no inference is attempted
    // and no output is fabricated.
    return Result<RecognizeResult>::failure(
        StatusCode::unvalidated,
        "no OCR model is validated yet: recognition is refused; see examples/ocr/MODELS.md",
        "model");
}

}// namespace luisa::ocr
