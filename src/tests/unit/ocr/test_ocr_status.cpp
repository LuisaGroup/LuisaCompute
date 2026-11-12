// Test for examples/ocr/ocr/status.h and the Phase 0 session/option argument
// validation (examples/ocr/ocr/session.cpp, api.h).
// Host-only: no device, no backend, no model, no exceptions.
//
// This test covers:
// - Status/Result success/failure semantics and stable code names;
// - RecognizeOptions structural validation (batch/token budgets, sampling,
//   policies, time budget) and the compatibility pixel-bound rule;
// - Session::validate_request page/schema rules;
// - SessionConfig validation (backend, package/manifest, model selector);
// - Phase 0 fail-closed behavior: create() refuses with `unvalidated` AFTER
//   argument validation, recognize() never fabricates output, cancellation
//   is honored before the refusal.
#include "ut/ut.hpp"

#include <cstdint>
#include <limits>

#include "ocr/api.h"
#include "ocr/session.h"
#include "ocr/status.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] SessionConfig valid_config() {
    SessionConfig config;
    config.backend = "vk";
    config.manifest = luisa::filesystem::path{"model.lcocr.json"};
    return config;
}

[[nodiscard]] luisa::vector<PageInput> one_page() {
    luisa::vector<PageInput> pages;
    PageInput page;
    page.path = luisa::filesystem::path{"page0.png"};
    pages.emplace_back(std::move(page));
    return pages;
}

void test_status_basics() {
    expect(Status::success().ok());
    expect(Status{}.ok());
    const auto error = Status::error(StatusCode::invalid_argument, "bad", "field");
    expect(!error.ok());
    expect(error.code == StatusCode::invalid_argument);
    expect(error.message == "bad");
    expect(error.detail == "field");
    expect(status_code_name(StatusCode::ok) == "ok");
    expect(status_code_name(StatusCode::invalid_argument) == "invalid_argument");
    expect(status_code_name(StatusCode::unsupported) == "unsupported");
    expect(status_code_name(StatusCode::unvalidated) == "unvalidated");
    expect(status_code_name(StatusCode::not_found) == "not_found");
    expect(status_code_name(StatusCode::io_error) == "io_error");
    expect(status_code_name(StatusCode::out_of_memory) == "out_of_memory");
    expect(status_code_name(StatusCode::cancelled) == "cancelled");
    expect(status_code_name(StatusCode::timeout) == "timeout");
    expect(status_code_name(StatusCode::internal_error) == "internal_error");
    expect(status_code_name(static_cast<StatusCode>(999u)) == "unknown");
    expect(diagnostic_severity_name(DiagnosticSeverity::info) == "info");
    expect(diagnostic_severity_name(DiagnosticSeverity::warning) == "warning");
    expect(diagnostic_severity_name(DiagnosticSeverity::error) == "error");
}

void test_result_basics() {
    const auto ok = Result<uint32_t>::success(7u);
    expect(ok.ok());
    expect(ok.value.has_value());
    expect(*ok.value == 7u);
    const auto bad = Result<uint32_t>::failure(StatusCode::unsupported, "nope");
    expect(!bad.ok());
    expect(!bad.value.has_value());
    expect(bad.status.code == StatusCode::unsupported);
    const auto bad2 = Result<uint32_t>::failure(Status::error(StatusCode::not_found, "gone"));
    expect(!bad2.ok());
    expect(bad2.status.code == StatusCode::not_found);
}

void test_option_defaults_valid() {
    RecognizeOptions options;
    expect(validate_recognize_options(options).ok());
    expect(options.mode == RecognizeMode::plain);
    expect(options.sampling.greedy);// greedy default
    // Precision policy defaults to the `automatic` sentinel (recipe default,
    // else FP32); it is NOT a concrete storage format.
    expect(options.precision.weights == Precision::automatic);
    expect(options.precision.activations == Precision::automatic);
    expect(options.precision.cache == Precision::automatic);
    expect(options.offload_policy == OffloadPolicy::disabled);
    expect(options.graph_policy == GraphPolicy::disabled);
    expect(options.fast_path_policy == FastPathPolicy::disabled);
}

void test_option_budget_validation() {
    {
        RecognizeOptions options;
        options.max_patches_per_vision_batch = 0;
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.max_regions_per_batch = 0;
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.max_decode_batch = 0;
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.max_output_tokens_per_region = 0;
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.max_output_tokens_per_page = 0;
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.max_output_tokens_per_region = 4096;
        options.max_output_tokens_per_page = 2048;
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.max_wall_time_ms = -1.0;
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.max_wall_time_ms = std::numeric_limits<double>::infinity();
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.max_wall_time_ms = std::numeric_limits<double>::quiet_NaN();
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
}

void test_sampling_validation() {
    {
        RecognizeOptions options;
        options.sampling.greedy = false;
        options.sampling.temperature = 0.0f;// non-greedy needs > 0
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.sampling.greedy = false;
        options.sampling.temperature = 0.7f;
        options.sampling.top_p = 0.0f;
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.sampling.greedy = false;
        options.sampling.temperature = 0.7f;
        options.sampling.top_p = 1.5f;
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.sampling.greedy = false;
        options.sampling.temperature = 0.7f;
        options.sampling.top_p = 0.9f;
        options.sampling.top_k = 40;
        expect(validate_recognize_options(options).ok());
    }
    {
        // Greedy ignores temperature/top_p entirely.
        RecognizeOptions options;
        options.sampling.greedy = true;
        options.sampling.temperature = 0.0f;
        options.sampling.top_p = 0.0f;
        expect(validate_recognize_options(options).ok());
    }
    {
        // Non-greedy NaN/inf temperature is rejected.
        RecognizeOptions options;
        options.sampling.greedy = false;
        options.sampling.temperature = std::numeric_limits<float>::quiet_NaN();
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.sampling.greedy = false;
        options.sampling.temperature = std::numeric_limits<float>::infinity();
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.sampling.greedy = false;
        options.sampling.temperature = -0.5f;
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
    {
        // Non-greedy NaN/inf top_p is rejected.
        RecognizeOptions options;
        options.sampling.greedy = false;
        options.sampling.temperature = 0.7f;
        options.sampling.top_p = std::numeric_limits<float>::quiet_NaN();
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.sampling.greedy = false;
        options.sampling.temperature = 0.7f;
        options.sampling.top_p = std::numeric_limits<float>::infinity();
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
}

void test_policy_validation() {
    {
        RecognizeOptions options;
        options.precision.weights = static_cast<Precision>(99u);
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.cache_policy = static_cast<CachePolicy>(99u);
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.offload_policy = static_cast<OffloadPolicy>(99u);
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.graph_policy = static_cast<GraphPolicy>(99u);
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.fast_path_policy = static_cast<FastPathPolicy>(99u);
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.mode = static_cast<RecognizeMode>(99u);
        expect(validate_recognize_options(options).code == StatusCode::invalid_argument);
    }
}

void test_pixel_bounds() {
    {
        RecognizeOptions options;
        const auto bounds = effective_pixel_bounds(options);
        expect(bounds.overview == 0u);
        expect(bounds.crop == 0u);
    }
    {
        RecognizeOptions options;
        options.max_overview_pixels = 1000;
        options.max_crop_pixels = 2000;
        const auto bounds = effective_pixel_bounds(options);
        expect(bounds.overview == 1000u);
        expect(bounds.crop == 2000u);
    }
    {
        // Compatibility bound caps both (min wins) and fills unset bounds.
        RecognizeOptions options;
        options.max_overview_pixels = 1000;
        options.max_crop_pixels = 2000;
        options.max_pixels = 1500;
        const auto bounds = effective_pixel_bounds(options);
        expect(bounds.overview == 1000u);
        expect(bounds.crop == 1500u);
    }
    {
        RecognizeOptions options;
        options.max_pixels = 1500;
        const auto bounds = effective_pixel_bounds(options);
        expect(bounds.overview == 1500u);
        expect(bounds.crop == 1500u);
    }
}

void test_request_validation() {
    const auto pages = one_page();
    {
        RecognizeOptions options;
        expect(Session::validate_request(pages, options).ok());
    }
    {
        RecognizeOptions options;
        expect(Session::validate_request({}, options).code == StatusCode::invalid_argument);
    }
    {
        luisa::vector<PageInput> bad_pages;
        bad_pages.emplace_back(PageInput{});// empty path
        RecognizeOptions options;
        expect(Session::validate_request(bad_pages, options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.mode = RecognizeMode::extract;// no schema/prompt
        expect(Session::validate_request(pages, options).code == StatusCode::invalid_argument);
    }
    {
        RecognizeOptions options;
        options.mode = RecognizeMode::extract;
        options.schema = "{\"type\":\"object\"}";
        expect(Session::validate_request(pages, options).ok());
    }
    {
        RecognizeOptions options;
        options.mode = RecognizeMode::extract;
        options.prompt = "extract the invoice total";
        expect(Session::validate_request(pages, options).ok());
    }
    {
        RecognizeOptions options;
        options.max_decode_batch = 0;
        expect(Session::validate_request(pages, options).code == StatusCode::invalid_argument);
    }
}

void test_session_config_validation() {
    expect(validate_session_config(valid_config()).ok());
    {
        auto config = valid_config();
        config.backend.clear();
        expect(validate_session_config(config).code == StatusCode::invalid_argument);
    }
    {
        auto config = valid_config();
        config.backend = "CUDA";// uppercase rejected: plugin names are lowercase
        expect(validate_session_config(config).code == StatusCode::invalid_argument);
    }
    {
        auto config = valid_config();
        config.package.clear();
        config.manifest.clear();
        expect(validate_session_config(config).code == StatusCode::invalid_argument);
    }
    {
        auto config = valid_config();
        config.model = "not_a_model";
        expect(validate_session_config(config).code == StatusCode::invalid_argument);
    }
    {
        auto config = valid_config();
        config.model = "pp_ocrv6";// bare family without size rejected
        expect(validate_session_config(config).code == StatusCode::invalid_argument);
    }
    {
        auto config = valid_config();
        config.model = "pp_ocrv6:medium";
        expect(validate_session_config(config).ok());
    }
    {
        auto config = valid_config();
        config.precision.cache = static_cast<Precision>(99u);
        expect(validate_session_config(config).code == StatusCode::invalid_argument);
    }
}

void test_session_fail_closed() {
    // Valid arguments -> unvalidated refusal (never invalid_argument, never
    // a fabricated session).
    const auto session = Session::create(valid_config());
    expect(!session.ok());
    expect(session.status.code == StatusCode::unvalidated);
    expect(!session.value.has_value());
    // Invalid arguments are reported BEFORE the unvalidated refusal.
    auto config = valid_config();
    config.backend.clear();
    const auto bad = Session::create(config);
    expect(!bad.ok());
    expect(bad.status.code == StatusCode::invalid_argument);
    // The unsafe override does not make Phase 0 creation succeed.
    auto unsafe_config = valid_config();
    unsafe_config.registry_policy.allow_unsafe_experimental = true;
    const auto unsafe_session = Session::create(unsafe_config);
    expect(!unsafe_session.ok());
    expect(unsafe_session.status.code == StatusCode::unvalidated);
}

void test_result_schema_defaults() {
    // The result schema must default to "no confidence" and planned support.
    RecognizeResult result;
    expect(result.text.empty());
    expect(result.lines.empty());
    expect(result.blocks.empty());
    expect(result.metadata.support_state == SupportState::planned);
    expect(!result.metadata.validated);
    // Metadata precision fields are concrete (never the automatic sentinel).
    expect(result.metadata.weight_precision == Precision::fp32);
    expect(result.metadata.activation_precision == Precision::fp32);
    expect(result.metadata.cache_precision == Precision::fp32);
    // Default policy state and no implicit routing/fallback.
    expect(result.metadata.cache_policy == CachePolicy::automatic);
    expect(result.metadata.offload_policy == OffloadPolicy::disabled);
    expect(result.metadata.graph_policy == GraphPolicy::disabled);
    expect(result.metadata.fast_path_policy == FastPathPolicy::disabled);
    expect(!result.metadata.router.used);
    expect(result.metadata.router.initial_model == ModelId::unknown);
    expect(result.metadata.router.final_model == ModelId::unknown);
    OcrLine line;
    expect(!line.has_confidence);
    OcrBlock block;
    expect(!block.has_confidence);
    expect(block.type == BlockType::text);
    expect(block_type_name(BlockType::table) == "table");
    expect(block_type_name(BlockType::formula) == "formula");
    SourceBox box;
    expect(box.empty());
    box.x1 = 10.0f;
    box.y1 = 5.0f;
    expect(!box.empty());
}

void test_preflight_ordering() {
    const auto pages = one_page();
    // Valid request, no cancellation -> ok.
    {
        RecognizeOptions options;
        expect(Session::preflight_request(pages, options).ok());
    }
    // Valid request + cancellation -> cancelled.
    {
        RecognizeOptions options;
        options.cancel_callback = [] { return true; };
        expect(Session::preflight_request(pages, options).code == StatusCode::cancelled);
    }
    // Invalid request + cancellation -> invalid_argument wins (validation
    // precedes cancellation).
    {
        RecognizeOptions options;
        options.max_decode_batch = 0;// invalid
        options.cancel_callback = [] { return true; };
        expect(Session::preflight_request(pages, options).code == StatusCode::invalid_argument);
    }
    // Empty pages + cancellation -> invalid_argument wins.
    {
        RecognizeOptions options;
        options.cancel_callback = [] { return true; };
        expect(Session::preflight_request({}, options).code == StatusCode::invalid_argument);
    }
    // Cancellation callback returning false does not cancel.
    {
        RecognizeOptions options;
        options.cancel_callback = [] { return false; };
        expect(Session::preflight_request(pages, options).ok());
    }
    // recognize() uses the same preflight: cancelled before the refusal.
    // (Session cannot be constructed in Phase 0, so we assert the preflight
    // contract directly; recognize() delegates to preflight_request.)
}

void test_precision_resolution() {
    // Concrete requests pass through unchanged.
    expect(resolve_precision(Precision::fp16, Precision::fp32) == Precision::fp16);
    expect(resolve_precision(Precision::bf16, Precision::fp32) == Precision::bf16);
    expect(resolve_precision(Precision::int4, Precision::fp16) == Precision::int4);
    // automatic resolves to the recipe default.
    expect(resolve_precision(Precision::automatic, Precision::bf16) == Precision::bf16);
    expect(resolve_precision(Precision::automatic, Precision::fp32) == Precision::fp32);
    // automatic with an automatic/unknown default falls back to FP32.
    expect(resolve_precision(Precision::automatic, Precision::automatic) == Precision::fp32);
    expect(resolve_precision(Precision::automatic, static_cast<Precision>(99u)) == Precision::fp32);
    // The automatic sentinel is a valid policy value, not an invalid enum.
    RecognizeOptions options;
    options.precision.weights = Precision::automatic;
    expect(validate_recognize_options(options).ok());
    expect(precision_name(Precision::automatic) == "automatic");
    expect(precision_from_name("automatic").has_value());
    expect(*precision_from_name("automatic") == Precision::automatic);
}

}// namespace

static auto test_ocr_status_registration = [] {
    "status_basics"_test = [] { test_status_basics(); };
    "result_basics"_test = [] { test_result_basics(); };
    "option_defaults_valid"_test = [] { test_option_defaults_valid(); };
    "option_budget_validation"_test = [] { test_option_budget_validation(); };
    "sampling_validation"_test = [] { test_sampling_validation(); };
    "policy_validation"_test = [] { test_policy_validation(); };
    "pixel_bounds"_test = [] { test_pixel_bounds(); };
    "request_validation"_test = [] { test_request_validation(); };
    "session_config_validation"_test = [] { test_session_config_validation(); };
    "session_fail_closed"_test = [] { test_session_fail_closed(); };
    "result_schema_defaults"_test = [] { test_result_schema_defaults(); };
    "preflight_ordering"_test = [] { test_preflight_ordering(); };
    "precision_resolution"_test = [] { test_precision_resolution(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
}
