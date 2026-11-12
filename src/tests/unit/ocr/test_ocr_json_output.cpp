// Test for examples/ocr/ocr/output/json_output.h/.cpp -- full result schema.
// Host-only: no device, no backend, no model, no exceptions.
//
// This test covers:
// - the FULL RecognizeResult schema is serialized (text, lines/blocks with
//   coordinates/page/order/content/confidence presence, metadata/policies/
//   router, diagnostics, counters);
// - strict status/error JSON;
// - non-finite floats are rejected (never invalid JSON);
// - dynamic strings with quotes/control chars are escaped correctly;
// - the output parses back as valid JSON (yyjson round trip).
#include "ut/ut.hpp"

#include <limits>

#include <yyjson.h>

#include "ocr/output/json_output.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] RecognizeResult full_result() {
    RecognizeResult result;
    result.text = "hello \"world\"\nline2";
    OcrLine line;
    line.text = "line text";
    line.source.page_index = 1;
    line.source.box = SourceBox{1.0f, 2.0f, 3.0f, 4.0f};
    line.source.polygon.emplace_back(SourcePoint{1.0f, 2.0f});
    line.reading_order = 0;
    line.has_confidence = true;
    line.confidence = 0.9f;
    result.lines.emplace_back(line);
    OcrBlock block;
    block.type = BlockType::table;
    block.content = "| a | b |";
    block.source.page_index = 1;
    block.source.box = SourceBox{0.0f, 0.0f, 10.0f, 10.0f};
    block.reading_order = 1;
    block.has_confidence = false;
    result.blocks.emplace_back(block);
    result.metadata.model_id = ModelId::glm_ocr;
    result.metadata.target_name = "glm_ocr";
    result.metadata.backend = "vk";
    result.metadata.weight_precision = Precision::fp32;
    result.metadata.checkpoint_revision = "rev";
    result.metadata.package_hash = "hash";
    result.metadata.recipe_version = "1";
    result.metadata.enabled_optimizations.emplace_back("none");
    result.metadata.router.used = false;
    Diagnostic diag;
    diag.severity = DiagnosticSeverity::warning;
    diag.code = "output_truncated";
    diag.message = "truncated";
    result.diagnostics.emplace_back(diag);
    result.counters.total_ms = 12.5;
    result.counters.pages_processed = 1;
    result.counters.output_tokens = 7;
    return result;
}

[[nodiscard]] bool is_valid_json(const luisa::string &text) {
    yyjson_doc *doc = yyjson_read(text.data(), text.size(), 0);
    if (doc == nullptr) { return false; }
    yyjson_doc_free(doc);
    return true;
}

void test_full_schema() {
    auto result = full_result();
    auto json = output::serialize_result(result);
    expect(json.ok());
    if (!json.ok()) { return; }
    expect(is_valid_json(*json.value));
    // Every top-level field is present.
    for (auto key : {"\"text\"", "\"lines\"", "\"blocks\"", "\"metadata\"",
                     "\"diagnostics\"", "\"counters\"", "\"ok\""}) {
        expect(json.value->find(key) != luisa::string::npos) << "missing " << key;
    }
    // Line/block detail fields.
    for (auto key : {"\"reading_order\"", "\"has_confidence\"", "\"confidence\"",
                     "\"page_index\"", "\"polygon\"", "\"box\""}) {
        expect(json.value->find(key) != luisa::string::npos) << "missing " << key;
    }
    // Metadata/policy/router fields.
    for (auto key : {"\"model_id\"", "\"weight_precision\"", "\"cache_policy\"",
                     "\"offload_policy\"", "\"graph_policy\"", "\"fast_path_policy\"",
                     "\"router\"", "\"support_state\"", "\"validated\"",
                     "\"enabled_optimizations\""}) {
        expect(json.value->find(key) != luisa::string::npos) << "missing " << key;
    }
    // Counter fields.
    for (auto key : {"\"total_ms\"", "\"pages_processed\"", "\"output_tokens\"",
                     "\"peak_device_bytes\"", "\"resident_weight_bytes\""}) {
        expect(json.value->find(key) != luisa::string::npos) << "missing " << key;
    }
}

void test_escaping() {
    RecognizeResult result;
    result.text = "quote\" backslash\\ newline\n tab\t control\x01";
    auto json = output::serialize_result(result);
    expect(json.ok());
    if (json.ok()) {
        expect(is_valid_json(*json.value));
        expect(json.value->find("\\\"") != luisa::string::npos);
        expect(json.value->find("\\n") != luisa::string::npos);
        expect(json.value->find("\\u0001") != luisa::string::npos);
    }
}

void test_nonfinite_rejected() {
    auto result = full_result();
    result.lines[0].confidence = std::numeric_limits<float>::quiet_NaN();
    auto json = output::serialize_result(result);
    expect(!json.ok());// never emit invalid JSON
    auto result2 = full_result();
    result2.counters.total_ms = std::numeric_limits<double>::infinity();
    expect(!output::serialize_result(result2).ok());
    auto result3 = full_result();
    result3.lines[0].source.box.x0 = std::numeric_limits<float>::infinity();
    expect(!output::serialize_result(result3).ok());
}

void test_status_json() {
    auto status = Status::error(StatusCode::unvalidated, "no \"model\"", "detail");
    auto json = output::serialize_status(status);
    expect(json.ok());
    if (json.ok()) {
        expect(is_valid_json(*json.value));
        expect(json.value->find("\"ok\": false") != luisa::string::npos ||
               json.value->find("\"ok\":false") != luisa::string::npos);
        expect(json.value->find("unvalidated") != luisa::string::npos);
    }
    auto ok = output::serialize_status(Status::success());
    expect(ok.ok());
    if (ok.ok()) { expect(is_valid_json(*ok.value)); }
}

void test_empty_result() {
    RecognizeResult result;// all defaults
    auto json = output::serialize_result(result);
    expect(json.ok());
    if (json.ok()) {
        expect(is_valid_json(*json.value));
        expect(json.value->find("\"lines\"") != luisa::string::npos);
        expect(json.value->find("\"blocks\"") != luisa::string::npos);
    }
}

}// namespace

static auto test_ocr_json_output_registration = [] {
    "full_schema"_test = [] { test_full_schema(); };
    "escaping"_test = [] { test_escaping(); };
    "nonfinite_rejected"_test = [] { test_nonfinite_rejected(); };
    "status_json"_test = [] { test_status_json(); };
    "empty_result"_test = [] { test_empty_result(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}