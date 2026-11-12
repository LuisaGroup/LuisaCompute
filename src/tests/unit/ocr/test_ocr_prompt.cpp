// Test for examples/ocr/ocr/prompt.h/.cpp -- recipe-ID-specific prompt builders.
// Host-only: no device, no backend, no model, no exceptions.
//
// This test covers:
// - known template IDs build exact control-token layouts;
// - unknown template IDs fail closed (no general Jinja fallback);
// - extract requires a schema; mode restrictions;
// - image placeholder repetition.
#include "ut/ut.hpp"

#include "ocr/prompt.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void test_known_templates() {
    const auto ids = known_prompt_template_ids();
    expect(!ids.empty());
    for (auto id : ids) {
        PromptRequest req;
        req.template_id = luisa::string{id};
        req.mode = RecognizeMode::plain;
        req.num_images = 1;
        req.image_token = "<image>";
        auto r = build_prompt(req);
        expect(r.ok()) << "template " << id;
    }
}

void test_qwen3_layout() {
    PromptRequest req;
    req.template_id = "qwen3_vl_ocr";
    req.mode = RecognizeMode::plain;
    req.num_images = 1;
    req.image_token = "<image>";
    auto r = build_prompt(req);
    expect(r.ok());
    if (r.ok()) {
        expect(r.value->find("<|im_start|>system") != luisa::string::npos);
        expect(r.value->find("<|im_start|>user") != luisa::string::npos);
        expect(r.value->find("<|im_start|>assistant") != luisa::string::npos);
        expect(r.value->find("<image>") != luisa::string::npos);
    }
}

void test_unknown_template_fails_closed() {
    PromptRequest req;
    req.template_id = "some_unknown_jinja_template";
    req.mode = RecognizeMode::plain;
    auto r = build_prompt(req);
    expect(!r.ok());
    expect(r.status.code == StatusCode::unsupported);
}

void test_extract_requires_schema() {
    PromptRequest req;
    req.template_id = "qwen3_vl_ocr";
    req.mode = RecognizeMode::extract;
    auto r = build_prompt(req);
    expect(!r.ok());
    expect(r.status.code == StatusCode::invalid_argument);
    req.schema = "{\"type\":\"object\"}";
    auto r2 = build_prompt(req);
    expect(r2.ok());
    expect(r2.value->find("{\"type\":\"object\"}") != luisa::string::npos);
}

void test_pp_ocrv6_plain_only() {
    PromptRequest req;
    req.template_id = "pp_ocrv6_plain";
    req.mode = RecognizeMode::plain;
    auto r = build_prompt(req);
    expect(r.ok());
    expect(r.value->empty());// CNN OCR has no chat prompt
    req.mode = RecognizeMode::markdown;
    expect(!build_prompt(req).ok());
}

void test_image_repetition() {
    PromptRequest req;
    req.template_id = "generic_ocr";
    req.mode = RecognizeMode::plain;
    req.num_images = 3;
    req.image_token = "<img>";
    auto r = build_prompt(req);
    expect(r.ok());
    if (r.ok()) {
        size_t count = 0;
        size_t pos = 0;
        while ((pos = r.value->find("<img>", pos)) != luisa::string::npos) {
            count++;
            pos += 5;
        }
        expect(count == 3u);
    }
}

}// namespace

static auto test_ocr_prompt_registration = [] {
    "known_templates"_test = [] { test_known_templates(); };
    "qwen3_layout"_test = [] { test_qwen3_layout(); };
    "unknown_template_fails_closed"_test = [] { test_unknown_template_fails_closed(); };
    "extract_requires_schema"_test = [] { test_extract_requires_schema(); };
    "pp_ocrv6_plain_only"_test = [] { test_pp_ocrv6_plain_only(); };
    "image_repetition"_test = [] { test_image_repetition(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}