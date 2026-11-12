// Test for examples/ocr/ocr/document.h/.cpp -- externally-rasterized page input.
// Host-only: no device, no backend, no model, no exceptions. NO in-process PDF.
//
// This test covers:
// - document manifest parsing (relative path resolution, page metadata);
// - page/pixel/dimension limits;
// - loading decoded pages in order;
// - fail-closed on empty/oversized/mismatched manifests;
// - the frozen synthetic document manifest (never regenerated here).
#include "ut/ut.hpp"

#include <filesystem>

#include <luisa/core/stl/filesystem.h>

#include "ocr/document.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] luisa::filesystem::path data_dir() {
    return luisa::filesystem::path{LUISA_OCR_TEST_DATA_DIR};
}

void test_frozen_manifest() {
    auto r = load_document_manifest(data_dir() / "document_manifest.json");
    expect(r.ok());
    if (!r.ok()) { return; }
    expect(r.value->pages.size() == 2u);
    expect(r.value->dpi == 72u);
    expect(r.value->pages[0].source_page_index == 0u);
    expect(r.value->pages[1].source_page_index == 1u);
    DocumentLimits limits;
    auto doc = load_document(*r.value, limits);
    expect(doc.ok());
    if (doc.ok()) {
        expect(doc.value->pages.size() == 2u);
        expect(doc.value->pages[0].width == 4u);
        expect(doc.value->pages[1].width == 8u);
        expect(doc.value->total_pixels == 4u * 4u + 8u * 8u);
    }
}

void test_parse_inline() {
    const char *json =
        "{\"format_version\":1,\"producer\":\"t\",\"source\":\"s\",\"rasterizer\":\"r\",\"dpi\":96,"
        "\"pages\":[{\"source_page_index\":0,\"image\":\"blank_4x4.png\",\"width\":4,\"height\":4}]}";
    auto r = parse_document_manifest(json, data_dir());
    expect(r.ok());
    if (r.ok()) {
        expect(r.value->pages.size() == 1u);
        expect(r.value->pages[0].declared_width == 4u);
        // Relative path resolved against the manifest dir.
        expect(r.value->pages[0].image_path.filename() == "blank_4x4.png");
    }
}

void test_limits() {
    auto r = load_document_manifest(data_dir() / "document_manifest.json");
    expect(r.ok());
    if (!r.ok()) { return; }
    // Page-count limit.
    DocumentLimits limits;
    limits.max_pages = 1;
    expect(!validate_document(*r.value, limits).ok());
    // Total-pixel limit.
    DocumentLimits limits2;
    limits2.max_total_pixels = 10;// 16+64=80 > 10
    expect(!validate_document(*r.value, limits2).ok());
    // Dimension limit.
    DocumentLimits limits3;
    limits3.image.max_width = 4;// second page is 8 wide
    expect(!validate_document(*r.value, limits3).ok());
}

void test_malformed() {
    // Empty pages.
    expect(!parse_document_manifest("{\"format_version\":1,\"pages\":[]}", data_dir()).ok());
    // Missing pages.
    expect(!parse_document_manifest("{\"format_version\":1}", data_dir()).ok());
    // Wrong version.
    expect(!parse_document_manifest("{\"format_version\":2,\"pages\":[]}", data_dir()).ok());
    // Page without image.
    expect(!parse_document_manifest(
        "{\"format_version\":1,\"pages\":[{\"source_page_index\":0}]}", data_dir()).ok());
    // Bad JSON.
    expect(!parse_document_manifest("{not json", data_dir()).ok());
}

void test_dimension_mismatch() {
    // Declared dimensions that do not match the actual image are rejected.
    const char *json =
        "{\"format_version\":1,\"pages\":[{\"image\":\"blank_4x4.png\",\"width\":99,\"height\":99}]}";
    auto r = parse_document_manifest(json, data_dir());
    expect(r.ok());
    if (r.ok()) {
        DocumentLimits limits;
        auto doc = load_document(*r.value, limits);
        expect(!doc.ok());
    }
}

// Oversized uint values (> UINT32_MAX) must be rejected, not silently narrowed.
void test_uint32_overflow_rejected() {
    const char *big_dpi =
        "{\"format_version\":1,\"dpi\":4294967296,"
        "\"pages\":[{\"image\":\"blank_4x4.png\"}]}";
    expect(!parse_document_manifest(big_dpi, data_dir()).ok());
    const char *big_index =
        "{\"format_version\":1,\"pages\":[{\"source_page_index\":4294967296,\"image\":\"blank_4x4.png\"}]}";
    expect(!parse_document_manifest(big_index, data_dir()).ok());
    const char *big_width =
        "{\"format_version\":1,\"pages\":[{\"image\":\"blank_4x4.png\",\"width\":4294967296}]}";
    expect(!parse_document_manifest(big_width, data_dir()).ok());
    const char *big_height =
        "{\"format_version\":1,\"pages\":[{\"image\":\"blank_4x4.png\",\"height\":4294967296}]}";
    expect(!parse_document_manifest(big_height, data_dir()).ok());
    // A valid uint32 max value is accepted (boundary).
    const char *max_width =
        "{\"format_version\":1,\"pages\":[{\"image\":\"blank_4x4.png\",\"width\":4294967295}]}";
    expect(parse_document_manifest(max_width, data_dir()).ok());
}

}// namespace

static auto test_ocr_document_registration = [] {
    "frozen_manifest"_test = [] { test_frozen_manifest(); };
    "parse_inline"_test = [] { test_parse_inline(); };
    "limits"_test = [] { test_limits(); };
    "malformed"_test = [] { test_malformed(); };
    "dimension_mismatch"_test = [] { test_dimension_mismatch(); };
    "uint32_overflow_rejected"_test = [] { test_uint32_overflow_rejected(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}