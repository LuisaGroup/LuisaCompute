#pragma once
// OCR engine -- document input (Phase 1). Host-only.
//
// A Document is an ordered set of externally-rasterized page images plus an
// optional manifest. There is NO in-process PDF parser: PDF pages must be
// rasterized outside the inference process (tools/rasterize_pdf.py) and handed
// in as images. The document loader enforces page/pixel/file limits and reads
// a checked JSON manifest describing the pages.
#include <cstdint>

#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/image.h"
#include "ocr/status.h"

namespace luisa::ocr {

struct DocumentLimits {
    uint32_t max_pages{256};
    uint64_t max_total_pixels{512ull * 1024ull * 1024ull};
    ImageLimits image;
};

// One page description from a document manifest.
struct DocumentPage {
    uint32_t source_page_index{0};
    luisa::filesystem::path image_path;
    // Optional declared dimensions (0 = unknown; verified against the decode).
    uint32_t declared_width{0};
    uint32_t declared_height{0};
};

struct DocumentManifest {
    luisa::string producer;
    luisa::string source;       // original document identity (path/name)
    luisa::string rasterizer;   // external rasterizer identity + version
    uint32_t dpi{0};            // rasterization DPI (0 = unknown)
    luisa::vector<DocumentPage> pages;
};

// Parse a document manifest JSON (checked). Relative image paths are resolved
// against `manifest_dir`.
[[nodiscard]] Result<DocumentManifest> parse_document_manifest(
    luisa::string_view json, const luisa::filesystem::path &manifest_dir);
[[nodiscard]] Result<DocumentManifest> load_document_manifest(const luisa::filesystem::path &path);

// Validate page count / declared dimensions against the limits (no decode).
[[nodiscard]] Status validate_document(const DocumentManifest &manifest, const DocumentLimits &limits);

// A loaded document: decoded, oriented page images in reading order.
struct LoadedDocument {
    DocumentManifest manifest;
    luisa::vector<DecodedImage> pages;
    uint64_t total_pixels{0};
};

// Decode every page image with the given options and enforce the limits.
[[nodiscard]] Result<LoadedDocument> load_document(
    const DocumentManifest &manifest, const DocumentLimits &limits,
    const ImageDecodeOptions &image_options = {});

}// namespace luisa::ocr