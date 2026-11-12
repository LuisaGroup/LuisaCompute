// OCR engine -- document input (Phase 1). No in-process PDF parser.
#include "ocr/document.h"

#include <utility>

#include <yyjson.h>

namespace luisa::ocr {
namespace {

struct JsonDoc {
    yyjson_doc *doc{nullptr};
    ~JsonDoc() noexcept {
        if (doc != nullptr) { yyjson_doc_free(doc); }
    }
};

[[nodiscard]] luisa::string_view json_str(yyjson_val *v) noexcept {
    if (v == nullptr || !yyjson_is_str(v)) { return {}; }
    return {yyjson_get_str(v), yyjson_get_len(v)};
}

// Read a JSON unsigned integer that must fit in uint32. yyjson stores uints as
// 64-bit; narrowing without this check would silently wrap oversized values.
[[nodiscard]] bool json_u32(yyjson_val *v, uint32_t &out) noexcept {
    if (v == nullptr || !yyjson_is_uint(v)) { return false; }
    const auto value = yyjson_get_uint(v);
    if (value > 0xffffffffull) { return false; }
    out = static_cast<uint32_t>(value);
    return true;
}

[[nodiscard]] Result<luisa::string> read_text_file(const luisa::filesystem::path &path, uint64_t max_bytes) {
    using R = Result<luisa::string>;
    const auto path_text = luisa::to_string(path);
    std::FILE *file = std::fopen(path_text.c_str(), "rb");
    if (file == nullptr) {
        return R::failure(StatusCode::not_found,
                          luisa::string{"cannot open document manifest '"} + path_text + "'", "manifest");
    }
    luisa::string bytes;
    char buffer[65536];
    size_t got = 0;
    while ((got = std::fread(buffer, 1, sizeof(buffer), file)) > 0) {
        if (bytes.size() + got > max_bytes) {
            std::fclose(file);
            return R::failure(StatusCode::invalid_argument, "document manifest exceeds the size limit", "manifest");
        }
        bytes.append(buffer, got);
    }
    const bool read_ok = std::ferror(file) == 0;
    std::fclose(file);
    if (!read_ok) { return R::failure(StatusCode::io_error, "failed to read document manifest", "manifest"); }
    return R::success(std::move(bytes));
}

}// namespace

[[nodiscard]] Result<DocumentManifest> parse_document_manifest(
    luisa::string_view json, const luisa::filesystem::path &manifest_dir) {
    using R = Result<DocumentManifest>;
    if (json.size() > 16ull * 1024ull * 1024ull) {
        return R::failure(StatusCode::invalid_argument, "document manifest exceeds the size limit", "manifest");
    }
    JsonDoc holder;
    yyjson_read_err err;
    holder.doc = yyjson_read_opts(const_cast<char *>(json.data()), json.size(), YYJSON_READ_NOFLAG, nullptr, &err);
    if (holder.doc == nullptr) {
        return R::failure(StatusCode::invalid_argument,
                          luisa::string{"document manifest JSON parse error: "} + (err.msg != nullptr ? err.msg : "unknown"), "manifest");
    }
    auto *root = yyjson_doc_get_root(holder.doc);
    if (!yyjson_is_obj(root)) {
        return R::failure(StatusCode::invalid_argument, "document manifest root must be an object", "manifest");
    }
    auto *version = yyjson_obj_get(root, "format_version");
    if (version == nullptr || !yyjson_is_uint(version) || yyjson_get_uint(version) != 1u) {
        return R::failure(StatusCode::unsupported, "document manifest format_version must be 1", "format_version");
    }
    DocumentManifest manifest;
    const auto assign = [&](const char *key, luisa::string &dst) {
        dst = luisa::string{json_str(yyjson_obj_get(root, key))};
    };
    assign("producer", manifest.producer);
    assign("source", manifest.source);
    assign("rasterizer", manifest.rasterizer);
    if (auto *dpi = yyjson_obj_get(root, "dpi"); dpi != nullptr && !yyjson_is_null(dpi)) {
        uint32_t dpi_value = 0;
        if (!json_u32(dpi, dpi_value)) {
            return R::failure(StatusCode::invalid_argument, "dpi must be a uint32", "dpi");
        }
        manifest.dpi = dpi_value;
    }
    auto *pages = yyjson_obj_get(root, "pages");
    if (pages == nullptr || !yyjson_is_arr(pages)) {
        return R::failure(StatusCode::invalid_argument, "document manifest needs a pages array", "pages");
    }
    yyjson_val *item;
    size_t pi, pm;
    yyjson_arr_foreach(pages, pi, pm, item) {
        if (!yyjson_is_obj(item)) {
            return R::failure(StatusCode::invalid_argument, "page entry must be an object", "pages");
        }
        DocumentPage page;
        if (auto *idx = yyjson_obj_get(item, "source_page_index"); idx != nullptr && !yyjson_is_null(idx)) {
            uint32_t index_value = 0;
            if (!json_u32(idx, index_value)) {
                return R::failure(StatusCode::invalid_argument, "source_page_index must be a uint32", "pages");
            }
            page.source_page_index = index_value;
        } else {
            if (manifest.pages.size() > 0xffffffffull) {
                return R::failure(StatusCode::invalid_argument, "too many pages for a uint32 index", "pages");
            }
            page.source_page_index = static_cast<uint32_t>(manifest.pages.size());
        }
        const auto image_text = json_str(yyjson_obj_get(item, "image"));
        if (image_text.empty()) {
            return R::failure(StatusCode::invalid_argument, "page entry needs a non-empty image path", "pages");
        }
        luisa::filesystem::path image_path;
        if (!luisa::path_from_utf8(image_text, image_path)) {
            return R::failure(StatusCode::invalid_argument, "page image path is not valid UTF-8", "pages");
        }
        if (image_path.is_relative()) {
            image_path = manifest_dir / image_path;
        }
        page.image_path = std::move(image_path);
        if (auto *w = yyjson_obj_get(item, "width"); w != nullptr && !yyjson_is_null(w)) {
            uint32_t width_value = 0;
            if (!json_u32(w, width_value)) {
                return R::failure(StatusCode::invalid_argument, "page width must be a uint32", "pages");
            }
            page.declared_width = width_value;
        }
        if (auto *h = yyjson_obj_get(item, "height"); h != nullptr && !yyjson_is_null(h)) {
            uint32_t height_value = 0;
            if (!json_u32(h, height_value)) {
                return R::failure(StatusCode::invalid_argument, "page height must be a uint32", "pages");
            }
            page.declared_height = height_value;
        }
        manifest.pages.emplace_back(std::move(page));
    }
    if (manifest.pages.empty()) {
        return R::failure(StatusCode::invalid_argument, "document manifest has no pages", "pages");
    }
    return R::success(std::move(manifest));
}

[[nodiscard]] Result<DocumentManifest> load_document_manifest(const luisa::filesystem::path &path) {
    auto text = read_text_file(path, 16ull * 1024ull * 1024ull);
    if (!text.ok()) { return Result<DocumentManifest>::failure(text.status); }
    return parse_document_manifest(*text.value, path.parent_path());
}

[[nodiscard]] Status validate_document(const DocumentManifest &manifest, const DocumentLimits &limits) {
    if (manifest.pages.empty()) {
        return Status::error(StatusCode::invalid_argument, "document has no pages", "pages");
    }
    if (manifest.pages.size() > limits.max_pages) {
        return Status::error(StatusCode::invalid_argument, "document page count exceeds the limit", "pages");
    }
    uint64_t total = 0;
    for (const auto &page : manifest.pages) {
        if (page.declared_width != 0 && page.declared_height != 0) {
            if (page.declared_width > limits.image.max_width || page.declared_height > limits.image.max_height) {
                return Status::error(StatusCode::invalid_argument, "declared page dimensions exceed the limit", "pages");
            }
            const uint64_t pixels = static_cast<uint64_t>(page.declared_width) * page.declared_height;
            if (!checked_add(total, pixels, total) || total > limits.max_total_pixels) {
                return Status::error(StatusCode::invalid_argument, "document total pixel count exceeds the limit", "pages");
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Result<LoadedDocument> load_document(
    const DocumentManifest &manifest, const DocumentLimits &limits,
    const ImageDecodeOptions &image_options) {
    using R = Result<LoadedDocument>;
    auto valid = validate_document(manifest, limits);
    if (!valid.ok()) { return R::failure(std::move(valid)); }
    LoadedDocument document;
    document.manifest = manifest;
    for (const auto &page : manifest.pages) {
        auto image = decode_image_file(page.image_path, image_options);
        if (!image.ok()) { return R::failure(image.status); }
        if (page.declared_width != 0 && image.value->width != page.declared_width) {
            return R::failure(StatusCode::invalid_argument, "decoded page width does not match the manifest", "pages");
        }
        if (page.declared_height != 0 && image.value->height != page.declared_height) {
            return R::failure(StatusCode::invalid_argument, "decoded page height does not match the manifest", "pages");
        }
        const uint64_t pixels = static_cast<uint64_t>(image.value->width) * image.value->height;
        if (!checked_add(document.total_pixels, pixels, document.total_pixels) ||
            document.total_pixels > limits.max_total_pixels) {
            return R::failure(StatusCode::invalid_argument, "document total pixel count exceeds the limit", "pages");
        }
        document.pages.emplace_back(std::move(*image.value));
    }
    return R::success(std::move(document));
}

}// namespace luisa::ocr