// OCR engine -- checked image decode + EXIF orientation (Phase 1).
// stb_image through the repository dependency; no exceptions on bad input.
#include "ocr/image.h"

#include <cstring>
#include <utility>

#include <stb/stb_image.h>

namespace luisa::ocr {

[[nodiscard]] luisa::string_view pixel_order_name(PixelOrder order) noexcept {
    switch (order) {
        case PixelOrder::rgb: return "rgb";
        case PixelOrder::bgr: return "bgr";
        default: return "unknown";
    }
}

namespace {

[[nodiscard]] uint16_t read_u16(const std::byte *p, bool big_endian) noexcept {
    const auto b0 = static_cast<uint16_t>(p[0]);
    const auto b1 = static_cast<uint16_t>(p[1]);
    return big_endian ? static_cast<uint16_t>((b0 << 8u) | b1) : static_cast<uint16_t>((b1 << 8u) | b0);
}
[[nodiscard]] uint32_t read_u32(const std::byte *p, bool big_endian) noexcept {
    const auto b0 = static_cast<uint32_t>(p[0]);
    const auto b1 = static_cast<uint32_t>(p[1]);
    const auto b2 = static_cast<uint32_t>(p[2]);
    const auto b3 = static_cast<uint32_t>(p[3]);
    return big_endian ? (b0 << 24u) | (b1 << 16u) | (b2 << 8u) | b3
                      : (b3 << 24u) | (b2 << 16u) | (b1 << 8u) | b0;
}

}// namespace

// ---------------------------------------------------------------------------
// EXIF orientation (checked JPEG APP1/TIFF parser)
// ---------------------------------------------------------------------------

[[nodiscard]] Result<ExifOrientation> parse_jpeg_exif_orientation(luisa::span<const std::byte> bytes) {
    using R = Result<ExifOrientation>;
    // Walk JPEG segments looking for APP1 "Exif\0\0".
    if (bytes.size() < 4) { return R::success(ExifOrientation::normal); }
    if (static_cast<uint8_t>(bytes[0]) != 0xffu || static_cast<uint8_t>(bytes[1]) != 0xd8u) {
        return R::success(ExifOrientation::normal);// not a JPEG
    }
    size_t pos = 2;
    while (pos + 4 <= bytes.size()) {
        if (static_cast<uint8_t>(bytes[pos]) != 0xffu) { return R::success(ExifOrientation::normal); }
        const auto marker = static_cast<uint8_t>(bytes[pos + 1]);
        // Skip fill bytes.
        if (marker == 0xffu) {
            pos += 1;
            continue;
        }
        if (marker == 0xd8u || marker == 0x01u || (marker >= 0xd0u && marker <= 0xd7u)) {
            pos += 2;
            continue;
        }
        if (marker == 0xdau || marker == 0xd9u) { break; }// start of scan / end
        if (pos + 4 > bytes.size()) { break; }
        const auto seg_len = static_cast<size_t>(read_u16(bytes.data() + pos + 2, true));
        if (seg_len < 2 || pos + 2 + seg_len > bytes.size()) { break; }
        const auto seg_start = pos + 4;
        const auto seg_end = pos + 2 + seg_len;
        if (marker == 0xe1u && seg_end - seg_start >= 6) {
            const auto *payload = bytes.data() + seg_start;
            if (std::memcmp(payload, "Exif\0\0", 6) == 0) {
                const auto tiff = bytes.subspan(seg_start + 6, seg_end - seg_start - 6);
                if (tiff.size() < 8) { return R::success(ExifOrientation::normal); }
                const auto b0 = static_cast<uint8_t>(tiff[0]);
                const auto b1 = static_cast<uint8_t>(tiff[1]);
                bool big_endian = false;
                if (b0 == 'M' && b1 == 'M') { big_endian = true; }
                else if (b0 == 'I' && b1 == 'I') { big_endian = false; }
                else { return R::success(ExifOrientation::normal); }
                if (read_u16(tiff.data() + 2, big_endian) != 42u) { return R::success(ExifOrientation::normal); }
                const auto ifd_offset = read_u32(tiff.data() + 4, big_endian);
                if (ifd_offset < 8 || static_cast<uint64_t>(ifd_offset) + 2 > tiff.size()) {
                    return R::success(ExifOrientation::normal);
                }
                const auto entry_count = read_u16(tiff.data() + ifd_offset, big_endian);
                const auto entries_start = static_cast<uint64_t>(ifd_offset) + 2;
                if (entries_start + static_cast<uint64_t>(entry_count) * 12u > tiff.size()) {
                    return R::success(ExifOrientation::normal);
                }
                for (uint32_t i = 0; i < entry_count; i++) {
                    const auto *entry = tiff.data() + entries_start + static_cast<size_t>(i) * 12u;
                    const auto tag = read_u16(entry, big_endian);
                    if (tag != 0x0112u) { continue; }
                    const auto type = read_u16(entry + 2, big_endian);
                    const auto count = read_u32(entry + 4, big_endian);
                    if (type != 3u || count != 1u) { return R::success(ExifOrientation::normal); }
                    const auto value = read_u16(entry + 8, big_endian);
                    if (value < 1u || value > 8u) { return R::success(ExifOrientation::normal); }
                    return R::success(static_cast<ExifOrientation>(value));
                }
                return R::success(ExifOrientation::normal);
            }
        }
        pos = seg_end;
    }
    return R::success(ExifOrientation::normal);
}

// ---------------------------------------------------------------------------
// Orientation transform
// ---------------------------------------------------------------------------

[[nodiscard]] Status orient_image(luisa::span<const uint8_t> src, uint32_t width, uint32_t height,
                                  ExifOrientation orientation, luisa::vector<uint8_t> &out,
                                  uint32_t &out_width, uint32_t &out_height) {
    const uint64_t pixel_count = static_cast<uint64_t>(width) * height;
    uint64_t byte_count = 0;
    if (!checked_mul(pixel_count, 3u, byte_count) || src.size() != byte_count) {
        return Status::error(StatusCode::invalid_argument, "source pixel span does not match width*height*3", "pixels");
    }
    if (width == 0 || height == 0) {
        return Status::error(StatusCode::invalid_argument, "degenerate image dimensions", "dimensions");
    }
    const auto value = static_cast<uint32_t>(orientation);
    if (value < 1u || value > 8u) {
        return Status::error(StatusCode::invalid_argument, "invalid EXIF orientation", "orientation");
    }
    const bool swaps = value >= 5u;
    out_width = swaps ? height : width;
    out_height = swaps ? width : height;
    out.resize(static_cast<size_t>(byte_count));
    // Map each destination pixel to its source coordinate.
    for (uint32_t y = 0; y < out_height; y++) {
        for (uint32_t x = 0; x < out_width; x++) {
            uint32_t sx = x, sy = y;
            switch (orientation) {
                case ExifOrientation::normal: sx = x; sy = y; break;
                case ExifOrientation::flip_horizontal: sx = width - 1 - x; sy = y; break;
                case ExifOrientation::rotate_180: sx = width - 1 - x; sy = height - 1 - y; break;
                case ExifOrientation::flip_vertical: sx = x; sy = height - 1 - y; break;
                case ExifOrientation::transpose: sx = y; sy = x; break;
                case ExifOrientation::rotate_90_cw: sx = y; sy = height - 1 - x; break;
                // Transverse = 180-degree rotation of the transpose. For a
                // W x H source the destination is H x W, so the source column
                // is bounded by W (width-1-y) and the source row by H
                // (height-1-x). Using height-1-y / width-1-x here reads out of
                // bounds for non-square images.
                case ExifOrientation::transverse: sx = width - 1 - y; sy = height - 1 - x; break;
                case ExifOrientation::rotate_270_cw: sx = width - 1 - y; sy = x; break;
            }
            const auto src_index = (static_cast<size_t>(sy) * width + sx) * 3u;
            const auto dst_index = (static_cast<size_t>(y) * out_width + x) * 3u;
            out[dst_index] = src[src_index];
            out[dst_index + 1] = src[src_index + 1];
            out[dst_index + 2] = src[src_index + 2];
        }
    }
    return Status::success();
}

[[nodiscard]] Status convert_rgb_bgr(luisa::span<uint8_t> pixels, uint64_t pixel_count) {
    uint64_t byte_count = 0;
    if (!checked_mul(pixel_count, 3u, byte_count) || pixels.size() != byte_count) {
        return Status::error(StatusCode::invalid_argument, "pixel span does not match pixel_count*3", "pixels");
    }
    for (uint64_t i = 0; i < pixel_count; i++) {
        const auto base = static_cast<size_t>(i * 3u);
        const auto tmp = pixels[base];
        pixels[base] = pixels[base + 2];
        pixels[base + 2] = tmp;
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

[[nodiscard]] Result<DecodedImage> decode_image(luisa::span<const std::byte> bytes,
                                                const ImageDecodeOptions &options) {
    using R = Result<DecodedImage>;
    if (bytes.empty()) {
        return R::failure(StatusCode::invalid_argument, "empty image buffer", "image");
    }
    if (bytes.size() > options.limits.max_file_bytes) {
        return R::failure(StatusCode::invalid_argument, "image exceeds the file-size limit", "image");
    }
    if (bytes.size() > static_cast<size_t>(INT32_MAX)) {
        return R::failure(StatusCode::invalid_argument, "image too large for the decoder", "image");
    }
    // Probe dimensions first (no pixel allocation) to enforce limits.
    int w = 0, h = 0, comp = 0;
    const auto *raw = reinterpret_cast<const stbi_uc *>(bytes.data());
    if (stbi_info_from_memory(raw, static_cast<int>(bytes.size()), &w, &h, &comp) == 0) {
        return R::failure(StatusCode::invalid_argument,
                          luisa::string{"unsupported or corrupt image: "} + stbi_failure_reason(), "image");
    }
    if (w <= 0 || h <= 0) {
        return R::failure(StatusCode::invalid_argument, "degenerate image dimensions", "image");
    }
    if (static_cast<uint32_t>(w) > options.limits.max_width || static_cast<uint32_t>(h) > options.limits.max_height) {
        return R::failure(StatusCode::invalid_argument, "image dimensions exceed the limit", "image");
    }
    const uint64_t pixel_count = static_cast<uint64_t>(w) * static_cast<uint64_t>(h);
    if (pixel_count > options.limits.max_pixels) {
        return R::failure(StatusCode::invalid_argument, "image pixel count exceeds the limit", "image");
    }
    if (stbi_is_16_bit_from_memory(raw, static_cast<int>(bytes.size())) != 0) {
        return R::failure(StatusCode::unsupported, "16-bit images are not supported (8-bit only)", "image");
    }
    int dw = 0, dh = 0, dc = 0;
    stbi_uc *decoded = stbi_load_from_memory(raw, static_cast<int>(bytes.size()), &dw, &dh, &dc, 3);
    if (decoded == nullptr) {
        return R::failure(StatusCode::invalid_argument,
                          luisa::string{"image decode failed: "} + stbi_failure_reason(), "image");
    }
    DecodedImage image;
    image.width = static_cast<uint32_t>(dw);
    image.height = static_cast<uint32_t>(dh);
    image.channels = 3;
    image.order = options.order;
    uint64_t byte_count = 0;
    if (!checked_mul(static_cast<uint64_t>(dw) * dh, 3u, byte_count)) {
        stbi_image_free(decoded);
        return R::failure(StatusCode::invalid_argument, "decoded image size overflow", "image");
    }
    image.pixels.assign(decoded, decoded + byte_count);
    stbi_image_free(decoded);
    image.row_stride = static_cast<uint64_t>(image.width) * 3u;

    // EXIF orientation (JPEG only; stb does not orient).
    if (options.apply_exif_orientation) {
        auto orientation = parse_jpeg_exif_orientation(bytes);
        if (!orientation.ok()) { return R::failure(orientation.status); }
        image.applied_orientation = *orientation.value;
        if (image.applied_orientation != ExifOrientation::normal) {
            luisa::vector<uint8_t> oriented;
            uint32_t ow = 0, oh = 0;
            auto os = orient_image(image.pixels, image.width, image.height,
                                   image.applied_orientation, oriented, ow, oh);
            if (!os.ok()) { return R::failure(os); }
            image.pixels = std::move(oriented);
            image.width = ow;
            image.height = oh;
            image.row_stride = static_cast<uint64_t>(ow) * 3u;
        }
    }
    if (options.order == PixelOrder::bgr) {
        auto cs = convert_rgb_bgr(image.pixels, static_cast<uint64_t>(image.width) * image.height);
        if (!cs.ok()) { return R::failure(cs); }
    }
    return R::success(std::move(image));
}

[[nodiscard]] Result<DecodedImage> decode_image_file(const luisa::filesystem::path &path,
                                                     const ImageDecodeOptions &options) {
    using R = Result<DecodedImage>;
    const auto path_text = luisa::to_string(path);
    std::FILE *file = std::fopen(path_text.c_str(), "rb");
    if (file == nullptr) {
        return R::failure(StatusCode::not_found,
                          luisa::string{"cannot open image file '"} + path_text + "'", "image");
    }
    luisa::vector<std::byte> bytes;
    std::byte buffer[65536];
    size_t got = 0;
    while ((got = std::fread(buffer, 1, sizeof(buffer), file)) > 0) {
        if (bytes.size() + got > options.limits.max_file_bytes) {
            std::fclose(file);
            return R::failure(StatusCode::invalid_argument, "image exceeds the file-size limit", "image");
        }
        bytes.insert(bytes.end(), buffer, buffer + got);
    }
    const bool read_ok = std::ferror(file) == 0;
    std::fclose(file);
    if (!read_ok) {
        return R::failure(StatusCode::io_error, "failed to read image file", "image");
    }
    return decode_image(bytes, options);
}

}// namespace luisa::ocr