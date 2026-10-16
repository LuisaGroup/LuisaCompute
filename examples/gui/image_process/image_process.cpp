// Image Process example -- host side image I/O and small helpers.
//
// All stb interactions go through memory buffers instead of `stbi_load(char*)`
// so that paths stay UTF-8 safe on Windows (where `fopen` uses the ANSI code
// page). Files are read/written with `std::filesystem::path`, which converts
// UTF-8 to the native wide path.

#include "image_process.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>

namespace image_process {

namespace {

[[nodiscard]] luisa::filesystem::path path_from_utf8(luisa::string_view utf8) noexcept {
#ifdef _WIN32
    return luisa::filesystem::path{
        std::u8string{reinterpret_cast<const char8_t *>(utf8.data()), utf8.size()}};
#else
    return luisa::filesystem::path{std::string{utf8}};
#endif
}

[[nodiscard]] bool read_file(luisa::string_view utf8_path,
                             luisa::vector<std::byte> &bytes,
                             luisa::string &error) noexcept {
    auto path = path_from_utf8(utf8_path);
    std::ifstream file{path, std::ios::binary | std::ios::ate};
    if (!file) {
        error = luisa::format("cannot open file '{}'", utf8_path);
        return false;
    }
    auto size = static_cast<std::streamsize>(file.tellg());
    if (size <= 0) {
        error = luisa::format("file '{}' is empty", utf8_path);
        return false;
    }
    file.seekg(0, std::ios::beg);
    bytes.resize(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char *>(bytes.data()), size)) {
        error = luisa::format("failed to read file '{}'", utf8_path);
        return false;
    }
    return true;
}

[[nodiscard]] bool write_file(luisa::string_view utf8_path,
                              luisa::span<const std::byte> bytes,
                              luisa::string &error) noexcept {
    auto path = path_from_utf8(utf8_path);
    std::ofstream file{path, std::ios::binary | std::ios::trunc};
    if (!file) {
        error = luisa::format("cannot open file '{}' for writing", utf8_path);
        return false;
    }
    file.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        error = luisa::format("failed to write file '{}'", utf8_path);
        return false;
    }
    return true;
}

void append_to_vector(void *context, void *data, int size) noexcept {
    auto &out = *static_cast<luisa::vector<std::byte> *>(context);
    auto *begin = static_cast<std::byte *>(data);
    out.insert(out.end(), begin, begin + size);
}

[[nodiscard]] uint8_t to_byte(float v) noexcept {
    return static_cast<uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f));
}

}// namespace

luisa::string lower_extension(luisa::string_view path) noexcept {
    auto dot = path.find_last_of('.');
    auto slash = path.find_last_of("/\\");
    if (dot == luisa::string_view::npos ||
        (slash != luisa::string_view::npos && dot < slash)) {
        return {};
    }
    auto extension = path.substr(dot);
    luisa::string result{extension};
    luisa::transform(result.begin(), result.end(), result.begin(),
                   [](char c) noexcept { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); });
    return result;
}

bool is_supported_load_extension(luisa::string_view extension) noexcept {
    static constexpr luisa::string_view extensions[]{
        ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".hdr", ".psd",
        ".gif", ".pic", ".ppm", ".pgm", ".pnm"};
    return std::any_of(std::begin(extensions), std::end(extensions),
                       [extension](luisa::string_view e) noexcept { return e == extension; });
}

bool is_supported_save_extension(luisa::string_view extension) noexcept {
    return extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
           extension == ".bmp" || extension == ".tga" || extension == ".hdr";
}

bool decode_image(luisa::span<const std::byte> bytes, ImageData &image,
                  luisa::string &error) noexcept {
    // stb applies a 2.2 gamma when converting 8 bit LDR data to float; keep the
    // stored values as they are so load/save round trips stay lossless.
    stbi_ldr_to_hdr_gamma(1.0f);
    stbi_ldr_to_hdr_scale(1.0f);
    auto width = 0;
    auto height = 0;
    auto channels = 0;
    auto *data = stbi_loadf_from_memory(reinterpret_cast<const stbi_uc *>(bytes.data()),
                                        static_cast<int>(bytes.size()),
                                        &width, &height, &channels, 4);
    if (data == nullptr) {
        error = luisa::format("failed to decode image ({})",
                              stbi_failure_reason() == nullptr ? "unknown" : stbi_failure_reason());
        return false;
    }
    if (width <= 0 || height <= 0) {
        stbi_image_free(data);
        error = "decoded image has invalid dimensions";
        return false;
    }
    image.width = static_cast<uint32_t>(width);
    image.height = static_cast<uint32_t>(height);
    image.pixels.resize(static_cast<size_t>(width) * height * 4u);
    std::memcpy(image.pixels.data(), data, image.pixels.size() * sizeof(float));
    stbi_image_free(data);
    return true;
}

bool load_image_file(luisa::string_view utf8_path, ImageData &image,
                     luisa::string &error) noexcept {
    luisa::vector<std::byte> bytes;
    if (!read_file(utf8_path, bytes, error)) { return false; }
    if (!decode_image(bytes, image, error)) {
        error = luisa::format("'{}': {}", utf8_path, error);
        return false;
    }
    return true;
}

bool decode_image_to_rgba(luisa::span<const std::byte> bytes, luisa::vector<float> &rgba,
                          uint32_t &width, uint32_t &height, luisa::string &error) noexcept {
    ImageData image;
    if (!decode_image(bytes, image, error)) { return false; }
    width = image.width;
    height = image.height;
    rgba = std::move(image.pixels);
    return true;
}

bool encode_image(luisa::string_view extension, luisa::span<const float> rgba,
                  uint32_t width, uint32_t height,
                  luisa::vector<std::byte> &bytes, luisa::string &error) noexcept {
    if (width == 0u || height == 0u) {
        error = "cannot encode an empty image";
        return false;
    }
    if (rgba.size() != static_cast<size_t>(width) * height * 4u) {
        error = "pixel data does not match the image size";
        return false;
    }
    bytes.clear();
    auto pixel_count = static_cast<size_t>(width) * height;
    auto ok = false;
    if (extension == ".hdr") {
        // Radiance HDR stores linear RGB floats; clamp negatives (and drop alpha).
        luisa::vector<float> rgb(pixel_count * 3u);
        for (auto i = 0u; i < pixel_count; i++) {
            rgb[i * 3u + 0u] = std::max(rgba[i * 4u + 0u], 0.0f);
            rgb[i * 3u + 1u] = std::max(rgba[i * 4u + 1u], 0.0f);
            rgb[i * 3u + 2u] = std::max(rgba[i * 4u + 2u], 0.0f);
        }
        ok = stbi_write_hdr_to_func(append_to_vector, &bytes,
                                    static_cast<int>(width), static_cast<int>(height), 3,
                                    rgb.data()) != 0;
    } else if (extension == ".jpg" || extension == ".jpeg") {
        // JPEG has no alpha channel, encode RGB only.
        luisa::vector<uint8_t> ldr(pixel_count * 3u);
        for (auto i = 0u; i < pixel_count; i++) {
            ldr[i * 3u + 0u] = to_byte(rgba[i * 4u + 0u]);
            ldr[i * 3u + 1u] = to_byte(rgba[i * 4u + 1u]);
            ldr[i * 3u + 2u] = to_byte(rgba[i * 4u + 2u]);
        }
        ok = stbi_write_jpg_to_func(append_to_vector, &bytes,
                                    static_cast<int>(width), static_cast<int>(height), 3,
                                    ldr.data(), 95) != 0;
    } else if (extension == ".png" || extension == ".bmp" || extension == ".tga") {
        luisa::vector<uint8_t> ldr(pixel_count * 4u);
        for (auto i = 0u; i < pixel_count * 4u; i++) { ldr[i] = to_byte(rgba[i]); }
        if (extension == ".png") {
            ok = stbi_write_png_to_func(append_to_vector, &bytes,
                                        static_cast<int>(width), static_cast<int>(height), 4,
                                        ldr.data(), static_cast<int>(width) * 4) != 0;
        } else if (extension == ".bmp") {
            ok = stbi_write_bmp_to_func(append_to_vector, &bytes,
                                        static_cast<int>(width), static_cast<int>(height), 4,
                                        ldr.data()) != 0;
        } else {
            ok = stbi_write_tga_to_func(append_to_vector, &bytes,
                                        static_cast<int>(width), static_cast<int>(height), 4,
                                        ldr.data()) != 0;
        }
    } else {
        error = luisa::format("unsupported image extension '{}'", extension);
        return false;
    }
    if (!ok) {
        error = luisa::format("failed to encode image as '{}'", extension);
        return false;
    }
    return true;
}

bool save_image_file(luisa::string_view utf8_path, luisa::span<const float> rgba,
                     uint32_t width, uint32_t height, luisa::string &error) noexcept {
    luisa::vector<std::byte> bytes;
    if (!encode_image(lower_extension(utf8_path), rgba, width, height, bytes, error)) {
        return false;
    }
    return write_file(utf8_path, bytes, error);
}

PixelDiffStats compare_pixels(luisa::span<const float> a,
                              luisa::span<const float> b) noexcept {
    PixelDiffStats stats;
    if (a.size() != b.size() || a.empty()) { return stats; }
    stats.pixels = a.size();
    auto sum = 0.0;
    for (auto i = 0u; i < a.size(); i++) {
        auto diff = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        sum += diff;
        stats.max_abs = std::max(stats.max_abs, diff);
        if (diff > 0.0) { stats.differing++; }
    }
    stats.mean_abs = sum / static_cast<double>(a.size());
    return stats;
}

}// namespace image_process
