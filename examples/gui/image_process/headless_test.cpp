// Image Process example -- headless self test.
//
// The self test does not open any window: it can run on every compute backend
// (vk, dx, cuda, ...). It covers:
//   1. operator encoding / decoding and the host CPU reference,
//   2. the GPU kernel against the CPU reference for several operator chains,
//   3. every image format the example can write (generated with stb itself),
//      checking load -> upload -> process -> readback -> save -> reload,
//   4. edge cases: empty operator list, tiny images, partial thread blocks,
//      the maximum operator count and load-only formats (PNM).
//
// Example:
//   xmake run example_image_process vk --headless --output-dir image_process_out

#include "image_process.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>

namespace image_process {

namespace {

[[nodiscard]] luisa::filesystem::path to_path(luisa::string_view utf8) noexcept {
#ifdef _WIN32
    return luisa::filesystem::path{
        std::u8string{reinterpret_cast<const char8_t *>(utf8.data()), utf8.size()}};
#else
    return luisa::filesystem::path{std::string{utf8}};
#endif
}

[[nodiscard]] luisa::string to_utf8(const luisa::filesystem::path &path) noexcept {
    auto u8 = path.u8string();
    return luisa::string{reinterpret_cast<const char *>(u8.data()), u8.size()};
}

[[nodiscard]] OperatorEntry op(OpCode code, float r = 0.0f, float g = 0.0f,
                               float b = 0.0f, float a = 0.0f) noexcept {
    OperatorEntry entry;
    entry.code = code;
    entry.argument[0] = r;
    entry.argument[1] = g;
    entry.argument[2] = b;
    entry.argument[3] = a;
    return entry;
}

struct TestReporter {
    const char *name{""};
    size_t checks{0u};
    size_t failures{0u};

    void check(bool condition, luisa::string_view what) noexcept {
        checks++;
        if (!condition) {
            failures++;
            LUISA_WARNING("[test:{}] FAILED: {}", name, what);
        }
    }
};

/// Deterministic pseudo random source for the host test images.
struct Rng {
    uint32_t state;
    [[nodiscard]] float next() noexcept {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>((state >> 8u) & 0xffffu) / 65535.0f;
    }
};

/// Smooth gradients + soft blobs; friendly to lossy formats.
[[nodiscard]] luisa::vector<float> make_smooth_image(uint32_t width, uint32_t height) noexcept {
    luisa::vector<float> pixels(static_cast<size_t>(width) * height * 4u);
    for (auto y = 0u; y < height; y++) {
        for (auto x = 0u; x < width; x++) {
            auto u = static_cast<float>(x) / static_cast<float>(std::max(width - 1u, 1u));
            auto v = static_cast<float>(y) / static_cast<float>(std::max(height - 1u, 1u));
            auto blob = [&](float cx, float cy, float r) noexcept {
                auto dx = u - cx;
                auto dy = v - cy;
                return std::exp(-(dx * dx + dy * dy) / (2.0f * r * r));
            };
            auto r = std::clamp(0.2f + 0.6f * u + 0.2f * blob(0.3f, 0.3f, 0.15f), 0.0f, 1.0f);
            auto g = std::clamp(0.3f + 0.5f * v + 0.2f * blob(0.7f, 0.5f, 0.2f), 0.0f, 1.0f);
            auto b = std::clamp(0.5f + 0.4f * (1.0f - u) * v + 0.1f * blob(0.5f, 0.8f, 0.1f), 0.0f, 1.0f);
            auto a = std::clamp(0.1f + 0.9f * u, 0.0f, 1.0f);
            auto offset = (static_cast<size_t>(y) * width + x) * 4u;
            pixels[offset + 0u] = r;
            pixels[offset + 1u] = g;
            pixels[offset + 2u] = b;
            pixels[offset + 3u] = a;
        }
    }
    return pixels;
}

/// High frequency pattern, used to exercise the pipeline against the CPU reference.
[[nodiscard]] luisa::vector<float> make_noise_image(uint32_t width, uint32_t height,
                                                    uint32_t seed) noexcept {
    luisa::vector<float> pixels(static_cast<size_t>(width) * height * 4u);
    Rng rng{seed};
    for (auto y = 0u; y < height; y++) {
        for (auto x = 0u; x < width; x++) {
            auto offset = (static_cast<size_t>(y) * width + x) * 4u;
            auto checker = ((x / 3u + y / 5u) % 2u) == 0u ? 0.85f : 0.15f;
            pixels[offset + 0u] = std::clamp(checker * 0.7f + rng.next() * 0.3f, 0.0f, 1.0f);
            pixels[offset + 1u] = std::clamp(rng.next(), 0.0f, 1.0f);
            pixels[offset + 2u] = std::clamp(1.0f - checker, 0.0f, 1.0f);
            pixels[offset + 3u] = std::clamp(rng.next() * 0.5f + 0.5f, 0.0f, 1.0f);
        }
    }
    return pixels;
}

/// Minimal binary PPM (P6) writer: stb_image can *load* PNM but cannot write it.
[[nodiscard]] luisa::vector<std::byte> make_ppm_bytes(luisa::span<const float> rgba,
                                                      uint32_t width, uint32_t height) noexcept {
    luisa::string header = luisa::format("P6\n{} {}\n255\n", width, height);
    luisa::vector<std::byte> bytes;
    bytes.reserve(header.size() + static_cast<size_t>(width) * height * 3u);
    for (auto c : header) { bytes.emplace_back(static_cast<std::byte>(c)); }
    for (size_t i = 0u; i < static_cast<size_t>(width) * height; i++) {
        for (auto c = 0u; c < 3u; c++) {
            auto v = std::lround(std::clamp(rgba[i * 4u + c], 0.0f, 1.0f) * 255.0f);
            bytes.emplace_back(static_cast<std::byte>(v));
        }
    }
    return bytes;
}

/// Minimal uncompressed 8 bit RGB PSD writer (stb_image can *load* PSD but not write it).
[[nodiscard]] luisa::vector<std::byte> make_psd_bytes(luisa::span<const float> rgba,
                                                      uint32_t width, uint32_t height) noexcept {
    luisa::vector<std::byte> bytes;
    auto put8 = [&](uint32_t v) { bytes.emplace_back(static_cast<std::byte>(v & 0xffu)); };
    auto put16 = [&](uint32_t v) { put8(v >> 8u); put8(v); };
    auto put32 = [&](uint32_t v) { put16(v >> 16u); put16(v & 0xffffu); };
    for (auto c : luisa::string_view{"8BPS"}) { bytes.emplace_back(static_cast<std::byte>(c)); }
    put16(1u);                                  // version
    for (auto i = 0u; i < 6u; i++) { put8(0u); }// reserved
    put16(3u);                                  // channels (R, G, B)
    put32(height);                              // rows
    put32(width);                               // columns
    put16(8u);                                  // bit depth
    put16(3u);                                  // color mode: RGB
    put32(0u);                                  // color mode data length
    put32(0u);                                  // image resources length
    put32(0u);                                  // layer and mask length
    put16(0u);                                  // compression: none, planar channels
    for (auto c = 0u; c < 3u; c++) {
        for (size_t i = 0u; i < static_cast<size_t>(width) * height; i++) {
            put8(static_cast<uint32_t>(std::lround(std::clamp(rgba[i * 4u + c], 0.0f, 1.0f) * 255.0f)));
        }
    }
    return bytes;
}

/// Quantize an RGBA float image the same way the 8 bit encoders do.
[[nodiscard]] luisa::vector<float> quantize8(luisa::span<const float> rgba) noexcept {
    luisa::vector<float> result(rgba.size());
    for (size_t i = 0u; i < rgba.size(); i++) {
        auto v = std::lround(std::clamp(rgba[i], 0.0f, 1.0f) * 255.0f);
        result[i] = static_cast<float>(v) / 255.0f;
    }
    return result;
}

/// Force the alpha channel to 1 (formats that store RGB only).
[[nodiscard]] luisa::vector<float> with_opaque_alpha(luisa::vector<float> pixels) noexcept {
    for (size_t i = 0u; i < pixels.size() / 4u; i++) { pixels[i * 4u + 3u] = 1.0f; }
    return pixels;
}

/// Tolerance aware comparison: error / (abs_tol + rel_tol * |expected|) <= 1.
[[nodiscard]] double max_scaled_error(luisa::span<const float> actual,
                                      luisa::span<const float> expected,
                                      double abs_tolerance, double rel_tolerance) noexcept {
    if (actual.size() != expected.size() || actual.empty()) { return 1e30; }
    auto worst = 0.0;
    for (size_t i = 0u; i < actual.size(); i++) {
        auto error = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
        auto scale = std::max(abs_tolerance + rel_tolerance * std::abs(static_cast<double>(expected[i])),
                              1e-30);
        worst = std::max(worst, error / scale);
    }
    return worst;
}

struct FormatExpectation {
    const char *extension;
    bool keeps_alpha;
    bool is_8bit;        // exact 8 bit round trip when true
    double abs_tolerance;// tolerance against the source image
    double rel_tolerance;
};

constexpr FormatExpectation format_expectations[]{
    {".png", true, true, 1e-6, 1e-6},  // lossless, exact 8 bit round trip
    {".bmp", true, true, 1e-6, 1e-6},  // lossless
    {".tga", true, true, 1e-6, 1e-6},  // lossless
    {".jpg", false, false, 0.06, 0.10},// lossy, smooth image + quality 95
    {".hdr", false, false, 0.02, 0.05},// Radiance RGBE, 8 bit mantissa
};

class HeadlessTest {

private:
    Stream _stream;
    ImageProcessPipeline _pipeline;
    luisa::filesystem::path _output_dir;
    TestReporter _reporter;

private:
    [[nodiscard]] luisa::vector<OperatorEntry> complex_chain() const noexcept {
        return {
            op(OpCode::Add, 0.10f, 0.05f, 0.00f, 0.25f),
            op(OpCode::Mul, 0.90f, 0.85f, 0.95f, 1.00f),
            op(OpCode::Pow, 0.85f, 0.75f, 0.65f, 0.90f),
            op(OpCode::Sub, 0.05f, 0.10f, 0.15f, 0.05f),
            op(OpCode::Max, 0.02f, 0.02f, 0.02f, 0.00f),
            op(OpCode::Min, 0.98f, 0.98f, 0.98f, 1.00f),
            op(OpCode::Div, 0.50f, 0.60f, 0.70f, 0.90f),
            op(OpCode::Abs),
            op(OpCode::Add, 0.05f, 0.07f, 0.09f, 0.15f),
            op(OpCode::Mul, 0.95f, 0.95f, 0.95f, 1.00f),
        };
    }

    /// Upload an image, apply a chain on the device and compare with the CPU reference.
    void check_chain(const char *name, luisa::span<const float> rgba, uint2 size,
                     luisa::span<const OperatorEntry> ops,
                     double abs_tolerance, double rel_tolerance) noexcept {
        _pipeline.load(rgba, size);
        _pipeline.set_operators(ops);
        auto gpu = _pipeline.readback_result();
        luisa::vector<uint32_t> encoded;
        encode_operators(ops, encoded);
        auto cpu = apply_operators_cpu(rgba, luisa::span<const uint32_t>{encoded.data(), encoded.size()});
        auto worst = max_scaled_error(gpu, cpu, abs_tolerance, rel_tolerance);
        _reporter.check(worst <= 1.0,
                        luisa::format("chain '{}': GPU/CPU mismatch (scaled error {:.3e})", name, worst));
        if (worst > 1.0) {
            auto stats = compare_pixels(gpu, cpu);
            LUISA_WARNING("chain '{}': max_abs={:.3e} mean_abs={:.3e}", name,
                          stats.max_abs, stats.mean_abs);
        }
        // The alpha pane must expose the processed alpha as opaque gray.
        auto alpha = _pipeline.readback_alpha();
        auto alpha_ok = alpha.size() == gpu.size();
        for (size_t i = 0u; alpha_ok && i < static_cast<size_t>(size.x) * size.y; i++) {
            auto a = gpu[i * 4u + 3u];
            alpha_ok = alpha[i * 4u + 0u] == a && alpha[i * 4u + 1u] == a &&
                       alpha[i * 4u + 2u] == a && alpha[i * 4u + 3u] == 1.0f;
        }
        _reporter.check(alpha_ok, luisa::format("chain '{}': alpha pane mismatch", name));
    }

    void test_operator_encoding() noexcept {
        _reporter.name = "operator-encoding";
        float color[4]{0.25f, 0.50f, 0.75f, 1.00f};
        auto unpacked = unpack_argument(pack_argument(color));
        for (auto c = 0u; c < 4u; c++) {
            _reporter.check(std::abs(unpacked[c] - color[c]) <= 1.0f / 255.0f,
                            "pack/unpack round trip");
        }
        float overflow[4]{2.0f, -1.0f, 0.5f, 0.5f};
        auto clamped = unpack_argument(pack_argument(overflow));
        _reporter.check(clamped[0] == 1.0f && clamped[1] == 0.0f, "argument clamping");

        luisa::vector<OperatorEntry> ops{op(OpCode::Mul, 0.5f, 0.5f, 0.5f, 0.5f),
                                         op(OpCode::Add, 0.25f, 0.25f, 0.25f, 0.25f),
                                         op(OpCode::Abs)};
        luisa::vector<uint32_t> encoded;
        encode_operators(ops, encoded);
        _reporter.check(encoded.size() == ops.size() * operator_stride, "encoded stride");
        _reporter.check(encoded[0] == static_cast<uint32_t>(OpCode::Mul) &&
                            encoded[2] == static_cast<uint32_t>(OpCode::Add) &&
                            encoded[4] == static_cast<uint32_t>(OpCode::Abs),
                        "encoded op codes");

        // 0.5 -> *mul = -> +add = -> abs, evaluated with the quantized arguments
        auto mul_arg = unpack_argument(encoded[1]);
        auto add_arg = unpack_argument(encoded[3]);
        auto expected = std::fabs(0.5f * mul_arg[0] + add_arg[0]);
        auto value = apply_operators_cpu(make_float4(0.5f),
                                         luisa::span<const uint32_t>{encoded.data(), encoded.size()});
        for (auto c = 0u; c < 4u; c++) {
            _reporter.check(std::abs(value[c] - expected) < 1e-6f, "cpu reference chain");
        }

        // Extension helpers used by the load/save dialogs.
        _reporter.check(lower_extension("C:\\pics\\Lena.PNG") == ".png", "lower_extension upper case");
        _reporter.check(lower_extension("no_extension") == "", "lower_extension missing");
        _reporter.check(lower_extension("dir.d/image") == "", "lower_extension dotted dir");
        _reporter.check(is_supported_load_extension(".gif") && is_supported_load_extension(".psd"),
                        "stb loadable formats listed");
        _reporter.check(is_supported_save_extension(".png") && !is_supported_save_extension(".gif"),
                        "save formats listed");
    }

    void test_operator_chains() noexcept {
        _reporter.name = "operator-chains";
        static constexpr auto width = 96u;
        static constexpr auto height = 61u;
        auto image = make_noise_image(width, height, 0x12345678u);
        auto size = make_uint2(width, height);

        auto empty = luisa::vector<OperatorEntry>{};
        auto single_add = luisa::vector<OperatorEntry>{op(OpCode::Add, 0.1f, 0.2f, 0.3f, 0.4f)};
        auto abs_sub = luisa::vector<OperatorEntry>{op(OpCode::Sub, 0.9f, 0.8f, 0.7f, 0.6f),
                                                    op(OpCode::Abs)};
        auto div_zero = luisa::vector<OperatorEntry>{op(OpCode::Div, 0.0f, 0.0f, 0.0f, 0.0f)};
        auto complex = complex_chain();

        check_chain("identity", image, size, empty, 0.0, 0.0);
        // Note: a few ULP of difference between host math and the backend's
        // (fast-math) device math are expected.
        check_chain("single-add", image, size, single_add, 1e-6, 1e-6);
        check_chain("abs-sub", image, size, abs_sub, 1e-6, 1e-6);
        check_chain("div-safe", image, size, div_zero, 1e-3, 1e-5);
        check_chain("complex", image, size, complex, 1e-4, 1e-5);

        // many operators: the loop bound is a runtime uniform, verify it is honored
        luisa::vector<OperatorEntry> many;
        many.reserve(128u);
        for (auto i = 0u; i < 128u; i++) {
            many.emplace_back(op(OpCode::Add, 0.001f, 0.002f, 0.003f, 0.001f));
            many.emplace_back(op(OpCode::Mul, 0.999f, 0.999f, 0.999f, 1.0f));
        }
        check_chain("many-ops", image, size, many, 1e-3, 1e-4);
    }

    void test_image_formats() noexcept {
        _reporter.name = "image-formats";
        static constexpr auto width = 128u;
        static constexpr auto height = 96u;
        auto image = make_smooth_image(width, height);
        auto expected_ldr = quantize8(image);
        auto size = make_uint2(width, height);
        auto ops = luisa::vector<OperatorEntry>{op(OpCode::Mul, 0.9f, 0.9f, 0.9f, 1.0f),
                                                op(OpCode::Add, 0.05f, 0.05f, 0.05f, 0.0f)};
        luisa::vector<uint32_t> encoded;
        encode_operators(ops, encoded);
        auto cpu_processed = apply_operators_cpu(image, luisa::span<const uint32_t>{encoded.data(), encoded.size()});

        for (auto &&expectation : format_expectations) {
            auto extension = luisa::string_view{expectation.extension};

            // 1) write the example image with stb itself and reload it from disk
            luisa::vector<std::byte> encoded_bytes;
            luisa::string error;
            if (!encode_image(extension, image, width, height, encoded_bytes, error)) {
                _reporter.check(false, luisa::format("encode {}: {}", extension, error));
                continue;
            }
            auto file_path = _output_dir / ("source" + std::string{extension});
            auto utf8_path = to_utf8(file_path);
            {
                std::ofstream file{file_path, std::ios::binary | std::ios::trunc};
                file.write(reinterpret_cast<const char *>(encoded_bytes.data()),
                           static_cast<std::streamsize>(encoded_bytes.size()));
                _reporter.check(static_cast<bool>(file),
                                luisa::format("write {} test image", extension));
            }

            ImageData loaded;
            if (!load_image_file(utf8_path, loaded, error)) {
                _reporter.check(false, luisa::format("load {}: {}", extension, error));
                continue;
            }
            _reporter.check(loaded.width == width && loaded.height == height,
                            luisa::format("{} image size", extension));
            // expected content of the reloaded image
            auto expected_loaded = expectation.is_8bit ? expected_ldr : image;
            if (!expectation.keeps_alpha) { expected_loaded = with_opaque_alpha(expected_loaded); }
            auto load_error = max_scaled_error(loaded.pixels, expected_loaded,
                                               expectation.abs_tolerance,
                                               expectation.rel_tolerance);
            _reporter.check(load_error <= 1.0,
                            luisa::format("{} load round trip (scaled error {:.3e})", extension, load_error));

            // 2) process the loaded image on the device and compare with the CPU reference
            auto loaded_size = make_uint2(loaded.width, loaded.height);
            _pipeline.load(loaded.pixels, loaded_size);
            _pipeline.set_operators(ops);
            auto gpu = _pipeline.readback_result();
            auto cpu = apply_operators_cpu(loaded.pixels,
                                           luisa::span<const uint32_t>{encoded.data(), encoded.size()});
            auto gpu_error = max_scaled_error(gpu, cpu, 1e-5, 1e-6);
            _reporter.check(gpu_error <= 1.0,
                            luisa::format("{} gpu vs cpu ({:.3e})", extension, gpu_error));

            // 3) save the processed texture (the GUI path) and reload it
            auto result_path = _output_dir / ("processed" + std::string{extension});
            auto result_utf8 = to_utf8(result_path);
            if (!save_image_file(result_utf8, gpu, width, height, error)) {
                _reporter.check(false, luisa::format("save {}: {}", extension, error));
                continue;
            }
            ImageData reloaded;
            if (!load_image_file(result_utf8, reloaded, error)) {
                _reporter.check(false, luisa::format("reload {}: {}", extension, error));
                continue;
            }
            auto expected_saved = expectation.is_8bit ? quantize8(cpu) : cpu;
            if (!expectation.keeps_alpha) { expected_saved = with_opaque_alpha(expected_saved); }
            auto save_error = max_scaled_error(reloaded.pixels, expected_saved,
                                               expectation.abs_tolerance,
                                               expectation.rel_tolerance);
            _reporter.check(save_error <= 1.0,
                            luisa::format("{} save round trip ({:.3e})", extension, save_error));
            LUISA_INFO("[test:image-formats] {} ok (load {:.3e}, gpu {:.3e}, save {:.3e})",
                       extension, load_error, gpu_error, save_error);
        }
    }

    void test_loadable_only_formats() noexcept {
        _reporter.name = "loadable-formats";
        static constexpr auto width = 32u;
        static constexpr auto height = 24u;
        auto image = make_smooth_image(width, height);
        auto expected = with_opaque_alpha(quantize8(image));// PNM/PSD store RGB only
        struct Case {
            const char *name;
            luisa::vector<std::byte> bytes;
        };
        auto cases = luisa::vector<Case>{};
        cases.push_back(Case{"ppm", make_ppm_bytes(image, width, height)});
        cases.push_back(Case{"psd", make_psd_bytes(image, width, height)});
        for (auto &&c : cases) {
            ImageData loaded;
            luisa::string error;
            auto ok = decode_image(luisa::span<const std::byte>{c.bytes.data(), c.bytes.size()},
                                   loaded, error);
            _reporter.check(ok, luisa::format("decode {}: {}", c.name, error));
            if (!ok) { continue; }
            _reporter.check(loaded.width == width && loaded.height == height,
                            luisa::format("{} size", c.name));
            auto pixels_error = max_scaled_error(loaded.pixels, expected, 1e-6, 1e-6);
            _reporter.check(pixels_error <= 1.0,
                            luisa::format("{} pixels ({:.3e})", c.name, pixels_error));
        }
    }

    void test_edge_cases() noexcept {
        _reporter.name = "edge-cases";
        // tiny image, non multiple of the 16x16 thread block, maximum operator count
        struct Case {
            const char *name;
            uint32_t width;
            uint32_t height;
            size_t operator_count;
        };
        static const Case cases[]{
            {"1x1", 1u, 1u, 3u},
            {"37x23", 37u, 23u, 5u},
            {"64x64-max-ops", 64u, 64u, max_operator_count},
        };
        for (auto &&c : cases) {
            auto image = make_noise_image(c.width, c.height, 0xdeadbeefu);
            luisa::vector<OperatorEntry> ops;
            ops.reserve(c.operator_count);
            for (size_t i = 0u; i < c.operator_count; i++) {
                switch (i % 4u) {
                    case 0u: ops.emplace_back(op(OpCode::Add, 0.01f, 0.02f, 0.03f, 0.01f)); break;
                    case 1u: ops.emplace_back(op(OpCode::Mul, 0.99f, 0.99f, 0.99f, 1.00f)); break;
                    case 2u: ops.emplace_back(op(OpCode::Min, 0.95f, 0.95f, 0.95f, 1.00f)); break;
                    default: ops.emplace_back(op(OpCode::Abs)); break;
                }
            }
            auto tolerance = c.operator_count > 32u ? 1e-3 : 1e-5;
            check_chain(c.name, image, make_uint2(c.width, c.height), ops, tolerance, 1e-5);
        }
    }

public:
    HeadlessTest(Device &device, const luisa::filesystem::path &output_dir) noexcept
        : _stream{device.create_stream(StreamTag::COMPUTE)},
          _pipeline{device, _stream},
          _output_dir{output_dir} {
        std::error_code ec;
        luisa::filesystem::create_directories(_output_dir, ec);
        if (ec) {
            LUISA_WARNING("Cannot create output directory '{}': {}",
                          to_utf8(_output_dir), ec.message());
        }
    }

    [[nodiscard]] int run() noexcept {
        LUISA_INFO("Running headless image process tests (output dir: '{}')", to_utf8(_output_dir));
        test_operator_encoding();
        test_operator_chains();
        test_image_formats();
        test_loadable_only_formats();
        test_edge_cases();
        auto failures = _reporter.failures;
        LUISA_INFO("Headless image process tests: {} checks, {} failures", _reporter.checks, failures);
        return static_cast<int>(failures);
    }
};

}// namespace

int run_headless_tests(Device &device, luisa::string_view output_directory) noexcept {
    auto dir = output_directory.empty() ? luisa::string_view{"image_process_output"} : output_directory;
    HeadlessTest test{device, to_path(dir)};
    return test.run();
}

}// namespace image_process
