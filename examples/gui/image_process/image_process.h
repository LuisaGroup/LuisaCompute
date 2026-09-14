// Image Process example -- shared core.
//
// The example loads a picture (any format supported by stb_image), uploads it
// to a FLOAT4 texture, applies an ordered list of per-pixel operators inside a
// single compute dispatch, displays the result (RGB pane + alpha pane) and can
// write the processed texture back to disk (PNG/BMP/TGA/JPG/HDR).
//
// Operator encoding
// -----------------
// The operator list is uploaded as a flat buffer of uints, two uints per
// operator:  [ op_code, packed_rgba8_argument ].
// The kernel therefore keeps the exact signature asked for by the example
// contract -- `BufferVar<uint> operators, UInt operator_count` -- and a runtime
// `$for` loop with a `$switch` decides which operator to apply for each entry.
// Arguments are edited with a color picker in the GUI, so 8 bit per channel is
// plenty; CPU reference and GPU kernel decode the very same packed value.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include <luisa/core/basic_types.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/deque.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

namespace image_process {

using namespace luisa;
using namespace luisa::compute;

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

enum class OpCode : uint32_t {
    Add = 0u,
    Sub = 1u,
    Mul = 2u,
    Div = 3u,
    Min = 4u,
    Max = 5u,
    Pow = 6u,
    Abs = 7u,
};

inline constexpr size_t op_code_count = 8u;
/// Number of uints used by one encoded operator: [code, packed argument].
inline constexpr uint32_t operator_stride = 2u;
/// Division uses `value / copysign(max(|arg|, eps), arg)` to stay well defined.
inline constexpr float division_epsilon = 1e-6f;
/// Upper bound of operators the GUI allows (operator buffer is pre-allocated).
inline constexpr size_t max_operator_count = 256u;
/// Number of in-flight operator uploads kept alive for the async stream.
inline constexpr size_t pending_upload_retention = 8u;

[[nodiscard]] inline const char *op_name(OpCode op) noexcept {
    switch (op) {
        case OpCode::Add: return "Add";
        case OpCode::Sub: return "Sub";
        case OpCode::Mul: return "Mul";
        case OpCode::Div: return "Div";
        case OpCode::Min: return "Min";
        case OpCode::Max: return "Max";
        case OpCode::Pow: return "Pow";
        case OpCode::Abs: return "Abs";
    }
    return "Add";
}

[[nodiscard]] inline bool op_uses_argument(OpCode op) noexcept {
    return op != OpCode::Abs;
}

/// One entry of the operator list shown/edited in the GUI.
struct OperatorEntry {
    OpCode code{OpCode::Add};
    /// RGBA argument, edited with a color picker.
    float argument[4]{0.1f, 0.1f, 0.1f, 0.0f};
};

/// Quantize an RGBA color to 8 bit per channel (matches the kernel decode).
[[nodiscard]] inline uint32_t pack_argument(const float *rgba) noexcept {
    auto quantize = [](float v) noexcept {
        return static_cast<uint32_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f));
    };
    return quantize(rgba[0]) |
           (quantize(rgba[1]) << 8u) |
           (quantize(rgba[2]) << 16u) |
           (quantize(rgba[3]) << 24u);
}

[[nodiscard]] inline float4 unpack_argument(uint32_t packed) noexcept {
    return make_float4(static_cast<float>(packed & 0xffu),
                       static_cast<float>((packed >> 8u) & 0xffu),
                       static_cast<float>((packed >> 16u) & 0xffu),
                       static_cast<float>(packed >> 24u)) *
           (1.0f / 255.0f);
}

/// Flatten the operator list into the uint buffer layout used by the kernel.
inline void encode_operators(luisa::span<const OperatorEntry> ops,
                             luisa::vector<uint32_t> &out) noexcept {
    out.clear();
    out.reserve(ops.size() * operator_stride);
    for (auto &&op : ops) {
        out.emplace_back(static_cast<uint32_t>(op.code));
        out.emplace_back(pack_argument(op.argument));
    }
}

// ---------------------------------------------------------------------------
// CPU reference (mirrors the kernel switch exactly)
// ---------------------------------------------------------------------------

[[nodiscard]] inline float4 apply_operators_cpu(float4 value, uint32_t code, float4 arg) noexcept {
    switch (static_cast<OpCode>(code)) {
        case OpCode::Add:
            for (auto c = 0u; c < 4u; c++) { value[c] += arg[c]; }
            break;
        case OpCode::Sub:
            for (auto c = 0u; c < 4u; c++) { value[c] -= arg[c]; }
            break;
        case OpCode::Mul:
            for (auto c = 0u; c < 4u; c++) { value[c] *= arg[c]; }
            break;
        case OpCode::Div:
            for (auto c = 0u; c < 4u; c++) {
                auto denom = std::copysign(std::max(std::fabs(arg[c]), division_epsilon), arg[c]);
                value[c] /= denom;
            }
            break;
        case OpCode::Min:
            for (auto c = 0u; c < 4u; c++) { value[c] = std::min(value[c], arg[c]); }
            break;
        case OpCode::Max:
            for (auto c = 0u; c < 4u; c++) { value[c] = std::max(value[c], arg[c]); }
            break;
        case OpCode::Pow:
            for (auto c = 0u; c < 4u; c++) { value[c] = std::pow(value[c], arg[c]); }
            break;
        case OpCode::Abs:
            for (auto c = 0u; c < 4u; c++) { value[c] = std::fabs(value[c]); }
            break;
    }
    return value;
}

[[nodiscard]] inline float4 apply_operators_cpu(float4 value, luisa::span<const uint32_t> encoded) noexcept {
    auto count = encoded.size() / operator_stride;
    for (auto i = 0u; i < count; i++) {
        auto code = encoded[i * operator_stride + 0u];
        auto arg = unpack_argument(encoded[i * operator_stride + 1u]);
        value = apply_operators_cpu(value, code, arg);
    }
    return value;
}

/// Apply the encoded operators to every pixel of an RGBA float image.
[[nodiscard]] inline luisa::vector<float> apply_operators_cpu(luisa::span<const float> rgba,
                                                              luisa::span<const uint32_t> encoded) noexcept {
    luisa::vector<float> result(rgba.size());
    auto pixel_count = rgba.size() / 4u;
    for (auto i = 0u; i < pixel_count; i++) {
        auto offset = i * 4u;
        auto value = apply_operators_cpu(make_float4(rgba[offset + 0u], rgba[offset + 1u],
                                                     rgba[offset + 2u], rgba[offset + 3u]),
                                         encoded);
        for (auto c = 0u; c < 4u; c++) { result[offset + c] = value[c]; }
    }
    return result;
}

// ---------------------------------------------------------------------------
// GPU kernel
// ---------------------------------------------------------------------------

using ImageProcessShader = Shader<2, Image<float>, Image<float>, Image<float>, Buffer<uint32_t>, uint32_t>;

/// Build the pixel processing shader:
///   value = input.read(coord);
///   for (i < operator_count) { switch (operators[i*2]) { ... } }
///   output.write(coord, value);
///   alpha_output.write(coord, gray(value.a));
[[nodiscard]] inline ImageProcessShader make_image_process_shader(Device &device) noexcept {
    return device.compile<2>(
        [](ImageFloat input, ImageFloat output, ImageFloat alpha_output,
           BufferUInt operators, UInt operator_count) noexcept {
            set_block_size(16u, 16u, 1u);
            auto coord = dispatch_id().xy();
            Float4 value = input.read(coord);
            $for (index, operator_count) {
                UInt code = operators.read(index * operator_stride + 0u);
                UInt packed = operators.read(index * operator_stride + 1u);
                Float4 arg = make_float4(cast<float>(packed & 0xffu),
                                         cast<float>((packed >> 8u) & 0xffu),
                                         cast<float>((packed >> 16u) & 0xffu),
                                         cast<float>(packed >> 24u)) *
                             (1.0f / 255.0f);
                $switch (code) {
                    $case (static_cast<uint>(OpCode::Add)) {
                        value = value + arg;
                    };
                    $case (static_cast<uint>(OpCode::Sub)) {
                        value = value - arg;
                    };
                    $case (static_cast<uint>(OpCode::Mul)) {
                        value = value * arg;
                    };
                    $case (static_cast<uint>(OpCode::Div)) {
                        value = value / copysign(max(abs(arg), make_float4(division_epsilon)), arg);
                    };
                    $case (static_cast<uint>(OpCode::Min)) {
                        value = min(value, arg);
                    };
                    $case (static_cast<uint>(OpCode::Max)) {
                        value = max(value, arg);
                    };
                    $case (static_cast<uint>(OpCode::Pow)) {
                        value = pow(value, arg);
                    };
                    $case (static_cast<uint>(OpCode::Abs)) {
                        value = abs(value);
                    };
                    $default {};
                };
            };
            output.write(coord, value);
            alpha_output.write(coord, make_float4(value.w, value.w, value.w, 1.0f));
        });
}

// ---------------------------------------------------------------------------
// Device side pipeline shared by the GUI and the headless tests
// ---------------------------------------------------------------------------

class ImageProcessPipeline {

private:
    Device &_device;
    Stream &_stream;
    ImageProcessShader _shader;
    Buffer<uint32_t> _operators_buffer;
    Image<float> _input;
    Image<float> _result;
    Image<float> _alpha;
    uint2 _size{};
    luisa::deque<luisa::unique_ptr<luisa::vector<uint32_t>>> _pending_uploads;

public:
    ImageProcessPipeline(Device &device, Stream &stream) noexcept
        : _device{device},
          _stream{stream},
          _shader{make_image_process_shader(device)},
          _operators_buffer{device.create_buffer<uint32_t>(max_operator_count * operator_stride)} {}

    ImageProcessPipeline(const ImageProcessPipeline &) noexcept = delete;
    ImageProcessPipeline(ImageProcessPipeline &&) noexcept = delete;
    ImageProcessPipeline &operator=(const ImageProcessPipeline &) noexcept = delete;
    ImageProcessPipeline &operator=(ImageProcessPipeline &&) noexcept = delete;

    [[nodiscard]] bool has_image() const noexcept { return _size.x != 0u && _size.y != 0u; }
    [[nodiscard]] uint2 size() const noexcept { return _size; }
    [[nodiscard]] Image<float> &input() noexcept { return _input; }
    [[nodiscard]] Image<float> &result() noexcept { return _result; }
    [[nodiscard]] Image<float> &alpha() noexcept { return _alpha; }

    /// Upload an RGBA float image and (re-)create the device textures.
    void load(luisa::span<const float> rgba, uint2 size) noexcept {
        LUISA_ASSERT(size.x != 0u && size.y != 0u && !rgba.empty(),
                     "Invalid image size.");
        LUISA_ASSERT(rgba.size() == static_cast<size_t>(size.x) * size.y * 4u,
                     "Pixel data does not match the image size.");
        // Wait for any in-flight work that may still reference the old textures
        // before destroying/recreating them.
        _stream << synchronize();
        _input = _device.create_image<float>(PixelStorage::FLOAT4, size);
        _result = _device.create_image<float>(PixelStorage::FLOAT4, size);
        _alpha = _device.create_image<float>(PixelStorage::FLOAT4, size);
        _size = size;
        _stream << _input.copy_from(luisa::span<const float>{rgba.data(), rgba.size()})
                << synchronize();
    }

    /// Re-upload the operator list and dispatch the processing kernel.
    void set_operators(luisa::span<const OperatorEntry> ops) noexcept {
        if (!has_image()) { return; }
        LUISA_ASSERT(ops.size() <= max_operator_count,
                     "Too many operators ({} > {}).", ops.size(), max_operator_count);
        auto upload = luisa::make_unique<luisa::vector<uint32_t>>();
        encode_operators(ops, *upload);
        if (!upload->empty()) {
            auto view = _operators_buffer.view().subview(0u, upload->size());
            _stream << view.copy_from(luisa::span<uint32_t>{upload->data(), upload->size()});
            // Keep the staging data alive for a few dispatches: the backend
            // copies from the pointer when it records the command.
            _pending_uploads.emplace_back(std::move(upload));
            while (_pending_uploads.size() > pending_upload_retention) {
                _pending_uploads.pop_front();
            }
        }
        _stream << _shader(_input, _result, _alpha, _operators_buffer,
                           static_cast<uint32_t>(ops.size()))
                       .dispatch(_size);
    }

    /// Read the processed RGB(A) image back to host memory (synchronous).
    [[nodiscard]] luisa::vector<float> readback_result() noexcept {
        luisa::vector<float> pixels(static_cast<size_t>(_size.x) * _size.y * 4u);
        if (has_image()) {
            _stream << _result.copy_to(luisa::span<float>{pixels.data(), pixels.size()})
                    << synchronize();
        }
        return pixels;
    }

    /// Read the grayscale alpha image back to host memory (synchronous).
    [[nodiscard]] luisa::vector<float> readback_alpha() noexcept {
        luisa::vector<float> pixels(static_cast<size_t>(_size.x) * _size.y * 4u);
        if (has_image()) {
            _stream << _alpha.copy_to(luisa::span<float>{pixels.data(), pixels.size()})
                    << synchronize();
        }
        return pixels;
    }
};

// ---------------------------------------------------------------------------
// Host side image I/O (stb_image / stb_image_write, memory based)
// ---------------------------------------------------------------------------

struct ImageData {
    uint32_t width{0u};
    uint32_t height{0u};
    luisa::vector<float> pixels;// RGBA float
    [[nodiscard]] bool empty() const noexcept { return width == 0u || height == 0u; }
};

struct PixelDiffStats {
    size_t pixels{0u};
    size_t differing{0u};
    double max_abs{0.0};
    double mean_abs{0.0};
};

/// Lower cased extension of a path, including the leading dot ("" if none).
[[nodiscard]] luisa::string lower_extension(std::string_view path) noexcept;

/// All extensions accepted by the loader (stb_image).
[[nodiscard]] bool is_supported_load_extension(std::string_view extension) noexcept;
/// Extensions the example can write (stb_image_write).
[[nodiscard]] bool is_supported_save_extension(std::string_view extension) noexcept;

/// Decode an image from memory (any stb_image format) into RGBA float pixels.
[[nodiscard]] bool decode_image(luisa::span<const std::byte> bytes, ImageData &image,
                                luisa::string &error) noexcept;

/// Load an image from disk (UTF-8 path).
[[nodiscard]] bool load_image_file(std::string_view utf8_path, ImageData &image,
                                   luisa::string &error) noexcept;

/// Encode RGBA float pixels into an image file format (extension decides).
[[nodiscard]] bool encode_image(std::string_view extension, luisa::span<const float> rgba,
                                uint32_t width, uint32_t height,
                                luisa::vector<std::byte> &bytes,
                                luisa::string &error) noexcept;

/// Decode + encode convenience used to check save round trips in tests.
[[nodiscard]] bool decode_image_to_rgba(luisa::span<const std::byte> bytes,
                                        luisa::vector<float> &rgba,
                                        uint32_t &width, uint32_t &height,
                                        luisa::string &error) noexcept;

/// Save RGBA float pixels to disk (UTF-8 path, extension decides the format).
[[nodiscard]] bool save_image_file(std::string_view utf8_path, luisa::span<const float> rgba,
                                   uint32_t width, uint32_t height,
                                   luisa::string &error) noexcept;

/// Compare two RGBA float images.
[[nodiscard]] PixelDiffStats compare_pixels(luisa::span<const float> a,
                                            luisa::span<const float> b) noexcept;

// ---------------------------------------------------------------------------
// Headless self-test (defined in headless_test.cpp)
// ---------------------------------------------------------------------------

/// Runs the headless test suite: every writable image format is generated with
/// stb itself, loaded, processed on the device and saved again; operator chains
/// are validated against the CPU reference. Returns the number of failures.
[[nodiscard]] int run_headless_tests(Device &device, std::string_view output_directory) noexcept;

}// namespace image_process
