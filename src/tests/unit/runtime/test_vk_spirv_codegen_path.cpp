// Vulkan native XIR-to-SPIR-V code-generation path tests.
// This test covers JIT/AOT routing, shader identity, arithmetic edge cases,
// aggregate indexing, structured/autodiff callable inlining, and ray-instance metadata queries.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <string_view>
#include <vector>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void set_source_dump_env(const char *value) noexcept {
#ifdef _WIN32
    _putenv_s("LUISA_DUMP_SOURCE", value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv("LUISA_DUMP_SOURCE");
    } else {
        setenv("LUISA_DUMP_SOURCE", value, 1);
    }
#endif
}

[[nodiscard]] auto dump_exists(std::string_view name) noexcept {
    std::error_code ec;
    return std::filesystem::exists(std::filesystem::path{name}, ec);
}

[[nodiscard]] auto any_hlsl_dump_exists() {
    std::error_code ec;
    for (auto iter = std::filesystem::directory_iterator{".", ec};
         !ec && iter != std::filesystem::directory_iterator{}; iter.increment(ec)) {
        if (!iter->is_regular_file(ec)) { continue; }
        auto filename = iter->path().filename().string();
        if (filename.rfind("hlsl_output_", 0u) == 0u ||
            filename.rfind("spv_code_hlsl_", 0u) == 0u) {
            return true;
        }
    }
    return false;
}

void remove_hlsl_dumps() noexcept {
    std::error_code ec;
    for (auto iter = std::filesystem::directory_iterator{".", ec};
         !ec && iter != std::filesystem::directory_iterator{}; iter.increment(ec)) {
        if (!iter->is_regular_file(ec)) { continue; }
        auto filename = iter->path().filename().string();
        if (filename.rfind("hlsl_output_", 0u) == 0u ||
            filename.rfind("spv_code_hlsl_", 0u) == 0u) {
            std::filesystem::remove(iter->path(), ec);
        }
    }
}

void remove_dump(std::string_view name) noexcept {
    std::error_code ec;
    std::filesystem::remove(std::filesystem::path{name}, ec);
}

struct ScopedCurrentPath {
    std::filesystem::path previous;
    explicit ScopedCurrentPath(const std::filesystem::path &path)
        : previous{std::filesystem::current_path()} {
        std::filesystem::current_path(path);
    }
    ~ScopedCurrentPath() noexcept {
        std::error_code ec;
        std::filesystem::current_path(previous, ec);
    }
};

struct ScopedDirectoryCleanup {
    std::filesystem::path path;
    ~ScopedDirectoryCleanup() noexcept {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

struct ScopedSourceDump {
    std::optional<std::string> previous;
    ScopedSourceDump() {
        if (auto *value = std::getenv("LUISA_DUMP_SOURCE")) {
            previous.emplace(value);
        }
        set_source_dump_env("1");
    }
    ~ScopedSourceDump() noexcept {
        set_source_dump_env(previous ? previous->c_str() : nullptr);
    }
};

template<typename T, size_t N>
void expect_vector_equal(const Vector<T, N> &actual,
                         const Vector<T, N> &expected) noexcept {
    for (size_t i = 0u; i < N; i++) {
        expect(actual[i] == expected[i])
            << luisa::format("vector component {} mismatch", i);
    }
}

template<typename Scalar, typename Vector, bool test_log_exp = true>
void run_typed_float_constant_case(Device &device, double epsilon) {
    auto stream = device.create_stream();
    auto scalar_input = device.create_buffer<Scalar>(2u);
    auto vector_input = device.create_buffer<Vector>(2u);
    auto scalar_saturate_output = device.create_buffer<Scalar>(2u);
    auto vector_saturate_output = device.create_buffer<Vector>(2u);
    auto scalar_log_exp_output = device.create_buffer<Scalar>(2u);
    auto vector_log_exp_output = device.create_buffer<Vector>(2u);

    Kernel1D kernel = [](BufferVar<Scalar> scalar_in,
                         BufferVar<Vector> vector_in,
                         BufferVar<Scalar> scalar_saturate_out,
                         BufferVar<Vector> vector_saturate_out,
                         BufferVar<Scalar> scalar_log_exp_out,
                         BufferVar<Vector> vector_log_exp_out) noexcept {
        auto i = dispatch_x();
        auto scalar = scalar_in.read(i);
        auto vector = vector_in.read(i);
        scalar_saturate_out.write(i, saturate(scalar));
        vector_saturate_out.write(i, saturate(vector));
        if constexpr (test_log_exp) {
            auto quarter = cast<Scalar>(0.25f);
            scalar_log_exp_out.write(i, exp10(log10(abs(scalar) + quarter)));
            vector_log_exp_out.write(i, exp10(log10(abs(vector) + quarter)));
        } else {
            // Keep one kernel signature for f64 SATURATE coverage. Native SPIR-V
            // cannot legally emit GLSL.std.450 transcendental operations on f64.
            scalar_log_exp_out.write(i, scalar);
            vector_log_exp_out.write(i, vector);
        }
    };
    ShaderOption option{.enable_fast_math = false};
    auto shader = device.compile(kernel, option);

    std::array scalar_source{
        static_cast<Scalar>(-0.5),
        static_cast<Scalar>(1.5)};
    std::array vector_source{
        Vector{static_cast<Scalar>(-0.5), static_cast<Scalar>(0.25)},
        Vector{static_cast<Scalar>(1.5), static_cast<Scalar>(-2.0)}};
    std::array<Scalar, 2u> scalar_saturate_result{};
    std::array<Vector, 2u> vector_saturate_result{};
    std::array<Scalar, 2u> scalar_log_exp_result{};
    std::array<Vector, 2u> vector_log_exp_result{};
    stream << scalar_input.copy_from(luisa::span{scalar_source})
           << vector_input.copy_from(luisa::span{vector_source})
           << shader(scalar_input, vector_input,
                     scalar_saturate_output, vector_saturate_output,
                     scalar_log_exp_output, vector_log_exp_output)
                  .dispatch(2u)
           << scalar_saturate_output.copy_to(luisa::span{scalar_saturate_result})
           << vector_saturate_output.copy_to(luisa::span{vector_saturate_result})
           << scalar_log_exp_output.copy_to(luisa::span{scalar_log_exp_result})
           << vector_log_exp_output.copy_to(luisa::span{vector_log_exp_result})
           << synchronize();

    auto close = [epsilon](auto actual, double expected) noexcept {
        return std::abs(static_cast<double>(actual) - expected) <= epsilon;
    };
    for (auto i = 0u; i < scalar_source.size(); i++) {
        auto scalar_value = static_cast<double>(scalar_source[i]);
        auto scalar_saturate_expected = scalar_value < 0.0 ? 0.0 : std::min(scalar_value, 1.0);
        expect(close(scalar_saturate_result[i], scalar_saturate_expected));
        if constexpr (test_log_exp) {
            auto scalar_log_exp_expected = std::abs(scalar_value) + 0.25;
            expect(close(scalar_log_exp_result[i], scalar_log_exp_expected));
        }
        for (auto j = 0u; j < 2u; j++) {
            auto vector_value = static_cast<double>(vector_source[i][j]);
            auto vector_saturate_expected = vector_value < 0.0 ? 0.0 : std::min(vector_value, 1.0);
            expect(close(vector_saturate_result[i][j], vector_saturate_expected));
            if constexpr (test_log_exp) {
                auto vector_log_exp_expected = std::abs(vector_value) + 0.25;
                expect(close(vector_log_exp_result[i][j], vector_log_exp_expected));
            }
        }
    }
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc <= 1 || std::string_view{argv[1]} != "vk") {
        LUISA_INFO("Usage: {} vk", argc > 0 ? argv[0] : "test_vk_spirv_codegen_path");
        return 2;
    }
    std::vector<const char *> ut_argv;
    ut_argv.reserve(static_cast<size_t>(argc));
    ut_argv.emplace_back(argv[0]);
    for (auto i = 2; i < argc; i++) { ut_argv.emplace_back(argv[i]); }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        static_cast<int>(ut_argv.size()), ut_argv.data());

    auto executable_path = std::filesystem::absolute(argv[0]).string();
    argv[0] = executable_path.data();
    auto process_work_dir = std::filesystem::temp_directory_path() /
                            luisa::format("luisa_vk_spirv_codegen_path_process_{}",
                                          std::filesystem::path{argv[0]}.filename().string());
    std::error_code process_work_dir_ec;
    std::filesystem::remove_all(process_work_dir, process_work_dir_ec);
    std::filesystem::create_directories(process_work_dir);
    ScopedDirectoryCleanup process_work_dir_cleanup{process_work_dir};
    ScopedCurrentPath process_work_path{process_work_dir};

    "vk_user_compute_dumps_spirv_not_hlsl"_test = [&] {
        constexpr std::string_view hlsl_dump = "hlsl_output_vk_spirv_codegen_path.hlsl";
        constexpr std::string_view spv_dump = "spv_code_vk_spirv_codegen_path.spvasm";

        auto dc = luisa::test::create_device(argc, argv);
        auto dump_dir = std::filesystem::temp_directory_path() /
                        luisa::format("luisa_vk_spirv_codegen_path_{}", std::filesystem::path{argv[0]}.filename().string());
        std::error_code ec;
        std::filesystem::remove_all(dump_dir, ec);
        std::filesystem::create_directories(dump_dir);
        ScopedCurrentPath scoped_path{dump_dir};
        ScopedSourceDump scoped_source_dump;
        remove_hlsl_dumps();
        remove_dump(hlsl_dump);
        remove_dump(spv_dump);

        auto buffer = dc.device.create_buffer<uint32_t>(1u);
        auto stream = dc.device.create_stream();

        Kernel1D kernel = [](BufferUInt output) noexcept {
            output.write(0u, 42u);
        };
        ShaderOption option{.name = "vk_spirv_codegen_path"};
        auto shader = dc.device.compile(kernel, option);

        uint32_t value = 0u;
        stream << shader(buffer).dispatch(1u)
               << buffer.copy_to(luisa::span{&value, 1u})
               << synchronize();
        expect(value == 42u);

        expect(!dump_exists(hlsl_dump)) << "Vulkan user compute must not dump HLSL";
        expect(!any_hlsl_dump_exists()) << "Vulkan user compute must not emit any HLSL-derived dumps";
        expect(dump_exists(spv_dump)) << "Vulkan user compute should dump native SPIR-V when LUISA_DUMP_SOURCE=1";
    };

    "vk_user_compute_inlines_structured_callable_after_cfg_destructure"_test = [&] {
        constexpr std::string_view hlsl_dump = "hlsl_output_vk_spirv_structured_callable.hlsl";
        constexpr std::string_view spv_dump = "spv_code_vk_spirv_structured_callable.spvasm";

        auto dc = luisa::test::create_device(argc, argv);
        auto dump_dir = std::filesystem::temp_directory_path() /
                        luisa::format("luisa_vk_spirv_structured_callable_{}",
                                      std::filesystem::path{argv[0]}.filename().string());
        std::error_code ec;
        std::filesystem::remove_all(dump_dir, ec);
        std::filesystem::create_directories(dump_dir);
        ScopedCurrentPath scoped_path{dump_dir};
        ScopedSourceDump scoped_source_dump;
        remove_hlsl_dumps();
        remove_dump(hlsl_dump);
        remove_dump(spv_dump);

        auto output = dc.device.create_buffer<uint32_t>(4u);
        auto stream = dc.device.create_stream();
        Callable classify = [](UInt value) noexcept {
            UInt result;
            $if ((value & 1u) == 0u) {
                result = value * 3u + 1u;
            }
            $else {
                result = value + 7u;
            };
            return result;
        };
        Kernel1D kernel = [&](BufferUInt out) noexcept {
            auto i = dispatch_x();
            out.write(i, classify(i));
        };
        auto normalized_xir_dump = luisa::format(
            "kernel.{:016x}.norm.xir", kernel.function()->function().hash());
        ShaderOption option{.name = "vk_spirv_structured_callable"};
        auto shader = dc.device.compile(kernel, option);

        std::array<uint32_t, 4u> result{};
        stream << shader(output).dispatch(4u)
               << output.copy_to(luisa::span{result})
               << synchronize();
        constexpr std::array expected{1u, 8u, 7u, 10u};
        expect(result == expected)
            << "both branches of the structured callable should execute deterministically";

        expect(!dump_exists(hlsl_dump)) << "Vulkan structured callable must not dump HLSL";
        expect(!any_hlsl_dump_exists()) << "Vulkan structured callable must stay on native XIR-to-SPIR-V";
        expect(dump_exists(spv_dump)) << "Vulkan structured callable should dump native SPIR-V";
        expect(dump_exists(normalized_xir_dump)) << "Vulkan structured callable should dump normalized XIR";
        std::ifstream xir_stream{normalized_xir_dump.c_str()};
        auto normalized_xir = std::string{
            std::istreambuf_iterator<char>{xir_stream},
            std::istreambuf_iterator<char>{}};
        expect(normalized_xir.find("callable ") == std::string::npos)
            << "structured callable should be inlined after CFG destructuring";
    };

    "vk_user_compute_autodiff_inlines_multiblock_callable_after_cfg_destructure"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto dump_dir = std::filesystem::temp_directory_path() /
                        luisa::format("luisa_vk_spirv_autodiff_callable_{}",
                                      std::filesystem::path{argv[0]}.filename().string());
        std::error_code ec;
        std::filesystem::remove_all(dump_dir, ec);
        std::filesystem::create_directories(dump_dir);
        ScopedCurrentPath scoped_path{dump_dir};
        ScopedSourceDump scoped_source_dump;

        auto input = dc.device.create_buffer<float>(6u);
        auto selector = dc.device.create_buffer<uint32_t>(6u);
        auto output = dc.device.create_buffer<float>(6u);
        auto stream = dc.device.create_stream();
        Callable differentiated = [](Float x, UInt branch) noexcept {
            auto y = def(0.0f);
            $if (x > 0.0f) {
                y = x * x;
            }
            $else {
                y = x * x * x;
            };
            $switch (branch) {
                $case (0u) {
                    y = y + x;
                };
                $case (1u) {
                    y = y * 2.0f;
                };
                $default {
                    y = y - 3.0f * x;
                };
            };
            return y;
        };
        Kernel1D kernel = [&](BufferFloat in, BufferUInt branches, BufferFloat out) noexcept {
            auto i = dispatch_x();
            auto x = in.read(i);
            $autodiff {
                requires_grad(x);
                auto y = differentiated(x, branches.read(i));
                backward(y);
                out.write(i, grad(x));
            };
        };
        auto normalized_xir_dump = luisa::format(
            "kernel.{:016x}.norm.xir", kernel.function()->function().hash());
        ShaderOption option{.enable_fast_math = false,
                            .name = "vk_spirv_autodiff_callable"};
        auto shader = dc.device.compile(kernel, option);

        constexpr std::array input_values{-2.0f, -1.5f, -1.0f, 0.5f, 1.0f, 2.0f};
        constexpr std::array selector_values{0u, 1u, 2u, 0u, 1u, 2u};
        constexpr std::array expected{13.0f, 13.5f, 0.0f, 2.0f, 4.0f, 1.0f};
        std::array<float, 6u> result{};
        stream << input.copy_from(luisa::span{input_values})
               << selector.copy_from(luisa::span{selector_values})
               << shader(input, selector, output).dispatch(6u)
               << output.copy_to(luisa::span{result})
               << synchronize();

        auto gradients_match = true;
        for (auto i = 0u; i < result.size(); i++) {
            gradients_match &= std::isfinite(result[i]) &&
                               std::abs(result[i] - expected[i]) < 1e-4f;
        }
        expect(gradients_match)
            << "autodiff should preserve the selected if/switch derivative after callable inlining";
        expect(dump_exists(normalized_xir_dump))
            << "Vulkan autodiff callable should dump normalized XIR";
        std::ifstream xir_stream{normalized_xir_dump.c_str()};
        auto normalized_xir = std::string{
            std::istreambuf_iterator<char>{xir_stream},
            std::istreambuf_iterator<char>{}};
        expect(normalized_xir.find("callable ") == std::string::npos)
            << "autodiff callable should be inlined after CFG destructuring";
        expect(normalized_xir.find("autodiff_scope") == std::string::npos)
            << "autodiff scope should be lowered before SPIR-V emission";
    };

    "vk_user_compute_aot_uses_spirv_not_hlsl"_test = [&] {
        constexpr std::string_view hlsl_dump = "hlsl_output_vk_spirv_codegen_path_aot.hlsl";
        constexpr std::string_view spv_dump = "spv_code_vk_spirv_codegen_path_aot.spvasm";

        auto dc = luisa::test::create_device(argc, argv);
        auto dump_dir = std::filesystem::temp_directory_path() /
                        luisa::format("luisa_vk_spirv_codegen_path_aot_{}", std::filesystem::path{argv[0]}.filename().string());
        std::error_code ec;
        std::filesystem::remove_all(dump_dir, ec);
        std::filesystem::create_directories(dump_dir);
        ScopedCurrentPath scoped_path{dump_dir};
        ScopedSourceDump scoped_source_dump;
        remove_hlsl_dumps();
        remove_dump(hlsl_dump);
        remove_dump(spv_dump);

        constexpr std::string_view shader_path = "vk_spirv_codegen_path_aot";
        Kernel1D kernel = [](BufferUInt output) noexcept {
            output.write(0u, 7u);
        };
        dc.device.compile_to(kernel, shader_path);

        auto buffer = dc.device.create_buffer<uint32_t>(1u);
        auto stream = dc.device.create_stream();
        auto shader = dc.device.load_shader<1, Buffer<uint32_t>>(shader_path);

        uint32_t value = 0u;
        stream << shader(buffer).dispatch(1u)
               << buffer.copy_to(luisa::span{&value, 1u})
               << synchronize();
        expect(value == 7u);

        expect(!dump_exists(hlsl_dump)) << "Vulkan AOT user compute must not dump HLSL";
        expect(!any_hlsl_dump_exists()) << "Vulkan AOT user compute must not emit any HLSL-derived dumps";
        expect(dump_exists(spv_dump)) << "Vulkan compile_to should dump native SPIR-V when LUISA_DUMP_SOURCE=1";
    };

    "vk_user_compute_same_shape_jit_shaders_do_not_alias"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto buffer = device.create_buffer<uint32_t>(512u);

        Kernel1D first = [](BufferUInt out) noexcept {
            auto i = dispatch_x();
            out.write(i, i + 1u);
        };
        Kernel1D second = [](BufferUInt out) noexcept {
            auto i = dispatch_x();
            out.write(i, (i + 1u) * 3u);
        };

        auto shader_a = device.compile(first);
        stream << shader_a(buffer).dispatch(512u) << synchronize();

        auto shader_b = device.compile(second);
        stream << shader_b(buffer).dispatch(512u) << synchronize();

        luisa::vector<uint32_t> host(512u);
        stream << buffer.copy_to(luisa::span{host}) << synchronize();
        auto ok = true;
        for (auto i = 0u; i < host.size(); i++) {
            auto expected = static_cast<uint32_t>((i + 1u) * 3u);
            if (host[i] != expected) {
                LUISA_WARNING("same-shape JIT shader alias mismatch at {}: got {}, expected {}",
                              i, host[i], expected);
                ok = false;
                break;
            }
        }
        expect(ok) << "Vulkan JIT compute shaders with the same default identity must not reuse stale pipelines";
    };

    "vk_user_compute_vector_rounds_half_away_from_zero"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto input = device.create_buffer<float4>(2u);
        auto output = device.create_buffer<float4>(2u);
        Kernel1D kernel = [](BufferFloat4 in, BufferFloat4 out) noexcept {
            auto i = dispatch_x();
            out.write(i, round(in.read(i)));
        };
        ShaderOption option{.enable_fast_math = false};
        auto shader = device.compile(kernel, option);
        std::array source{
            float4{-2.5f, -0.5f, 0.5f, 3.5f},
            float4{-0.0f, 0.0f, 1.25f, -1.25f}};
        std::array<float4, 2u> result{};
        stream << input.copy_from(luisa::span{source})
               << shader(input, output).dispatch(2u)
               << output.copy_to(luisa::span{result})
               << synchronize();
        expect(result[0][0] == -3.0f);
        expect(result[0][1] == -1.0f);
        expect(result[0][2] == 1.0f);
        expect(result[0][3] == 4.0f);
        expect(std::signbit(result[1][0]));
        expect(!std::signbit(result[1][1]));
        expect(result[1][2] == 1.0f);
        expect(result[1][3] == -1.0f);
    };

    "vk_user_compute_float_to_bool_treats_nan_as_nonzero"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto input = device.create_buffer<float>(4u);
        auto output = device.create_buffer<uint32_t>(4u);
        Kernel1D kernel = [](BufferFloat in, BufferUInt out) noexcept {
            auto i = dispatch_x();
            out.write(i, ite(cast<bool>(in.read(i)), 1u, 0u));
        };
        ShaderOption option{.enable_fast_math = false};
        auto shader = device.compile(kernel, option);
        std::array source{
            std::numeric_limits<float>::quiet_NaN(),
            0.0f,
            -0.0f,
            -2.0f};
        std::array<uint32_t, 4u> result{};
        stream << input.copy_from(luisa::span{source})
               << shader(input, output).dispatch(4u)
               << output.copy_to(luisa::span{result})
               << synchronize();
        expect(result[0] == 1u);
        expect(result[1] == 0u);
        expect(result[2] == 0u);
        expect(result[3] == 1u);
    };

    "vk_user_compute_typed_floating_constants"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        run_typed_float_constant_case<half, half2>(device, 5e-2);
        run_typed_float_constant_case<float, float2>(device, 2e-4);
        run_typed_float_constant_case<double, double2, false>(device, 1e-10);
    };

    "vk_user_compute_integer_vectors_and_wide_constant_indices"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto integer_input = device.create_buffer<uint2>(2u);
        auto clz_output = device.create_buffer<uint2>(2u);
        auto ctz_output = device.create_buffer<uint2>(2u);
        auto vector_input = device.create_buffer<float4>(1u);
        auto aggregate_output = device.create_buffer<float2>(1u);
        auto extract_output = device.create_buffer<float>(1u);
        auto shuffle_output = device.create_buffer<float2>(1u);
        auto matrix_input = device.create_buffer<float4x4>(1u);
        auto dynamic_index_input = device.create_buffer<short>(1u);
        auto dynamic_extract_output = device.create_buffer<float4>(1u);
        Kernel1D kernel = [](BufferUInt2 integers,
                             BufferUInt2 clz_out,
                             BufferUInt2 ctz_out,
                             BufferFloat4 vectors,
                             BufferFloat2 aggregate_out,
                             BufferFloat extract_out,
                             BufferFloat2 shuffle_out,
                             BufferFloat4x4 matrices,
                             BufferShort dynamic_indices,
                             BufferFloat4 dynamic_extract_out) noexcept {
            auto i = dispatch_x();
            $if (i < 2u) {
                auto value = integers.read(i);
                clz_out.write(i, clz(value));
                ctz_out.write(i, ctz(value));
            };
            $if (i == 0u) {
                auto value = vectors.read(0u);
                aggregate_out.write(
                    0u, make_float2(
                            value[static_cast<int16_t>(1)],
                            value[static_cast<uint64_t>(3)]));
                extract_out.write(0u, value[static_cast<int8_t>(2)]);
                shuffle_out.write(0u, value.wy());
                dynamic_extract_out.write(
                    0u, matrices.read(0u)[dynamic_indices.read(0u)]);
            };
        };
        auto shader = device.compile(kernel);
        std::array integer_source{
            uint2{0u, 1u},
            uint2{0x80000000u, 0x10u}};
        std::array vector_source{float4{10.0f, 20.0f, 30.0f, 40.0f}};
        std::array matrix_source{make_float4x4(
            float4{1.0f, 2.0f, 3.0f, 4.0f},
            float4{5.0f, 6.0f, 7.0f, 8.0f},
            float4{9.0f, 10.0f, 11.0f, 12.0f},
            float4{13.0f, 14.0f, 15.0f, 16.0f})};
        std::array dynamic_index_source{static_cast<short>(2)};
        std::array<uint2, 2u> clz_result{};
        std::array<uint2, 2u> ctz_result{};
        std::array<float2, 1u> aggregate_result{};
        std::array<float, 1u> extract_result{};
        std::array<float2, 1u> shuffle_result{};
        std::array<float4, 1u> dynamic_extract_result{};
        stream << integer_input.copy_from(luisa::span{integer_source})
               << vector_input.copy_from(luisa::span{vector_source})
               << matrix_input.copy_from(luisa::span{matrix_source})
               << dynamic_index_input.copy_from(luisa::span{dynamic_index_source})
               << shader(integer_input, clz_output, ctz_output,
                         vector_input, aggregate_output, extract_output,
                         shuffle_output, matrix_input, dynamic_index_input,
                         dynamic_extract_output)
                      .dispatch(2u)
               << clz_output.copy_to(luisa::span{clz_result})
               << ctz_output.copy_to(luisa::span{ctz_result})
               << aggregate_output.copy_to(luisa::span{aggregate_result})
               << extract_output.copy_to(luisa::span{extract_result})
               << shuffle_output.copy_to(luisa::span{shuffle_result})
               << dynamic_extract_output.copy_to(luisa::span{dynamic_extract_result})
               << synchronize();
        expect_vector_equal(clz_result[0], uint2{32u, 31u});
        expect_vector_equal(clz_result[1], uint2{0u, 27u});
        expect_vector_equal(ctz_result[0], uint2{32u, 0u});
        expect_vector_equal(ctz_result[1], uint2{31u, 4u});
        expect_vector_equal(aggregate_result[0], float2{20.0f, 40.0f});
        expect(extract_result[0] == 30.0f);
        expect_vector_equal(shuffle_result[0], float2{40.0f, 20.0f});
        expect_vector_equal(dynamic_extract_result[0], float4{9.0f, 10.0f, 11.0f, 12.0f});
    };

    "vk_user_compute_dynamic_insert_preserves_source"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto input = device.create_buffer<float4>(1u);
        auto dynamic_index = device.create_buffer<short>(1u);
        auto original_output = device.create_buffer<float4>(1u);
        auto inserted_output = device.create_buffer<float4>(1u);

        Callable make_inserted = [](Float4 value, Short index) noexcept {
            value[index] = 99.0f;
            return value;
        };
        Kernel1D kernel = [&](BufferFloat4 source,
                              BufferShort indices,
                              BufferFloat4 original_out,
                              BufferFloat4 inserted_out) noexcept {
            auto original = source.read(0u);
            auto inserted = make_inserted(original, indices.read(0u));
            original_out.write(0u, original);
            inserted_out.write(0u, inserted);
        };
        auto shader = device.compile(kernel);

        std::array source{float4{1.0f, 2.0f, 3.0f, 4.0f}};
        std::array index{static_cast<short>(2)};
        std::array<float4, 1u> original_result{};
        std::array<float4, 1u> inserted_result{};
        stream << input.copy_from(luisa::span{source})
               << dynamic_index.copy_from(luisa::span{index})
               << shader(input, dynamic_index, original_output, inserted_output).dispatch(1u)
               << original_output.copy_to(luisa::span{original_result})
               << inserted_output.copy_to(luisa::span{inserted_result})
               << synchronize();
        expect_vector_equal(original_result[0], float4{1.0f, 2.0f, 3.0f, 4.0f});
        expect_vector_equal(inserted_result[0], float4{1.0f, 2.0f, 99.0f, 4.0f});
    };

    "vk_user_compute_float_edge_semantics_and_scalar_broadcasts"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();

        auto scalar_lhs = device.create_buffer<float>(4u);
        auto scalar_rhs = device.create_buffer<float>(4u);
        auto scalar_copysign_output = device.create_buffer<float>(4u);
        auto scalar_not_equal_output = device.create_buffer<uint32_t>(4u);
        auto vector_lhs = device.create_buffer<float4>(1u);
        auto vector_rhs = device.create_buffer<float4>(1u);
        auto vector_copysign_output = device.create_buffer<float4>(1u);
        auto vector_not_equal_output = device.create_buffer<uint4>(1u);
        Kernel1D edge_kernel = [](BufferFloat lhs,
                                  BufferFloat rhs,
                                  BufferFloat copysign_out,
                                  BufferUInt not_equal_out,
                                  BufferFloat4 vector_lhs_buffer,
                                  BufferFloat4 vector_rhs_buffer,
                                  BufferFloat4 vector_copysign_out,
                                  BufferUInt4 vector_not_equal_out) noexcept {
            auto i = dispatch_x();
            auto a = lhs.read(i);
            auto b = rhs.read(i);
            copysign_out.write(i, copysign(a, b));
            not_equal_out.write(i, ite(a != b, 1u, 0u));
            $if (i == 0u) {
                auto va = vector_lhs_buffer.read(0u);
                auto vb = vector_rhs_buffer.read(0u);
                vector_copysign_out.write(0u, copysign(va, vb));
                vector_not_equal_out.write(
                    0u, select(make_uint4(0u), make_uint4(1u), va != vb));
            };
        };
        ShaderOption option{.enable_fast_math = false};
        auto edge_shader = device.compile(edge_kernel, option);
        auto nan = std::numeric_limits<float>::quiet_NaN();
        std::array scalar_lhs_source{2.0f, -3.0f, 0.0f, -0.0f};
        std::array scalar_rhs_source{0.0f, -0.0f, -0.0f, 0.0f};
        std::array vector_lhs_source{float4{nan, 0.0f, 1.0f, nan}};
        std::array vector_rhs_source{float4{nan, -0.0f, 2.0f, 4.0f}};
        std::array<float, 4u> scalar_copysign_result{};
        std::array<uint32_t, 4u> scalar_not_equal_result{};
        std::array<float4, 1u> vector_copysign_result{};
        std::array<uint4, 1u> vector_not_equal_result{};
        stream << scalar_lhs.copy_from(luisa::span{scalar_lhs_source})
               << scalar_rhs.copy_from(luisa::span{scalar_rhs_source})
               << vector_lhs.copy_from(luisa::span{vector_lhs_source})
               << vector_rhs.copy_from(luisa::span{vector_rhs_source})
               << edge_shader(
                      scalar_lhs, scalar_rhs,
                      scalar_copysign_output, scalar_not_equal_output,
                      vector_lhs, vector_rhs,
                      vector_copysign_output, vector_not_equal_output)
                      .dispatch(4u)
               << scalar_copysign_output.copy_to(luisa::span{scalar_copysign_result})
               << scalar_not_equal_output.copy_to(luisa::span{scalar_not_equal_result})
               << vector_copysign_output.copy_to(luisa::span{vector_copysign_result})
               << vector_not_equal_output.copy_to(luisa::span{vector_not_equal_result})
               << synchronize();
        for (auto i = 0u; i < scalar_lhs_source.size(); i++) {
            auto expected = std::copysign(scalar_lhs_source[i], scalar_rhs_source[i]);
            expect(std::bit_cast<uint32_t>(scalar_copysign_result[i]) ==
                   std::bit_cast<uint32_t>(expected));
            expect(scalar_not_equal_result[i] ==
                   static_cast<uint32_t>(scalar_lhs_source[i] != scalar_rhs_source[i]));
        }
        for (auto i = 0u; i < 4u; i++) {
            auto expected = std::copysign(vector_lhs_source[0][i], vector_rhs_source[0][i]);
            expect(std::bit_cast<uint32_t>(vector_copysign_result[0][i]) ==
                   std::bit_cast<uint32_t>(expected));
            expect(vector_not_equal_result[0][i] ==
                   static_cast<uint32_t>(vector_lhs_source[0][i] != vector_rhs_source[0][i]));
        }

        auto lower = device.create_buffer<float2>(1u);
        auto upper = device.create_buffer<float2>(1u);
        auto edge0 = device.create_buffer<float>(1u);
        auto edge1 = device.create_buffer<float>(1u);
        auto lerp_output = device.create_buffer<float2>(1u);
        auto step_output = device.create_buffer<float2>(1u);
        auto smoothstep_output = device.create_buffer<float2>(1u);
        Kernel1D broadcast_kernel = [](BufferFloat2 lower_buffer,
                                       BufferFloat2 upper_buffer,
                                       BufferFloat edge0_buffer,
                                       BufferFloat edge1_buffer,
                                       BufferFloat2 lerp_out,
                                       BufferFloat2 step_out,
                                       BufferFloat2 smoothstep_out) noexcept {
            auto lower_value = lower_buffer.read(0u);
            auto upper_value = upper_buffer.read(0u);
            auto edge0_value = edge0_buffer.read(0u);
            auto edge1_value = edge1_buffer.read(0u);
            auto builder = luisa::compute::detail::FunctionBuilder::current();
            auto lerp_value = def<float2>(builder->call(
                Type::of<float2>(), CallOp::LERP,
                {lower_value.expression(), upper_value.expression(), edge0_value.expression()}));
            auto step_value = def<float2>(builder->call(
                Type::of<float2>(), CallOp::STEP,
                {edge0_value.expression(), lower_value.expression()}));
            auto smoothstep_value = def<float2>(builder->call(
                Type::of<float2>(), CallOp::SMOOTHSTEP,
                {edge0_value.expression(), edge1_value.expression(), lower_value.expression()}));
            lerp_out.write(0u, lerp_value);
            step_out.write(0u, step_value);
            smoothstep_out.write(0u, smoothstep_value);
        };
        auto broadcast_shader = device.compile(broadcast_kernel, option);
        std::array lower_source{float2{0.1f, 0.5f}};
        std::array upper_source{float2{2.1f, 2.5f}};
        std::array edge0_source{0.25f};
        std::array edge1_source{0.75f};
        std::array<float2, 1u> lerp_result{};
        std::array<float2, 1u> step_result{};
        std::array<float2, 1u> smoothstep_result{};
        stream << lower.copy_from(luisa::span{lower_source})
               << upper.copy_from(luisa::span{upper_source})
               << edge0.copy_from(luisa::span{edge0_source})
               << edge1.copy_from(luisa::span{edge1_source})
               << broadcast_shader(lower, upper, edge0, edge1,
                                   lerp_output, step_output, smoothstep_output)
                      .dispatch(1u)
               << lerp_output.copy_to(luisa::span{lerp_result})
               << step_output.copy_to(luisa::span{step_result})
               << smoothstep_output.copy_to(luisa::span{smoothstep_result})
               << synchronize();
        expect(std::abs(lerp_result[0][0] - 0.6f) < 1e-5f);
        expect(std::abs(lerp_result[0][1] - 1.0f) < 1e-5f);
        expect_vector_equal(step_result[0], float2{0.0f, 1.0f});
        expect(std::abs(smoothstep_result[0][0] - 0.0f) < 1e-5f);
        expect(std::abs(smoothstep_result[0][1] - 0.5f) < 1e-5f);
    };

    "vk_user_compute_ray_instance_metadata_queries"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();

        std::array vertices{
            float3{-0.5f, -0.5f, 0.0f},
            float3{0.5f, -0.5f, 0.0f},
            float3{0.0f, 0.5f, 0.0f}};
        std::array indices{0u, 1u, 2u};
        auto vertex_buffer = device.create_buffer<float3>(vertices.size());
        auto triangle_buffer = device.create_buffer<Triangle>(1u);
        auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
        auto accel = device.create_accel();
        constexpr auto expected_visibility = static_cast<uint8_t>(0x5au);
        constexpr auto expected_user_id = 0x00c0ffeeu;
        accel.emplace_back(mesh, make_float4x4(1.0f),
                           expected_visibility, true, expected_user_id);

        Kernel1D query_kernel = [](AccelVar accel_var, BufferUInt output) noexcept {
            output.write(0u, accel_var.instance_user_id(0));
            output.write(1u, accel_var.instance_visibility_mask(0u));
        };
        Kernel1D update_kernel = [](AccelVar accel_var) noexcept {
            accel_var.set_instance_user_id(0u, 0x000badc0u);
            accel_var.set_instance_visibility(0, 0xa5u);
        };
        auto shader = device.compile(query_kernel);
        auto update_shader = device.compile(update_kernel);
        auto output = device.create_buffer<uint32_t>(2u);
        std::array<uint32_t, 2u> result{};
        stream << vertex_buffer.copy_from(luisa::span{vertices})
               << triangle_buffer.copy_from(luisa::span{indices})
               << mesh.build()
               << accel.build()
               << shader(accel, output).dispatch(1u)
               << output.copy_to(luisa::span{result})
               << synchronize();

        expect(result[0] == expected_user_id);
        expect(result[1] == expected_visibility);

        result.fill(0u);
        stream << update_shader(accel).dispatch(1u)
               << shader(accel, output).dispatch(1u)
               << output.copy_to(luisa::span{result})
               << synchronize();
        expect(result[0] == 0x000badc0u)
            << luisa::format("updated user ID mismatch: got 0x{:08x}", result[0]);
        expect(result[1] == 0xa5u)
            << luisa::format("updated visibility mismatch: got 0x{:08x}", result[1]);
    };
    return 0;
}
