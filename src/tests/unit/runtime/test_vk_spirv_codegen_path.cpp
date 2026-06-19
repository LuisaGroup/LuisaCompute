#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <string_view>

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

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    if (argc <= 1 || std::string_view{argv[1]} != "vk") {
        LUISA_INFO("Usage: {} vk", argc > 0 ? argv[0] : "test_vk_spirv_codegen_path");
        return 0;
    }

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
    return 0;
}
