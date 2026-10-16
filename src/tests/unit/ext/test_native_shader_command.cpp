// Unit tests for NativeShaderDispatchCommand / NativeShaderLauncher
// (include/luisa/backends/ext/native_shader_ext.h). No device required.
//
// Covered risks:
//  * R6  - the per-argument Usage contract is surfaced and cross-checked
//          against the reflected binding class.
//  * R9  - binding-aware resolution + the canonical reflection order.
//  * R17 - max_dispatch_size() reports the exact thread count.
//  * R22 - null handles / missing / duplicate bindings are rejected.
#include "ut/ut.hpp"
#include <luisa/backends/ext/native_shader_ext.h>
#include <luisa/backends/ext/registry.h>
#include <cstring>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto test_handle_a = 0x1000ull;
constexpr auto test_handle_b = 0x2000ull;
constexpr auto test_shader = 0xfeedull;

[[nodiscard]] NativeShaderResourceBinding make_binding(
    uint32_t space, uint32_t reg, NativeShaderResourceKind kind) noexcept {
    NativeShaderResourceBinding b;
    b.kind = kind;
    b.space_index = space;
    b.register_index = reg;
    b.usage = native_shader_default_usage(kind);
    return b;
}

// SRV at (t0, space0), texture at (t1, space0), a constant buffer at (b2,
// space0) and a UAV at (u3, space0) - i.e. distinct (space, register) pairs, as
// the SPIR-V route requires.
[[nodiscard]] luisa::vector<NativeShaderResourceBinding> make_bindings() noexcept {
    luisa::vector<NativeShaderResourceBinding> bindings;
    bindings.emplace_back(make_binding(0u, 0u, NativeShaderResourceKind::StructuredBuffer));
    bindings.emplace_back(make_binding(0u, 1u, NativeShaderResourceKind::Texture2D));
    bindings.emplace_back(make_binding(0u, 2u, NativeShaderResourceKind::ConstantBuffer));
    bindings.emplace_back(make_binding(0u, 3u, NativeShaderResourceKind::RWStructuredBuffer));
    return bindings;
}

}// namespace

int main() {

    "uuid_and_command_metadata"_test = [] {
        NativeShaderLauncher launcher{test_shader, uint3{64u, 1u, 1u}, make_bindings()};
        launcher.add_buffer(0u, 0u, test_handle_a, 0u, 256u, Usage::READ)
            .add_buffer(3u, 0u, test_handle_b, 0u, 256u, Usage::WRITE)
            .add_texture(1u, 0u, 0x3000u, 0u, Usage::READ)
            .add_buffer(2u, 0u, 0x4000u, 0u, 64u, Usage::READ);
        expect(launcher.validate().empty());
        auto cmd = std::move(launcher).build(uint3{512u, 1u, 1u});
        expect(cmd->custom_cmd_uuid() ==
               luisa::to_underlying(CustomCommandUUID::NATIVE_SHADER_DISPATCH));
        expect(luisa::to_underlying(CustomCommandUUID::NATIVE_SHADER_DISPATCH) == 0x0600u);
        expect(to_string(CustomCommandUUID::NATIVE_SHADER_DISPATCH) ==
               luisa::string_view{"NATIVE_SHADER_DISPATCH"});
        expect(cmd->stream_tag() == StreamTag::COMPUTE);
        expect(cmd->shader_handle() == test_shader);
        expect(cmd->dispatch_size().x == 512u && cmd->dispatch_size().y == 1u &&
               cmd->dispatch_size().z == 1u);
        expect(cmd->block_size().x == 64u && cmd->block_size().y == 1u &&
               cmd->block_size().z == 1u);
        // R17: the reorder budget counts exactly this many threads.
        expect(cmd->max_dispatch_size().x == 512u);
        expect(cmd->argument_usages().size() == 4u);
        expect(cmd->arguments().size() == 4u);
    };

    "canonical_argument_order"_test = [] {
        // Bindings are supplied out of order; the command must carry them in the
        // canonical reflection order (space, then register).
        auto bindings = luisa::vector<NativeShaderResourceBinding>{};
        bindings.emplace_back(make_binding(1u, 0u, NativeShaderResourceKind::StructuredBuffer));
        bindings.emplace_back(make_binding(0u, 1u, NativeShaderResourceKind::RWStructuredBuffer));
        bindings.emplace_back(make_binding(0u, 0u, NativeShaderResourceKind::StructuredBuffer));
        expect(bindings.size() == 3u);
        NativeShaderLauncher launcher{test_shader, uint3{1u, 1u, 1u}, bindings};
        // (register, space) pairs: (0,1), (1,0) and (0,0) - supplied out of order.
        launcher.add_buffer(0u, 1u, test_handle_a, 0u, 16u, Usage::READ)
            .add_buffer(1u, 0u, test_handle_b, 0u, 16u, Usage::READ_WRITE)
            .add_buffer(0u, 0u, 0x5000u, 0u, 16u, Usage::READ);
        auto plan = launcher.plan();
        expect(plan.ok()) << plan.error.c_str();
        expect(plan.arguments.size() == 3u);
        expect(plan.arguments[0].buffer.handle == 0x5000u);// (0,0)
        expect(plan.arguments[1].buffer.handle == test_handle_b);// (0,1)
        expect(plan.arguments[2].buffer.handle == test_handle_a);// (1,0)
        expect(luisa::to_underlying(plan.usages[0]) == luisa::to_underlying(Usage::READ));
        expect(luisa::to_underlying(plan.usages[1]) == luisa::to_underlying(Usage::READ_WRITE));
        expect(luisa::to_underlying(plan.usages[2]) == luisa::to_underlying(Usage::READ));
    };

    "positional_arguments_follow_canonical_order"_test = [] {
        auto bindings = luisa::vector<NativeShaderResourceBinding>{};
        bindings.emplace_back(make_binding(0u, 3u, NativeShaderResourceKind::StructuredBuffer));
        bindings.emplace_back(make_binding(0u, 1u, NativeShaderResourceKind::StructuredBuffer));
        NativeShaderLauncher launcher{test_shader, uint3{1u, 1u, 1u}, bindings};
        launcher.add_buffer(test_handle_a, 0u, 16u, Usage::READ)
            .add_buffer(test_handle_b, 0u, 16u, Usage::READ);
        auto plan = launcher.plan();
        expect(plan.ok()) << plan.error.c_str();
        // The first positional argument takes the first canonical slot: (0,1).
        expect(plan.arguments[0].buffer.handle == test_handle_a);
        expect(plan.arguments[1].buffer.handle == test_handle_b);
    };

    "uniform_packing_and_offsets"_test = [] {
        auto bindings = luisa::vector<NativeShaderResourceBinding>{};
        bindings.emplace_back(make_binding(0u, 0u, NativeShaderResourceKind::StructuredBuffer));
        NativeShaderLauncher launcher{test_shader, uint3{1u, 1u, 1u}, bindings};
        launcher.add_buffer(test_handle_a, 0u, 16u, Usage::READ);
        auto scale = 2.5f;
        launcher.add_uniform(scale);
        auto flag = 7u;
        launcher.add_uniform(flag);
        auto plan = launcher.plan();
        expect(plan.ok()) << plan.error.c_str();
        expect(plan.arguments.size() == 3u);
        expect(plan.arguments[1].tag == Argument::Tag::UNIFORM);
        expect(plan.arguments[2].tag == Argument::Tag::UNIFORM);
        auto cmd = std::move(launcher).build(uint3{64u, 1u, 1u});
        expect(cmd->arguments().size() == 3u);
        // Uniform offsets are shifted past the argument header, and the payload
        // is reachable through the base helper.
        auto scale_value = 0.0f;
        std::memcpy(&scale_value, cmd->uniform(cmd->arguments()[1].uniform).data(),
                    sizeof(float));
        expect(scale_value == 2.5f);
        auto flag_value = 0u;
        std::memcpy(&flag_value, cmd->uniform(cmd->arguments()[2].uniform).data(),
                    sizeof(uint));
        expect(flag_value == 7u);
    };

    "usage_contract_is_cross_checked"_test = [] {
        // R6: a UAV bound as READ is rejected unless the override is allowed.
        auto bindings = luisa::vector<NativeShaderResourceBinding>{};
        bindings.emplace_back(make_binding(0u, 0u, NativeShaderResourceKind::RWStructuredBuffer));
        {
            NativeShaderLauncher launcher{test_shader, uint3{1u, 1u, 1u}, bindings};
            launcher.add_buffer(test_handle_a, 0u, 16u, Usage::READ);
            auto error = launcher.validate();
            expect(!error.empty());
            expect(error.find("read-only") != luisa::string::npos) << error.c_str();
        }
        {
            NativeShaderLauncher launcher{test_shader, uint3{1u, 1u, 1u}, bindings};
            launcher.set_allow_usage_override(true)
                .add_buffer(test_handle_a, 0u, 16u, Usage::READ);
            expect(launcher.validate().empty());
        }
        {
            NativeShaderLauncher launcher{test_shader, uint3{1u, 1u, 1u}, bindings};
            launcher.add_buffer(test_handle_a, 0u, 16u, Usage::READ_WRITE);
            expect(launcher.validate().empty());
        }
        // An SRV bound with a write-capable usage is always rejected.
        auto srv = luisa::vector<NativeShaderResourceBinding>{};
        srv.emplace_back(make_binding(0u, 0u, NativeShaderResourceKind::StructuredBuffer));
        {
            NativeShaderLauncher launcher{test_shader, uint3{1u, 1u, 1u}, srv};
            launcher.set_allow_usage_override(true)
                .add_buffer(test_handle_a, 0u, 16u, Usage::WRITE);
            auto error = launcher.validate();
            expect(!error.empty());
            expect(error.find("read-only") != luisa::string::npos) << error.c_str();
        }
        // Usage::NONE is never a valid declaration.
        {
            NativeShaderLauncher launcher{test_shader, uint3{1u, 1u, 1u}, srv};
            launcher.add_buffer(test_handle_a, 0u, 16u, Usage::NONE);
            expect(launcher.validate().find("Usage::NONE") != luisa::string::npos);
        }
    };

    "same_register_across_namespaces"_test = [] {
        // HLSL: `register(t0)` and `register(u0)` both report (space 0,
        // register 0); the declared usage picks the right one.
        auto bindings = luisa::vector<NativeShaderResourceBinding>{};
        bindings.emplace_back(make_binding(0u, 0u, NativeShaderResourceKind::StructuredBuffer));
        bindings.emplace_back(make_binding(0u, 0u, NativeShaderResourceKind::RWStructuredBuffer));
        NativeShaderLauncher launcher{test_shader, uint3{1u, 1u, 1u}, bindings};
        launcher.add_buffer(0u, 0u, test_handle_a, 0u, 16u, Usage::READ)
            .add_buffer(0u, 0u, test_handle_b, 0u, 16u, Usage::READ_WRITE);
        auto plan = launcher.plan();
        expect(plan.ok()) << plan.error.c_str();
        // Canonical order puts the SRV (kind 2) before the UAV (kind 3).
        expect(plan.arguments[0].buffer.handle == test_handle_a);
        expect(plan.arguments[1].buffer.handle == test_handle_b);
    };

    "missing_duplicate_and_null_arguments_are_rejected"_test = [] {
        auto bindings = luisa::vector<NativeShaderResourceBinding>{};
        bindings.emplace_back(make_binding(0u, 0u, NativeShaderResourceKind::StructuredBuffer));
        bindings.emplace_back(make_binding(0u, 1u, NativeShaderResourceKind::StructuredBuffer));
        // Missing second binding.
        {
            NativeShaderLauncher launcher{test_shader, uint3{1u, 1u, 1u}, bindings};
            launcher.add_buffer(test_handle_a, 0u, 16u, Usage::READ);
            expect(launcher.validate().find("was not supplied") != luisa::string::npos);
        }
        // Duplicate binding.
        {
            NativeShaderLauncher launcher{test_shader, uint3{1u, 1u, 1u}, bindings};
            launcher.add_buffer(0u, 0u, test_handle_a, 0u, 16u, Usage::READ)
                .add_buffer(0u, 0u, test_handle_b, 0u, 16u, Usage::READ);
            expect(launcher.validate().find("more than once") != luisa::string::npos);
        }
        // Unknown explicit binding.
        {
            NativeShaderLauncher launcher{test_shader, uint3{1u, 1u, 1u}, bindings};
            launcher.add_buffer(9u, 9u, test_handle_a, 0u, 16u, Usage::READ);
            expect(launcher.validate().find("no resource at register") !=
                   luisa::string::npos);
        }
        // Null handle.
        {
            NativeShaderLauncher launcher{test_shader, uint3{1u, 1u, 1u}, bindings};
            launcher.add_buffer(0u, 0u, invalid_resource_handle, 0u, 16u, Usage::READ);
            expect(launcher.validate().find("null handle") != luisa::string::npos);
        }
        // Too many positional arguments.
        {
            NativeShaderLauncher launcher{test_shader, uint3{1u, 1u, 1u}, bindings};
            launcher.add_buffer(test_handle_a, 0u, 16u, Usage::READ)
                .add_buffer(test_handle_b, 0u, 16u, Usage::READ)
                .add_buffer(0x6000u, 0u, 16u, Usage::READ);
            expect(launcher.validate().find("more resource arguments") !=
                   luisa::string::npos);
        }
    };

    "traverse_arguments_reports_declared_usages"_test = [] {
        NativeShaderLauncher launcher{test_shader, uint3{1u, 1u, 1u}, make_bindings()};
        launcher.add_buffer(0u, 0u, test_handle_a, 0u, 255u, Usage::READ)
            .add_buffer(3u, 0u, test_handle_b, 32u, 128u, Usage::WRITE)
            .add_texture(1u, 0u, 0x3000u, 2u, Usage::READ)
            .add_buffer(2u, 0u, 0x4000u, 0u, 64u, Usage::READ)
            .add_uniform(1.0f);
        auto cmd = std::move(launcher).build(uint3{64u, 1u, 1u});
        auto visited = 0u;
        auto reads = 0u;
        auto writes = 0u;
        auto textures = 0u;
        auto uniforms = 0u;
        cmd->traverse_arguments([&]<typename T>(T const &arg, Usage usage) noexcept {
            visited++;
            if (((luisa::to_underlying(usage) &
                  luisa::to_underlying(Usage::WRITE)) != 0u)) {
                writes++;
            } else {
                reads++;
            }
            if constexpr (std::is_same_v<T, Argument::Texture>) {
                textures++;
                expect(arg.level == 2u);
            } else if constexpr (std::is_same_v<T, Argument::Buffer>) {
                static_cast<void>(arg);
            } else {
                uniforms++;
            }
        });
        expect(visited == 4u);
        expect(reads == 3u);
        expect(writes == 1u);
        expect(textures == 1u);
        expect(uniforms == 0u);
    };

    "native_shader_raii_destroys_once"_test = [] {
        // The RAII owner must call destroy_shader exactly once; the fake
        // extension records the calls.
        struct FakeExt final : NativeShaderExt {
            int destroy_count{0};
            uint64_t destroyed{invalid_resource_handle};
            FakeExt() noexcept : NativeShaderExt{nullptr} {}
            NativeShaderCompileResult compile(const NativeShaderCompileInfo &) noexcept override {
                return {};
            }
            NativeShaderMetadata load(const NativeShaderCompileResult &,
                                      luisa::span<const Usage>) noexcept override {
                return {};
            }
            void destroy_shader(uint64_t handle) noexcept override {
                destroy_count++;
                destroyed = handle;
            }
        } ext;
        {
            NativeShaderMetadata meta;
            meta.handle = 0xabcull;
            meta.block_size = uint3{8u, 1u, 1u};
            NativeShader shader{ext, meta};
            expect(static_cast<bool>(shader));
            expect(shader.handle() == 0xabcull);
            expect(shader.block_size().x == 8u);
            auto moved = std::move(shader);
            expect(!static_cast<bool>(shader));
            expect(static_cast<bool>(moved));
            expect(ext.destroy_count == 0);
        }
        expect(ext.destroy_count == 1);
        expect(ext.destroyed == 0xabcull);
        // A default-constructed (invalid) shader destroys nothing.
        { NativeShader invalid; }
        expect(ext.destroy_count == 1);
    };

    return 0;
}
