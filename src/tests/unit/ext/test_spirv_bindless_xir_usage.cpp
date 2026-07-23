// Test for exact-XIR bindless resource usage at the native SPIR-V ABI boundary.
// This test covers:
// - XIR bindless usage that is absent from the ABI-identical AST function
// - unbounded buffer/texture heaps and bindless buffer-view metadata
// - Vulkan 1.2 validation of the resulting native SPIR-V module

#include "ut/ut.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>

#include <spirv-tools/libspirv.hpp>

#include <luisa/dsl/sugar.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/verifier.h>

#include "spirv_codegen/bindless_usage.h"
#include "spirv_codegen/entry.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void set_environment_variable(const char *name,
                              const char *value) noexcept {
#ifdef _WIN32
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

class ScopedEnvironmentVariable {
private:
    const char *_name;
    std::optional<std::string> _previous;

public:
    ScopedEnvironmentVariable(const char *name,
                              const char *value) noexcept
        : _name{name} {
        if (auto previous = std::getenv(name)) {
            _previous.emplace(previous);
        }
        set_environment_variable(name, value);
    }
    ~ScopedEnvironmentVariable() noexcept {
        set_environment_variable(
            _name, _previous ? _previous->c_str() : nullptr);
    }
    ScopedEnvironmentVariable(
        const ScopedEnvironmentVariable &) = delete;
    ScopedEnvironmentVariable &operator=(
        const ScopedEnvironmentVariable &) = delete;
};

[[nodiscard]] size_t count_properties(
    luisa::span<const lc::spirv::Property> properties,
    lc::spirv::ShaderVariableType type,
    uint32_t array_size) noexcept {
    return static_cast<size_t>(std::ranges::count_if(
        properties, [=](auto &&property) noexcept {
            return property.type == type &&
                   property.array_size == array_size;
        }));
}

[[nodiscard]] const lc::spirv::Property *find_property(
    luisa::span<const lc::spirv::Property> properties,
    lc::spirv::ShaderVariableType type,
    uint32_t array_size) noexcept {
    auto iter = std::ranges::find_if(
        properties, [=](auto &&property) noexcept {
            return property.type == type &&
                   property.array_size == array_size;
        });
    return iter == properties.end() ? nullptr : &*iter;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_bindless_xir_usage_classifies_metadata_only_buffer_queries"_test = [] {
        auto buffer_size = lc::spirv::spirv_bindless_resource_usage(
            ResourceQueryOp::BINDLESS_BUFFER_SIZE);
        auto byte_buffer_size = lc::spirv::spirv_bindless_resource_usage(
            ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE);
        expect(!buffer_size.buffer_heap);
        expect(buffer_size.buffer_metadata);
        expect(!byte_buffer_size.buffer_heap);
        expect(byte_buffer_size.buffer_metadata);
        expect(!buffer_size.texture_2d && !buffer_size.texture_3d);
        expect(!byte_buffer_size.texture_2d &&
               !byte_buffer_size.texture_3d);
    };

    "spirv_bindless_xir_usage_classifies_every_supported_resource_domain"_test = [] {
        auto expect_domain = [](lc::spirv::SpirvBindlessResourceUsage usage,
                                bool buffer_heap, bool buffer_metadata,
                                bool texture_2d,
                                bool texture_3d) noexcept {
            expect(usage.buffer_heap == buffer_heap);
            expect(usage.buffer_metadata == buffer_metadata);
            expect(usage.texture_2d == texture_2d);
            expect(usage.texture_3d == texture_3d);
        };
        constexpr std::array buffer_queries{
            ResourceQueryOp::BINDLESS_BUFFER_SIZE,
            ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE,
            ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS};
        constexpr std::array texture_2d_queries{
            ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE,
            ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL,
            ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE,
            ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL,
            ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD,
            ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL,
            ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER,
            ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER,
            ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER,
            ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER};
        constexpr std::array texture_3d_queries{
            ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE,
            ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL,
            ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE,
            ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL,
            ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD,
            ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL,
            ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER,
            ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER,
            ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER,
            ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER};
        for (auto op : buffer_queries) {
            expect_domain(
                lc::spirv::spirv_bindless_resource_usage(op),
                false, true, false, false);
        }
        for (auto op : texture_2d_queries) {
            expect_domain(
                lc::spirv::spirv_bindless_resource_usage(op),
                false, false, true, false);
        }
        for (auto op : texture_3d_queries) {
            expect_domain(
                lc::spirv::spirv_bindless_resource_usage(op),
                false, false, false, true);
        }

        constexpr std::array buffer_reads{
            ResourceReadOp::BINDLESS_BUFFER_READ,
            ResourceReadOp::BINDLESS_BYTE_BUFFER_READ};
        constexpr std::array texture_2d_reads{
            ResourceReadOp::BINDLESS_TEXTURE2D_READ,
            ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL};
        constexpr std::array texture_3d_reads{
            ResourceReadOp::BINDLESS_TEXTURE3D_READ,
            ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL};
        for (auto op : buffer_reads) {
            expect_domain(
                lc::spirv::spirv_bindless_resource_usage(op),
                true, true, false, false);
        }
        for (auto op : texture_2d_reads) {
            expect_domain(
                lc::spirv::spirv_bindless_resource_usage(op),
                false, false, true, false);
        }
        for (auto op : texture_3d_reads) {
            expect_domain(
                lc::spirv::spirv_bindless_resource_usage(op),
                false, false, false, true);
        }

        constexpr std::array buffer_writes{
            ResourceWriteOp::BINDLESS_BUFFER_WRITE,
            ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE};
        for (auto op : buffer_writes) {
            expect_domain(
                lc::spirv::spirv_bindless_resource_usage(op),
                true, true, false, false);
        }

        expect_domain(lc::spirv::spirv_bindless_resource_usage(
                          ResourceQueryOp::TEXTURE2D_SAMPLE),
                      false, false, false, false);
        expect_domain(lc::spirv::spirv_bindless_resource_usage(
                          ResourceReadOp::BUFFER_READ),
                      false, false, false, false);
        expect_domain(lc::spirv::spirv_bindless_resource_usage(
                          ResourceWriteOp::TEXTURE2D_WRITE),
                      false, false, false, false);
    };

    "spirv_bindless_size_only_uses_exact_per_argument_local_metadata"_test = [] {
        Kernel1D ast_kernel = [](BindlessVar, BindlessVar) noexcept {};
        auto ast_function = ast_kernel.function()->function();

        Module module;
        auto *xir_kernel = module.create_kernel();
        xir_kernel->set_block_size(ast_function.block_size());
        static_cast<void>(xir_kernel->create_resource_argument(
            Type::from("bindless_array")));
        auto *used_bindless = xir_kernel->create_resource_argument(
            Type::from("bindless_array"));
        auto *zero = module.create_constant_zero(Type::of<uint32_t>());
        XIRBuilder builder;
        builder.set_insertion_point(xir_kernel->create_body_block());
        static_cast<void>(builder.call(
            Type::of<uint32_t>(),
            ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE,
            {used_bindless, zero}));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        auto result = lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
            ast_function, &module,
            ShaderOption{.enable_cache = false}, {});

        expect(!result.useBufferBindless);
        expect(!result.useTex2DBindless);
        expect(!result.useTex3DBindless);
        expect(eq(result.required_target_features, 0u));
        constexpr auto unbounded =
            std::numeric_limits<uint32_t>::max();
        auto properties = luisa::span{
            result.properties.data(), result.properties.size()};
        expect(eq(count_properties(
                      properties,
                      lc::spirv::ShaderVariableType::SRVBufferHeap,
                      unbounded),
                  0u));
        expect(eq(count_properties(
                      properties,
                      lc::spirv::ShaderVariableType::StructuredBuffer,
                      1u),
                  2u));
        expect(eq(count_properties(
                      properties,
                      lc::spirv::ShaderVariableType::SPIRVBindlessBufferMetadata,
                      1u),
                  1u));
        auto *metadata = find_property(
            properties,
            lc::spirv::ShaderVariableType::SPIRVBindlessBufferMetadata,
            1u);
        expect(metadata != nullptr &&
               metadata->space_index == 0u &&
               metadata->register_index == 2u);

        spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
        expect(tools.Validate(
            result.spv_bin.data(), result.spv_bin.size()));
    };

    "spirv_bindless_device_address_uses_metadata_without_buffer_heap"_test = [] {
        Kernel1D ast_kernel = [](BindlessVar) noexcept {};
        auto ast_function = ast_kernel.function()->function();

        Module module;
        auto *xir_kernel = module.create_kernel();
        xir_kernel->set_block_size(ast_function.block_size());
        auto *bindless = xir_kernel->create_resource_argument(
            Type::from("bindless_array"));
        auto *zero = module.create_constant_zero(Type::of<uint32_t>());
        XIRBuilder builder;
        builder.set_insertion_point(xir_kernel->create_body_block());
        static_cast<void>(builder.call(
            Type::of<uint64_t>(),
            ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS,
            {bindless, zero}));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        constexpr auto required =
            lc::spirv::target_feature::buffer_device_address |
            lc::spirv::target_feature::shader_int64;
        auto result = lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
            ast_function, &module,
            ShaderOption{.enable_cache = false},
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(required));

        expect(!result.useBufferBindless)
            << "address metadata does not require the unbounded buffer heap";
        expect(eq(result.required_target_features, required));
        constexpr auto unbounded =
            std::numeric_limits<uint32_t>::max();
        auto properties = luisa::span{
            result.properties.data(), result.properties.size()};
        expect(eq(count_properties(
                      properties,
                      lc::spirv::ShaderVariableType::SRVBufferHeap,
                      unbounded),
                  0u));
        expect(eq(count_properties(
                      properties,
                      lc::spirv::ShaderVariableType::SPIRVBindlessBufferMetadata,
                      1u),
                  1u));

        spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
        expect(tools.Validate(
            result.spv_bin.data(), result.spv_bin.size()));
    };

    "spirv_bindless_exact_xir_usage_drives_descriptor_abi"_test = [] {
        Kernel1D ast_kernel = [](BindlessVar) noexcept {};
        auto ast_function = ast_kernel.function()->function();

        Module module;
        auto *xir_kernel = module.create_kernel();
        xir_kernel->set_block_size(ast_function.block_size());
        auto *bindless = xir_kernel->create_resource_argument(
            Type::from("bindless_array"));
        auto *body = xir_kernel->create_body_block();
        auto *zero = module.create_constant_zero(Type::of<uint32_t>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        static_cast<void>(builder.call(
            Type::of<uint2>(),
            ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE,
            {bindless, zero}));
        static_cast<void>(builder.call(
            Type::of<uint32_t>(),
            ResourceReadOp::BINDLESS_BUFFER_READ,
            {bindless, zero, zero}));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded())
            << "the exact-XIR fixture must be generically valid";

        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        constexpr auto all_target_features =
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(
                lc::spirv::target_feature::known_mask);
        auto result = lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
            ast_function, &module,
            ShaderOption{.enable_cache = false},
            all_target_features);

        expect(result.useBufferBindless)
            << "the exact XIR buffer read must enable the buffer heap";
        expect(result.useTex2DBindless)
            << "the exact XIR texture-size query must enable the 2D heap";
        expect(!result.useTex3DBindless)
            << "no 3D bindless operation was emitted";

        constexpr auto unbounded =
            std::numeric_limits<uint32_t>::max();
        auto properties = luisa::span{
            result.properties.data(), result.properties.size()};
        expect(eq(count_properties(
                      properties,
                      lc::spirv::ShaderVariableType::SRVBufferHeap,
                      unbounded),
                  1u));
        expect(eq(count_properties(
                      properties,
                      lc::spirv::ShaderVariableType::SRVTextureHeap,
                      unbounded),
                  1u));
        expect(eq(count_properties(
                      properties,
                      lc::spirv::ShaderVariableType::SPIRVBindlessBufferMetadata,
                      1u),
                  1u));

        auto *buffer_heap = find_property(
            properties,
            lc::spirv::ShaderVariableType::SRVBufferHeap,
            unbounded);
        auto *texture_heap = find_property(
            properties,
            lc::spirv::ShaderVariableType::SRVTextureHeap,
            unbounded);
        auto *metadata = find_property(
            properties,
            lc::spirv::ShaderVariableType::SPIRVBindlessBufferMetadata,
            1u);
        expect(buffer_heap != nullptr &&
               buffer_heap->space_index == 2u &&
               buffer_heap->register_index == 0u);
        expect(texture_heap != nullptr &&
               texture_heap->space_index == 3u &&
               texture_heap->register_index == 0u);
        expect(metadata != nullptr &&
               metadata->space_index == 0u &&
               metadata->register_index == 1u);

        constexpr auto expected_features =
            lc::spirv::target_feature::sampled_image_array_dynamic_indexing |
            lc::spirv::target_feature::storage_buffer_array_dynamic_indexing |
            lc::spirv::target_feature::descriptor_indexing |
            lc::spirv::target_feature::runtime_descriptor_array |
            lc::spirv::target_feature::descriptor_binding_partially_bound |
            lc::spirv::target_feature::descriptor_binding_sampled_image_update_after_bind |
            lc::spirv::target_feature::descriptor_binding_storage_buffer_update_after_bind;
        expect(eq(result.required_target_features,
                  expected_features));

        spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
        expect(tools.Validate(
            result.spv_bin.data(), result.spv_bin.size()))
            << "exact-XIR bindless SPIR-V must validate for Vulkan 1.2";
    };
}
